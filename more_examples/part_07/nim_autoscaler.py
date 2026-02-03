#!/usr/bin/env python3
"""
NVIDIA NIM GPU-Based Autoscaler
Intelligent autoscaling based on GPU utilization and inference metrics
Version: 1.0
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    avg_gpu_utilization: float
    p95_latency: float
    requests_per_second: float
    active_requests: int
    error_rate: float


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: str  # "scale_up", "scale_down", "no_action"
    target_replicas: int
    reason: str
    confidence: float


class NIMAutoscaler:
    """
    GPU-aware autoscaler for NVIDIA NIM deployments.

    Scales based on:
    - GPU utilization (primary signal)
    - Request latency (P95)
    - Throughput (requests/sec)
    - Error rate
    """

    def __init__(
        self,
        namespace: str = "nim-production",
        prometheus_url: str = "http://prometheus.monitoring.svc:9090",
        kubeconfig_path: str = None
    ):
        self.namespace = namespace
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)

        # Load Kubernetes configuration
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

        self.apps_v1 = client.AppsV1Api()

        # Scaling thresholds
        self.gpu_util_high = 85.0  # Scale up above 85%
        self.gpu_util_low = 50.0   # Scale down below 50%
        self.latency_threshold = 2.0  # P95 latency threshold (seconds)
        self.error_rate_threshold = 0.05  # 5% error rate

        # Scaling behavior
        self.min_replicas = 2
        self.max_replicas = 20
        self.scale_up_factor = 1.5  # Scale up by 50%
        self.scale_down_factor = 0.75  # Scale down by 25%
        self.cooldown_period = 300  # 5 minutes between scaling actions

        # State tracking
        self.last_scale_time: Dict[str, float] = {}

    def get_metrics(self, deployment: str) -> ScalingMetrics:
        """Fetch current metrics from Prometheus."""

        # GPU utilization
        gpu_util_query = f'avg(nim_gpu_utilization_percent{{namespace="{self.namespace}"}})'
        gpu_util_result = self.prometheus.custom_query(query=gpu_util_query)
        avg_gpu_util = float(gpu_util_result[0]['value'][1]) if gpu_util_result else 0.0

        # P95 latency
        latency_query = f'histogram_quantile(0.95, sum(rate(nim_inference_duration_seconds_bucket{{namespace="{self.namespace}"}}[5m])) by (le))'
        latency_result = self.prometheus.custom_query(query=latency_query)
        p95_latency = float(latency_result[0]['value'][1]) if latency_result else 0.0

        # Requests per second
        rps_query = f'sum(rate(nim_inference_requests_total{{namespace="{self.namespace}"}}[5m]))'
        rps_result = self.prometheus.custom_query(query=rps_query)
        rps = float(rps_result[0]['value'][1]) if rps_result else 0.0

        # Active requests
        active_query = f'sum(nim_inference_requests_in_progress{{namespace="{self.namespace}"}})'
        active_result = self.prometheus.custom_query(query=active_query)
        active_requests = int(float(active_result[0]['value'][1])) if active_result else 0

        # Error rate
        error_rate_query = f'sum(rate(nim_inference_errors_total{{namespace="{self.namespace}"}}[5m])) / sum(rate(nim_inference_requests_total{{namespace="{self.namespace}"}}[5m]))'
        error_rate_result = self.prometheus.custom_query(query=error_rate_query)
        error_rate = float(error_rate_result[0]['value'][1]) if error_rate_result else 0.0

        return ScalingMetrics(
            avg_gpu_utilization=avg_gpu_util,
            p95_latency=p95_latency,
            requests_per_second=rps,
            active_requests=active_requests,
            error_rate=error_rate
        )

    def get_current_replicas(self, deployment: str) -> int:
        """Get current replica count for deployment."""
        try:
            deploy = self.apps_v1.read_namespaced_deployment(
                name=deployment,
                namespace=self.namespace
            )
            return deploy.spec.replicas
        except client.ApiException as e:
            logger.error(f"Failed to get replicas for {deployment}: {e}")
            return 0

    def calculate_scaling_decision(
        self,
        deployment: str,
        current_replicas: int,
        metrics: ScalingMetrics
    ) -> ScalingDecision:
        """
        Calculate scaling decision based on metrics.

        Decision logic:
        1. Scale up if: GPU util > 85% OR (latency > 2s AND GPU util > 70%)
        2. Scale down if: GPU util < 50% AND latency < 1s AND error_rate < 1%
        3. Emergency scale up if: error_rate > 5%
        """

        # Check cooldown period
        last_scale = self.last_scale_time.get(deployment, 0)
        if time.time() - last_scale < self.cooldown_period:
            return ScalingDecision(
                action="no_action",
                target_replicas=current_replicas,
                reason="Cooldown period active",
                confidence=1.0
            )

        # Emergency scale up on high error rate
        if metrics.error_rate > self.error_rate_threshold:
            target = min(int(current_replicas * 2), self.max_replicas)
            return ScalingDecision(
                action="scale_up",
                target_replicas=target,
                reason=f"High error rate: {metrics.error_rate:.1%}",
                confidence=1.0
            )

        # Scale up conditions
        if metrics.avg_gpu_utilization > self.gpu_util_high:
            target = min(int(current_replicas * self.scale_up_factor), self.max_replicas)
            return ScalingDecision(
                action="scale_up",
                target_replicas=target,
                reason=f"GPU utilization high: {metrics.avg_gpu_utilization:.1f}%",
                confidence=0.9
            )

        if metrics.p95_latency > self.latency_threshold and metrics.avg_gpu_utilization > 70:
            target = min(int(current_replicas * self.scale_up_factor), self.max_replicas)
            return ScalingDecision(
                action="scale_up",
                target_replicas=target,
                reason=f"High latency ({metrics.p95_latency:.2f}s) and moderate GPU util",
                confidence=0.8
            )

        # Scale down conditions
        if (metrics.avg_gpu_utilization < self.gpu_util_low and
            metrics.p95_latency < 1.0 and
            metrics.error_rate < 0.01):

            target = max(int(current_replicas * self.scale_down_factor), self.min_replicas)
            if target < current_replicas:
                return ScalingDecision(
                    action="scale_down",
                    target_replicas=target,
                    reason=f"Low GPU utilization: {metrics.avg_gpu_utilization:.1f}%",
                    confidence=0.7
                )

        # No action needed
        return ScalingDecision(
            action="no_action",
            target_replicas=current_replicas,
            reason="Metrics within normal ranges",
            confidence=1.0
        )

    def apply_scaling(self, deployment: str, target_replicas: int) -> bool:
        """Apply scaling decision to deployment."""
        try:
            # Patch deployment with new replica count
            body = {
                "spec": {
                    "replicas": target_replicas
                }
            }

            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment,
                namespace=self.namespace,
                body=body
            )

            logger.info(f"Scaled {deployment} to {target_replicas} replicas")
            self.last_scale_time[deployment] = time.time()
            return True

        except client.ApiException as e:
            logger.error(f"Failed to scale {deployment}: {e}")
            return False

    def run_once(self, deployments: List[str]):
        """Run one iteration of autoscaling."""
        logger.info("=== Autoscaling Iteration ===")

        for deployment in deployments:
            logger.info(f"\nProcessing deployment: {deployment}")

            # Get current state
            current_replicas = self.get_current_replicas(deployment)
            if current_replicas == 0:
                logger.warning(f"Deployment {deployment} not found or has 0 replicas")
                continue

            # Fetch metrics
            metrics = self.get_metrics(deployment)
            logger.info(
                f"Metrics: GPU={metrics.avg_gpu_utilization:.1f}%, "
                f"P95={metrics.p95_latency:.2f}s, "
                f"RPS={metrics.requests_per_second:.1f}, "
                f"Errors={metrics.error_rate:.1%}"
            )

            # Calculate scaling decision
            decision = self.calculate_scaling_decision(
                deployment, current_replicas, metrics
            )

            logger.info(
                f"Decision: {decision.action} (confidence={decision.confidence:.1%}) "
                f"- {decision.reason}"
            )

            # Apply scaling if needed
            if decision.action == "scale_up" or decision.action == "scale_down":
                if decision.target_replicas != current_replicas:
                    logger.info(
                        f"Scaling {deployment}: {current_replicas} → {decision.target_replicas}"
                    )
                    self.apply_scaling(deployment, decision.target_replicas)
                else:
                    logger.info("Target replicas same as current, no action taken")

    def run_loop(self, deployments: List[str], interval: int = 60):
        """Run autoscaling loop continuously."""
        logger.info("Starting NIM Autoscaler")
        logger.info(f"Monitoring deployments: {', '.join(deployments)}")
        logger.info(f"Check interval: {interval}s")

        while True:
            try:
                self.run_once(deployments)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Autoscaler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in autoscaling loop: {e}", exc_info=True)
                time.sleep(interval)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NVIDIA NIM GPU-Based Autoscaler")
    parser.add_argument(
        "--namespace",
        default="nim-production",
        help="Kubernetes namespace (default: nim-production)"
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://prometheus.monitoring.svc:9090",
        help="Prometheus URL"
    )
    parser.add_argument(
        "--deployments",
        nargs="+",
        default=["nim-llama2-7b", "nim-mistral-7b"],
        help="List of deployments to autoscale"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)"
    )

    args = parser.parse_args()

    # Initialize autoscaler
    autoscaler = NIMAutoscaler(
        namespace=args.namespace,
        prometheus_url=args.prometheus_url
    )

    # Run autoscaler
    if args.once:
        autoscaler.run_once(args.deployments)
    else:
        autoscaler.run_loop(args.deployments, interval=args.interval)


if __name__ == "__main__":
    main()
