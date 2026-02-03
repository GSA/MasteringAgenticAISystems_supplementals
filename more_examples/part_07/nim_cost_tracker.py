#!/usr/bin/env python3
"""
NVIDIA NIM Cost Tracker
Track and analyze GPU inference costs with optimization recommendations
Version: 1.0
"""

import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, asdict
from prometheus_api_client import PrometheusConnect

@dataclass
class GPUCostConfig:
    """GPU pricing configuration."""
    name: str
    cost_per_hour: float


@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    timestamp: str
    gpu_hours: float
    total_cost: float
    requests: int
    cost_per_request: float
    avg_gpu_utilization: float
    wasted_cost: float


class NIMCostTracker:
    """
    Track and analyze NIM deployment costs.

    Features:
    - Real-time cost calculation
    - GPU utilization efficiency
    - Cost per request metrics
    - Optimization recommendations
    """

    # GPU pricing (example AWS pricing)
    GPU_COSTS = {
        "A100-40GB": GPUCostConfig("A100-40GB", 4.10),
        "A100-80GB": GPUCostConfig("A100-80GB", 5.50),
        "A10": GPUCostConfig("A10", 1.01),
        "L4": GPUCostConfig("L4", 0.80),
        "H100": GPUCostConfig("H100", 7.50),
        "V100": GPUCostConfig("V100", 3.06),
    }

    def __init__(
        self,
        prometheus_url: str = "http://prometheus.monitoring.svc:9090",
        namespace: str = "nim-production"
    ):
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        self.namespace = namespace

    def get_gpu_count(self) -> Dict[str, int]:
        """Get count of each GPU type."""
        query = f'count(nim_gpu_utilization_percent{{namespace="{self.namespace}"}}) by (gpu_model)'
        result = self.prometheus.custom_query(query=query)

        gpu_counts = {}
        for item in result:
            gpu_model = item['metric'].get('gpu_model', 'A100-40GB')
            count = int(float(item['value'][1]))
            gpu_counts[gpu_model] = count

        return gpu_counts

    def get_metrics(self, duration_hours: float = 1.0) -> CostMetrics:
        """Calculate cost metrics for specified duration."""

        # Get GPU counts and types
        gpu_counts = self.get_gpu_count()

        # Calculate GPU hours
        total_gpu_hours = sum(count * duration_hours for count in gpu_counts.values())

        # Calculate costs
        total_cost = 0.0
        for gpu_model, count in gpu_counts.items():
            cost_config = self.GPU_COSTS.get(gpu_model, self.GPU_COSTS["A100-40GB"])
            total_cost += cost_config.cost_per_hour * count * duration_hours

        # Get request count
        requests_query = f'sum(increase(nim_inference_requests_total{{namespace="{self.namespace}"}}[{int(duration_hours)}h]))'
        requests_result = self.prometheus.custom_query(query=requests_query)
        requests = int(float(requests_result[0]['value'][1])) if requests_result else 0

        # Calculate cost per request
        cost_per_request = total_cost / requests if requests > 0 else 0.0

        # Get average GPU utilization
        util_query = f'avg_over_time(avg(nim_gpu_utilization_percent{{namespace="{self.namespace}"}})[{int(duration_hours)}h:])'
        util_result = self.prometheus.custom_query(query=util_query)
        avg_utilization = float(util_result[0]['value'][1]) / 100 if util_result else 0.0

        # Calculate wasted cost (unused GPU capacity)
        wasted_cost = total_cost * (1 - avg_utilization)

        return CostMetrics(
            timestamp=datetime.utcnow().isoformat(),
            gpu_hours=total_gpu_hours,
            total_cost=round(total_cost, 4),
            requests=requests,
            cost_per_request=round(cost_per_request, 6),
            avg_gpu_utilization=round(avg_utilization * 100, 2),
            wasted_cost=round(wasted_cost, 4)
        )

    def generate_recommendations(self, metrics: CostMetrics) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []

        # Low GPU utilization
        if metrics.avg_gpu_utilization < 60:
            potential_savings = metrics.wasted_cost
            recommendations.append(
                f"⚠️ GPU utilization is {metrics.avg_gpu_utilization:.1f}% "
                f"(target: 70-85%). Potential savings: ${potential_savings:.2f}"
            )
            recommendations.append(
                "   → Consider reducing replica count or consolidating workloads"
            )

        # High GPU utilization
        elif metrics.avg_gpu_utilization > 90:
            recommendations.append(
                f"⚠️ GPU utilization is {metrics.avg_gpu_utilization:.1f}% "
                "(target: 70-85%)"
            )
            recommendations.append(
                "   → Consider adding more replicas to improve latency and reliability"
            )

        # High cost per request
        if metrics.cost_per_request > 0.001:  # $0.001 per request
            recommendations.append(
                f"💰 Cost per request is ${metrics.cost_per_request:.6f}"
            )
            recommendations.append(
                "   → Consider using smaller models or quantized versions"
            )

        # Optimization opportunities
        if metrics.avg_gpu_utilization > 70 and metrics.avg_gpu_utilization < 85:
            recommendations.append(
                "✅ GPU utilization is optimal (70-85%)"
            )

        return recommendations

    def print_report(self, metrics: CostMetrics, duration_hours: float):
        """Print formatted cost report."""
        print("\n" + "="*60)
        print("NVIDIA NIM Cost Report")
        print("="*60)
        print(f"Time Period: {duration_hours} hour(s)")
        print(f"Generated: {metrics.timestamp}")
        print()

        print("Cost Summary:")
        print(f"  Total GPU Hours: {metrics.gpu_hours:.2f}")
        print(f"  Total Cost: ${metrics.total_cost:.4f}")
        print(f"  Projected Daily: ${metrics.total_cost * (24 / duration_hours):.2f}")
        print(f"  Projected Monthly: ${metrics.total_cost * (24 * 30 / duration_hours):.2f}")
        print()

        print("Efficiency Metrics:")
        print(f"  Total Requests: {metrics.requests:,}")
        print(f"  Cost per Request: ${metrics.cost_per_request:.6f}")
        print(f"  Avg GPU Utilization: {metrics.avg_gpu_utilization:.1f}%")
        print(f"  Wasted Cost: ${metrics.wasted_cost:.4f}")
        print(f"  Efficiency: {100 - (metrics.wasted_cost / metrics.total_cost * 100):.1f}%")
        print()

        # Recommendations
        recommendations = self.generate_recommendations(metrics)
        if recommendations:
            print("Optimization Recommendations:")
            for rec in recommendations:
                print(rec)
            print()

        print("="*60)

    def track_continuous(self, interval_minutes: int = 15):
        """Continuously track costs."""
        print(f"Starting continuous cost tracking (interval: {interval_minutes}m)")

        while True:
            try:
                metrics = self.get_metrics(duration_hours=interval_minutes / 60)
                self.print_report(metrics, duration_hours=interval_minutes / 60)

                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                print("\nCost tracking stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="NVIDIA NIM Cost Tracker")
    parser.add_argument(
        "--prometheus-url",
        default="http://prometheus.monitoring.svc:9090",
        help="Prometheus URL"
    )
    parser.add_argument(
        "--namespace",
        default="nim-production",
        help="Kubernetes namespace"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Analysis duration in hours (default: 1)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Continuous tracking interval in minutes (default: 15)"
    )
    parser.add_argument(
        "--report-file",
        help="Save report to JSON file"
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = NIMCostTracker(
        prometheus_url=args.prometheus_url,
        namespace=args.namespace
    )

    if args.continuous:
        tracker.track_continuous(interval_minutes=args.interval)
    else:
        # Single report
        metrics = tracker.get_metrics(duration_hours=args.duration)
        tracker.print_report(metrics, duration_hours=args.duration)

        # Save to file if requested
        if args.report_file:
            with open(args.report_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
            print(f"\nReport saved to {args.report_file}")


if __name__ == "__main__":
    main()
