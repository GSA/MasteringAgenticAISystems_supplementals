"""
Code Example 8.1.2: Custom Agent Monitoring Dashboard

Purpose: Demonstrate comprehensive monitoring for production AI agents

Concepts Demonstrated:
- Real-time metrics collection: Latency, cost, quality, errors
- SLO tracking: Define and monitor service level objectives
- Drift detection: Identify input distribution changes
- Alerting logic: Intelligent alert thresholds and severity levels
- NVIDIA GPU monitoring: DCGM metrics integration

Prerequisites:
- Understanding of Prometheus metrics
- Basic statistics (percentiles, averages)
- Familiarity with SRE principles

Author: NVIDIA Certified Generative AI LLM Course
Chapter: 8, Section: 8.1
Exam Skill: 8.1 - Define Monitoring Dashboards and Reliability Metrics
"""

# ============================================================================
# IMPORTS
# ============================================================================

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# METRICS COLLECTION
# ============================================================================

@dataclass
class AgentMetrics:
    """Container for agent performance metrics."""

    # Performance
    latency_ms: float
    timestamp: datetime

    # Quality
    quality_score: Optional[float] = None  # 0-5 scale
    user_feedback: Optional[bool] = None   # thumbs up/down

    # Cost
    input_tokens: int = 0
    output_tokens: int = 0

    # Reliability
    error: Optional[str] = None

    def cost_usd(self) -> float:
        """Calculate cost in USD (example pricing)."""
        input_cost = self.input_tokens * 0.000002   # $2 per 1M tokens
        output_cost = self.output_tokens * 0.000006  # $6 per 1M tokens
        return input_cost + output_cost


class MetricsCollector:
    """
    Real-time metrics collection for agent monitoring.

    Tracks:
    - Request latency (p50, p95, p99)
    - Error rates
    - Cost metrics
    - Quality scores
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            window_size (int): Number of recent requests to track
        """
        self.window_size = window_size

        # Sliding window of recent metrics
        self.recent_metrics: deque[AgentMetrics] = deque(maxlen=window_size)

        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.total_cost = 0.0

    def record(self, metrics: AgentMetrics):
        """Record a single request's metrics."""
        self.recent_metrics.append(metrics)
        self.total_requests += 1

        if metrics.error:
            self.total_errors += 1

        self.total_cost += metrics.cost_usd()

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        latencies = [m.latency_ms for m in self.recent_metrics]

        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0}

        latencies.sort()

        return {
            "p50": self._percentile(latencies, 0.50),
            "p95": self._percentile(latencies, 0.95),
            "p99": self._percentile(latencies, 0.99)
        }

    def get_error_rate(self) -> float:
        """Calculate error rate over recent window."""
        if not self.recent_metrics:
            return 0.0

        errors = sum(1 for m in self.recent_metrics if m.error)
        return errors / len(self.recent_metrics)

    def get_quality_score(self) -> Optional[float]:
        """Calculate average quality score."""
        scores = [m.quality_score for m in self.recent_metrics if m.quality_score]

        if not scores:
            return None

        return statistics.mean(scores)

    def get_cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if not self.recent_metrics:
            return 0.0

        return self.total_cost / len(self.recent_metrics)

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = f + 1

        if c >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


# ============================================================================
# SLO TRACKING
# ============================================================================

@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    metric: str  # "latency", "availability", "quality"
    threshold: float
    target_percentage: float  # e.g., 99.5 for 99.5%

    def error_budget(self) -> float:
        """Calculate remaining error budget."""
        return 1.0 - (self.target_percentage / 100.0)


class SLOTracker:
    """
    Track SLO compliance over time.

    Example SLOs:
    - 99.5% of requests complete in <2s
    - 99.9% availability (error rate <0.1%)
    - 95% of responses score >4/5 quality
    """

    def __init__(self):
        self.slos: Dict[str, SLO] = {}
        self.metrics_collector = MetricsCollector()

    def define_slo(
        self,
        name: str,
        metric: str,
        threshold: float,
        target_percentage: float
    ):
        """Define a new SLO."""
        slo = SLO(name, metric, threshold, target_percentage)
        self.slos[name] = slo
        logger.info(f"Defined SLO: {name}")

    def check_slo_compliance(self, slo_name: str) -> Dict[str, Any]:
        """
        Check if SLO is being met.

        Returns:
            Dict with compliance status and metrics
        """
        if slo_name not in self.slos:
            raise ValueError(f"Unknown SLO: {slo_name}")

        slo = self.slos[slo_name]

        # Calculate compliance based on metric type
        if slo.metric == "latency":
            percentiles = self.metrics_collector.get_latency_percentiles()
            actual_value = percentiles["p95"]
            compliant = actual_value < slo.threshold
            compliance_pct = sum(
                1 for m in self.metrics_collector.recent_metrics
                if m.latency_ms < slo.threshold
            ) / len(self.metrics_collector.recent_metrics) * 100

        elif slo.metric == "availability":
            error_rate = self.metrics_collector.get_error_rate()
            actual_value = (1 - error_rate) * 100
            compliant = actual_value >= slo.target_percentage
            compliance_pct = actual_value

        elif slo.metric == "quality":
            quality = self.metrics_collector.get_quality_score()
            actual_value = quality if quality else 0
            compliant = actual_value >= slo.threshold
            compliance_pct = sum(
                1 for m in self.metrics_collector.recent_metrics
                if m.quality_score and m.quality_score >= slo.threshold
            ) / len(self.metrics_collector.recent_metrics) * 100

        else:
            raise ValueError(f"Unknown metric type: {slo.metric}")

        # Calculate error budget consumption
        error_budget_remaining = (
            (compliance_pct - slo.target_percentage) /
            (100 - slo.target_percentage) * 100
        )

        return {
            "slo_name": slo_name,
            "compliant": compliant,
            "target": slo.target_percentage,
            "actual": compliance_pct,
            "error_budget_remaining_pct": max(0, error_budget_remaining),
            "metric_value": actual_value
        }


# ============================================================================
# DRIFT DETECTION
# ============================================================================

class DriftDetector:
    """
    Detect distribution drift in agent inputs.

    Uses KL-divergence to compare current distribution to baseline.
    """

    def __init__(self, sensitivity: float = 0.1):
        """
        Initialize drift detector.

        Args:
            sensitivity (float): Drift threshold (0-1, lower = more sensitive)
        """
        self.sensitivity = sensitivity
        self.baseline_distribution: Optional[Dict[str, float]] = None

    def set_baseline(self, samples: List[str]):
        """
        Establish baseline distribution from training samples.

        Args:
            samples (List[str]): Representative input samples
        """
        # Simplified: Track token distribution
        # Production: Use embeddings or more sophisticated features

        token_counts = {}
        total = 0

        for sample in samples:
            tokens = sample.lower().split()
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                total += 1

        # Normalize to probability distribution
        self.baseline_distribution = {
            token: count / total
            for token, count in token_counts.items()
        }

        logger.info(
            f"Baseline set with {len(self.baseline_distribution)} unique tokens"
        )

    def detect_drift(self, current_samples: List[str]) -> Dict[str, Any]:
        """
        Detect if current distribution has drifted from baseline.

        Args:
            current_samples (List[str]): Recent input samples

        Returns:
            Dict with drift detection results
        """
        if not self.baseline_distribution:
            raise ValueError("Must set baseline before detecting drift")

        # Calculate current distribution
        token_counts = {}
        total = 0

        for sample in current_samples:
            tokens = sample.lower().split()
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                total += 1

        current_distribution = {
            token: count / total
            for token, count in token_counts.items()
        }

        # Calculate KL-divergence (simplified)
        kl_divergence = 0.0

        for token in self.baseline_distribution:
            p = self.baseline_distribution[token]
            q = current_distribution.get(token, 1e-10)
            kl_divergence += p * (p / q if q > 0 else 0)

        drift_detected = kl_divergence > self.sensitivity

        return {
            "drift_detected": drift_detected,
            "kl_divergence": kl_divergence,
            "threshold": self.sensitivity,
            "new_tokens": len(
                set(current_distribution.keys()) -
                set(self.baseline_distribution.keys())
            )
        }


# ============================================================================
# ALERTING
# ============================================================================

@dataclass
class Alert:
    """Alert notification."""

    severity: str  # "critical", "warning", "info"
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metric_value: Optional[float] = None


class AlertManager:
    """
    Intelligent alerting based on metrics and SLOs.

    Alert severity levels:
    - Critical: SLO violation, immediate action required
    - Warning: Approaching SLO threshold
    - Info: Notable but not urgent
    """

    def __init__(self):
        self.alerts: List[Alert] = []

    def check_alerts(
        self,
        metrics: MetricsCollector,
        slo_tracker: SLOTracker
    ) -> List[Alert]:
        """
        Check all conditions and generate alerts.

        Args:
            metrics (MetricsCollector): Current metrics
            slo_tracker (SLOTracker): SLO compliance tracker

        Returns:
            List[Alert]: New alerts generated
        """
        new_alerts = []

        # Check latency
        percentiles = metrics.get_latency_percentiles()
        if percentiles["p95"] > 5000:  # 5 seconds
            new_alerts.append(Alert(
                severity="critical",
                title="High Latency",
                message=f"P95 latency {percentiles['p95']:.0f}ms exceeds 5s threshold",
                metric_value=percentiles["p95"]
            ))
        elif percentiles["p95"] > 3000:
            new_alerts.append(Alert(
                severity="warning",
                title="Elevated Latency",
                message=f"P95 latency {percentiles['p95']:.0f}ms above normal",
                metric_value=percentiles["p95"]
            ))

        # Check error rate
        error_rate = metrics.get_error_rate()
        if error_rate > 0.05:  # 5%
            new_alerts.append(Alert(
                severity="critical",
                title="High Error Rate",
                message=f"Error rate {error_rate*100:.1f}% exceeds 5% threshold",
                metric_value=error_rate
            ))
        elif error_rate > 0.01:
            new_alerts.append(Alert(
                severity="warning",
                title="Elevated Error Rate",
                message=f"Error rate {error_rate*100:.1f}% above normal",
                metric_value=error_rate
            ))

        # Check SLO compliance
        for slo_name in slo_tracker.slos:
            compliance = slo_tracker.check_slo_compliance(slo_name)

            if not compliance["compliant"]:
                new_alerts.append(Alert(
                    severity="critical",
                    title=f"SLO Violation: {slo_name}",
                    message=(
                        f"{slo_name} at {compliance['actual']:.1f}% "
                        f"(target: {compliance['target']:.1f}%)"
                    ),
                    metric_value=compliance["actual"]
                ))
            elif compliance["error_budget_remaining_pct"] < 20:
                new_alerts.append(Alert(
                    severity="warning",
                    title=f"Low Error Budget: {slo_name}",
                    message=(
                        f"Only {compliance['error_budget_remaining_pct']:.0f}% "
                        "error budget remaining"
                    )
                ))

        self.alerts.extend(new_alerts)

        return new_alerts


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def simulate_agent_monitoring():
    """Simulate agent monitoring over time."""
    print("\n" + "="*70)
    print("AI Agent Monitoring Dashboard Simulation")
    print("="*70)

    # Initialize components
    slo_tracker = SLOTracker()
    alert_manager = AlertManager()
    drift_detector = DriftDetector(sensitivity=0.15)

    # Define SLOs
    slo_tracker.define_slo(
        name="latency_slo",
        metric="latency",
        threshold=2000,  # 2 seconds
        target_percentage=95.0  # 95% of requests
    )

    slo_tracker.define_slo(
        name="availability_slo",
        metric="availability",
        threshold=0.0,
        target_percentage=99.5  # 99.5% uptime
    )

    # Set drift baseline
    baseline_samples = [
        "How do I deploy NVIDIA NIM?",
        "What are the system requirements?",
        "Explain inference optimization"
    ] * 10
    drift_detector.set_baseline(baseline_samples)

    # Simulate requests over time
    print("\nSimulating 100 agent requests...\n")

    for i in range(100):
        # Simulate varying performance
        if i > 80:  # Degrade performance near end
            latency = 3500 + (i - 80) * 100
            error = "Timeout" if i % 10 == 0 else None
        else:
            latency = 1500 + (i % 20) * 50
            error = None if i % 25 != 0 else "API Error"

        metrics = AgentMetrics(
            latency_ms=latency,
            timestamp=datetime.now(),
            quality_score=4.2 if not error else 2.0,
            input_tokens=500,
            output_tokens=200,
            error=error
        )

        slo_tracker.metrics_collector.record(metrics)

        # Check for alerts every 10 requests
        if (i + 1) % 10 == 0:
            alerts = alert_manager.check_alerts(
                slo_tracker.metrics_collector,
                slo_tracker
            )

            for alert in alerts:
                print(f"[{alert.severity.upper()}] {alert.title}: {alert.message}")

    # Display final metrics
    print("\n" + "="*70)
    print("MONITORING SUMMARY")
    print("="*70)

    percentiles = slo_tracker.metrics_collector.get_latency_percentiles()
    print(f"\nLatency:")
    print(f"  P50: {percentiles['p50']:.0f}ms")
    print(f"  P95: {percentiles['p95']:.0f}ms")
    print(f"  P99: {percentiles['p99']:.0f}ms")

    error_rate = slo_tracker.metrics_collector.get_error_rate()
    print(f"\nError Rate: {error_rate*100:.1f}%")

    quality = slo_tracker.metrics_collector.get_quality_score()
    print(f"Quality Score: {quality:.2f}/5.0")

    cost = slo_tracker.metrics_collector.get_cost_per_request()
    print(f"Cost/Request: ${cost:.4f}")

    # SLO compliance
    print("\n" + "-"*70)
    print("SLO COMPLIANCE")
    print("-"*70)

    for slo_name in slo_tracker.slos:
        compliance = slo_tracker.check_slo_compliance(slo_name)
        status = "✅ PASS" if compliance["compliant"] else "❌ FAIL"

        print(f"\n{slo_name}: {status}")
        print(f"  Target: {compliance['target']:.1f}%")
        print(f"  Actual: {compliance['actual']:.1f}%")
        print(
            f"  Error Budget: {compliance['error_budget_remaining_pct']:.0f}% remaining"
        )

    print("\n" + "="*70)


def main():
    """Run monitoring example."""
    simulate_agent_monitoring()
    print("\n✅ Monitoring simulation complete!")


if __name__ == "__main__":
    main()
