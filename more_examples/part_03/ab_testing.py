"""
Code Example 3.1.2: A/B Testing Framework for Production Evaluation

Purpose: Safely evaluate new agent versions with real production traffic

Concepts Demonstrated:
- Traffic splitting (control vs treatment)
- Real-time metric collection
- Statistical significance testing
- Automatic rollback on degradation
- Gradual rollout strategy

Author: NVIDIA Agentic AI Certification
Chapter: 3, Section: 3.1
Exam Skill: 3.1 - Implement evaluation pipelines and task benchmarks
"""

import random
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    traffic_split: float = 0.1  # 10% to treatment
    min_sample_size: int = 1000
    significance_level: float = 0.05
    rollback_threshold: float = -0.05  # -5% TSR triggers rollback


@dataclass
class ABTestResults:
    """A/B test results"""
    control_tsr: float
    treatment_tsr: float
    control_samples: int
    treatment_samples: int
    p_value: float
    significant: bool
    decision: str  # "rollout", "rollback", "continue"


class ABTestingFramework:
    """
    A/B testing framework for production agent evaluation.

    Features:
    - Traffic splitting
    - Metric collection
    - Statistical significance testing
    - Automatic rollback
    - Gradual rollout
    """

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.metrics = {
            "control": defaultdict(int),
            "treatment": defaultdict(int)
        }

    def route_user(self) -> str:
        """Route user to control or treatment based on traffic split"""
        return "treatment" if random.random() < self.config.traffic_split else "control"

    def record_interaction(self, variant: str, success: bool):
        """Record interaction outcome"""
        self.metrics[variant]["total"] += 1
        if success:
            self.metrics[variant]["successes"] += 1

    def calculate_tsr(self, variant: str) -> float:
        """Calculate task success rate for variant"""
        total = self.metrics[variant]["total"]
        if total == 0:
            return 0.0
        return self.metrics[variant]["successes"] / total

    def calculate_p_value(self) -> float:
        """
        Calculate p-value using two-proportion z-test.

        H0: Treatment TSR = Control TSR
        H1: Treatment TSR ≠ Control TSR
        """
        control_successes = self.metrics["control"]["successes"]
        control_total = self.metrics["control"]["total"]
        treatment_successes = self.metrics["treatment"]["successes"]
        treatment_total = self.metrics["treatment"]["total"]

        if control_total == 0 or treatment_total == 0:
            return 1.0  # Not enough data

        # Pooled proportion
        p_pool = (control_successes + treatment_successes) / (control_total + treatment_total)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total))

        if se == 0:
            return 1.0

        # Z-score
        p_control = control_successes / control_total
        p_treatment = treatment_successes / treatment_total
        z = (p_treatment - p_control) / se

        # Two-tailed p-value (approximation)
        # In production: use scipy.stats.norm.sf
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF (for demo - use scipy in production)"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def analyze_results(self) -> ABTestResults:
        """Analyze A/B test results and make decision"""
        control_tsr = self.calculate_tsr("control")
        treatment_tsr = self.calculate_tsr("treatment")
        control_samples = self.metrics["control"]["total"]
        treatment_samples = self.metrics["treatment"]["total"]

        # Check minimum sample size
        if control_samples < self.config.min_sample_size or treatment_samples < self.config.min_sample_size:
            return ABTestResults(
                control_tsr=control_tsr,
                treatment_tsr=treatment_tsr,
                control_samples=control_samples,
                treatment_samples=treatment_samples,
                p_value=1.0,
                significant=False,
                decision="continue"
            )

        # Calculate statistical significance
        p_value = self.calculate_p_value()
        significant = p_value < self.config.significance_level

        # Make decision
        tsr_diff = treatment_tsr - control_tsr

        if tsr_diff < self.config.rollback_threshold:
            decision = "rollback"  # Significant degradation
        elif significant and tsr_diff > 0:
            decision = "rollout"  # Significant improvement
        else:
            decision = "continue"  # Keep testing

        return ABTestResults(
            control_tsr=control_tsr,
            treatment_tsr=treatment_tsr,
            control_samples=control_samples,
            treatment_samples=treatment_samples,
            p_value=p_value,
            significant=significant,
            decision=decision
        )

    def print_results(self, results: ABTestResults):
        """Print A/B test results"""
        print(f"\n{'='*60}")
        print(f"A/B Test Results: {self.config.test_name}")
        print(f"{'='*60}")
        print(f"\nControl Group:")
        print(f"  Task Success Rate: {results.control_tsr:.2%}")
        print(f"  Sample Size: {results.control_samples:,}")
        print(f"\nTreatment Group:")
        print(f"  Task Success Rate: {results.treatment_tsr:.2%}")
        print(f"  Sample Size: {results.treatment_samples:,}")
        print(f"\nStatistical Analysis:")
        print(f"  Difference: {(results.treatment_tsr - results.control_tsr):.2%}")
        print(f"  P-value: {results.p_value:.4f}")
        print(f"  Significant: {results.significant} (α={self.config.significance_level})")
        print(f"\n{'='*60}")

        if results.decision == "rollout":
            print("✅ DECISION: ROLL OUT to 100%")
            print("Treatment shows statistically significant improvement")
        elif results.decision == "rollback":
            print("❌ DECISION: ROLLBACK to control")
            print("Treatment shows significant degradation")
        else:
            print("⏸️  DECISION: CONTINUE TESTING")
            print("Not enough evidence to make decision yet")

        print(f"{'='*60}\n")


# Mock agents
class ControlAgent:
    def query(self, query: str) -> Tuple[str, bool]:
        """Process query and return (response, success)"""
        time.sleep(0.001)
        # Baseline: 80% success rate
        success = random.random() < 0.80
        return "Response from control agent", success


class TreatmentAgent:
    def query(self, query: str) -> Tuple[str, bool]:
        """Process query and return (response, success)"""
        time.sleep(0.0012)  # Slightly slower
        # Improved: 88% success rate (+8% improvement)
        success = random.random() < 0.88
        return "Response from treatment agent", success


# Main simulation
def simulate_ab_test(num_queries: int = 5000):
    """Simulate A/B test with production traffic"""
    print("\nA/B Testing Framework - Production Evaluation\n")

    config = ABTestConfig(
        test_name="GPT-4 vs GPT-3.5 for Customer Support",
        traffic_split=0.1,  # 10% to treatment
        min_sample_size=400,  # Minimum samples per group
        significance_level=0.05
    )

    framework = ABTestingFramework(config)
    control_agent = ControlAgent()
    treatment_agent = TreatmentAgent()

    print(f"Simulating {num_queries:,} production queries...")
    print(f"Traffic split: {config.traffic_split:.0%} treatment, {1-config.traffic_split:.0%} control\n")

    # Simulate production traffic
    for i in range(num_queries):
        variant = framework.route_user()

        if variant == "control":
            response, success = control_agent.query("user query")
        else:
            response, success = treatment_agent.query("user query")

        framework.record_interaction(variant, success)

        # Check results periodically
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1:,}/{num_queries:,} queries processed")

            # Analyze intermediate results
            results = framework.analyze_results()
            if results.decision != "continue":
                print(f"\n⚠️  Early stopping triggered after {i+1:,} queries")
                break

    # Final analysis
    print("\n\nFinal Analysis:")
    results = framework.analyze_results()
    framework.print_results(results)

    return results


def main():
    """Run A/B test simulation"""
    # Simulate A/B test
    results = simulate_ab_test(num_queries=5000)

    # Demonstrate gradual rollout strategy
    if results.decision == "rollout":
        print("\n" + "="*60)
        print("Gradual Rollout Strategy")
        print("="*60)
        print("\nRecommended rollout plan:")
        print("  Day 1: 10% traffic (current)")
        print("  Day 2: 25% traffic (if metrics stable)")
        print("  Day 3: 50% traffic (if metrics stable)")
        print("  Day 5: 100% traffic (full rollout)")
        print("\nMonitor metrics at each stage before proceeding.")
        print("="*60)


if __name__ == "__main__":
    main()
