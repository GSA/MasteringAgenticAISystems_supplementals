"""
Independent Challenge: End-to-End Evaluation System for Travel Booking Agent

This module implements comprehensive evaluation combining offline testing,
staging validation, and production A/B testing. The architecture separates
three concerns: metrics computation (what to measure), comparison logic
(how to interpret measurements), and deployment decisions (when to rollout).

Design Philosophy:
- Offline evaluation catches obvious regressions quickly and cheaply
- Staging validation tests integration with near-production conditions
- A/B testing validates real-world impact with actual users
- Statistical rigor prevents false positives from random variation
- Automatic rollback limits blast radius when things go wrong

Concepts Applied from Prior Sections:
- Offline evaluation patterns from Worked Example 1 (Section 3.1.3)
- A/B testing framework from Worked Example 2 (Section 3.1.3)
- Custom metrics development from Guided Exercise 1 (Section 3.1.4)
- Statistical significance testing throughout
- MLflow integration for experiment tracking and reproducibility
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class BookingComplexity(Enum):
    """Categorize bookings by complexity to enable stratified analysis"""
    SIMPLE_ONEWAY = "simple_oneway"           # Single flight, one direction
    STANDARD_ROUNDTRIP = "standard_roundtrip" # Return trip, same carrier
    MULTILEG_DOMESTIC = "multileg_domestic"   # Multiple flights, same country
    INTERNATIONAL = "international"           # Cross-border, visa/passport considerations
    COMPLEX_MULTICITY = "complex_multicity"   # 3+ destinations, multiple carriers

@dataclass
class EvaluationMetrics:
    """
    Structured metrics from evaluation run.

    Separating metrics from evaluation logic enables easier testing,
    comparison across runs, and serialization to MLflow.
    """
    overall_tsr: float                           # Task Success Rate across all scenarios
    tsr_by_complexity: Dict[str, float]          # TSR segmented by booking type
    latency_p50: float                           # Median response time (milliseconds)
    latency_p95: float                           # 95th percentile (SLA threshold)
    latency_p99: float                           # 99th percentile (worst-case behavior)
    cost_per_success: float                      # Average API cost for successful bookings
    cost_by_component: Dict[str, float]          # Cost breakdown (flights, hotels, transfers)
    failure_modes: Dict[str, int]                # Count by failure type
    total_scenarios: int                         # Test dataset size
    evaluation_time_seconds: float               # Pipeline execution time

class TravelAgentEvaluator:
    """
    Comprehensive evaluation system for travel booking agents.

    Architecture separates three phases:
    1. Offline evaluation: Fast iteration with curated test datasets
    2. Staging validation: Integration testing with production-like environment
    3. A/B testing: Real user validation before full rollout

    Each phase builds on the previous, with stricter gates at each stage.
    """

    def __init__(self, test_dataset_path: str, mlflow_uri: str):
        """
        Initialize evaluator with test data and experiment tracking.

        Args:
            test_dataset_path: Path to JSON file containing test scenarios
                              with ground truth expected outcomes
            mlflow_uri: MLflow tracking server URI for experiment logging

        Implementation Notes:
        - Load and validate test dataset structure at initialization
        - Verify MLflow connectivity before running evaluations
        - Cache test scenarios in memory for fast repeated access
        - Consider stratifying dataset by complexity for balanced coverage
        """
        self.test_dataset_path = Path(test_dataset_path)
        self.mlflow_uri = mlflow_uri
        # TODO: Your initialization logic
        #   - Load test dataset from JSON
        #   - Parse into structured test cases
        #   - Validate required fields (query, expected_outcome, complexity)
        #   - Initialize MLflow client
        #   - Set up any caching infrastructure
        pass

    def run_offline_evaluation(self, agent_version: str) -> Dict[str, Any]:
        """
        Execute offline evaluation on curated test dataset.

        This represents the first quality gate: fast feedback on whether
        the agent performs correctly on known scenarios. Offline evaluation
        should complete in minutes, not hours, enabling rapid iteration.

        Args:
            agent_version: Identifier for the agent being tested (e.g., "v2.0")

        Returns:
            Dictionary containing:
            - metrics: EvaluationMetrics object with computed measurements
            - decision: "pass" or "fail" based on threshold comparison
            - details: Per-scenario results for debugging failures
            - baseline_comparison: Statistical comparison to previous version

        Implementation Strategy:
        - Iterate through test scenarios, running agent on each
        - Compute accuracy by comparing booking details to ground truth
        - Track latency per scenario with high-resolution timing
        - Accumulate costs from API calls and LLM token usage
        - Categorize failures by type (timeout, pricing mismatch, availability, etc.)
        - Calculate percentile statistics for latency distributions
        - Segment TSR by booking complexity to identify weak areas
        - Compare metrics to baseline using statistical significance tests

        Performance Considerations:
        - Run scenarios in parallel to meet <5 minute target
        - Use mocked APIs for deterministic testing (no real bookings)
        - Cache expensive computations where appropriate
        - Provide progress indication for long-running evaluations

        Statistical Rigor:
        - Don't just report means; include standard deviations
        - Calculate confidence intervals for key metrics
        - Test for statistical significance when comparing to baseline
        - Account for multiple comparisons (Bonferroni correction)
        """
        # TODO: Your implementation
        #   Suggested approach:
        #   1. Load baseline metrics from previous evaluation (if exists)
        #   2. Initialize metrics accumulator
        #   3. For each test scenario:
        #      a. Extract query and expected outcome
        #      b. Time agent execution
        #      c. Compare result to ground truth
        #      d. Track costs (API calls, tokens)
        #      e. Categorize any failures
        #   4. Compute aggregate statistics:
        #      - Overall TSR and TSR by complexity
        #      - Latency percentiles (P50, P95, P99)
        #      - Cost per successful booking
        #      - Failure mode distribution
        #   5. Compare to baseline with statistical tests
        #   6. Make pass/fail decision based on thresholds
        #   7. Log everything to MLflow
        #   8. Return comprehensive results
        pass

    def run_ab_test(
        self,
        control_agent: Any,
        treatment_agent: Any,
        traffic_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Execute A/B test comparing control vs treatment with production traffic.

        A/B testing validates that offline improvements translate to real-world
        gains with actual users under production conditions. This catches issues
        that offline evaluation misses: network variability, concurrent load,
        unexpected user behaviors, and integration edge cases.

        Args:
            control_agent: Current production agent (baseline)
            treatment_agent: New agent version being evaluated
            traffic_split: Fraction of traffic routed to treatment (default 0.1 = 10%)

        Returns:
            Dictionary containing:
            - control_metrics: Metrics observed for control group
            - treatment_metrics: Metrics observed for treatment group
            - statistical_tests: Results of significance testing
            - rollout_decision: "proceed", "rollback", or "insufficient_data"
            - confidence: Statistical confidence in observed differences
            - recommendation: Next steps (e.g., "increase to 25%", "rollback")

        Implementation Strategy:
        - Route incoming queries randomly to control or treatment based on split
        - Track identical metrics for both groups (TSR, latency, satisfaction, business)
        - Accumulate sufficient samples for statistical power
        - Monitor for significant degradation requiring automatic rollback
        - Compute statistical tests comparing groups (two-proportion z-test for TSR)
        - Calculate required additional samples if current data is inconclusive
        - Provide rollout recommendation based on results

        Statistical Requirements:
        - Minimum 1,000 samples per group for 80% power
        - Two-sided tests to catch both improvements and regressions
        - Bonferroni correction for multiple metrics
        - Continuous monitoring for early stopping if severe regression

        Rollback Criteria:
        - Treatment TSR drops >5% relative to control (p < 0.05)
        - P95 latency increases >50% (SLA violation)
        - User satisfaction decreases significantly
        - Any critical errors (data corruption, billing issues)

        Deployment Philosophy:
        - Start conservative (10% split) to limit blast radius
        - Ramp gradually (10% → 25% → 50% → 100%) with validation at each step
        - Require sustained improvement, not just initial gains
        - Automatic rollback beats manual intervention every time
        """
        # TODO: Your implementation
        #   Suggested approach:
        #   1. Validate traffic split parameter (0 < split < 1)
        #   2. Initialize separate metric accumulators for control and treatment
        #   3. Simulate production queries or use replay log:
        #      a. For each query, randomly assign to control or treatment
        #      b. Route to appropriate agent
        #      c. Record metrics (TSR, latency, user feedback, business)
        #      d. Check for rollback conditions continuously
        #   4. After sufficient samples collected:
        #      a. Compute aggregate metrics for both groups
        #      b. Run statistical significance tests:
        #         - Two-proportion z-test for TSR
        #         - T-test for latency distributions
        #         - Chi-square for failure mode distributions
        #      c. Calculate confidence intervals
        #      d. Apply Bonferroni correction for multiple comparisons
        #   5. Make rollout decision:
        #      - If significant improvement: "proceed" to next split level
        #      - If significant regression: "rollback" immediately
        #      - If inconclusive: "insufficient_data", calculate needed samples
        #   6. Log complete A/B test to MLflow with both groups' data
        #   7. Return results with actionable recommendation
        pass

    def compare_to_baseline(
        self,
        new_metrics: EvaluationMetrics,
        baseline_metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """
        Statistical comparison of new metrics against baseline.

        Raw metric differences can mislead: a 2% TSR improvement might be
        genuine progress or random noise depending on sample size. This method
        applies proper statistical testing to distinguish signal from noise.

        Args:
            new_metrics: Metrics from current evaluation
            baseline_metrics: Metrics from previous production version

        Returns:
            Dictionary containing:
            - differences: Absolute and percentage changes per metric
            - significance_tests: Statistical test results (p-values, effect sizes)
            - classification: "improvement", "regression", or "no_change" per metric
            - overall_decision: "approve", "reject", or "investigate"
            - confidence_intervals: Ranges for each metric difference

        Implementation Strategy:
        - Compare each metric individually with appropriate statistical test
        - For proportions (TSR): use two-proportion z-test
        - For continuous metrics (latency): use t-test or Mann-Whitney U
        - Calculate effect sizes (Cohen's d) to measure practical significance
        - Set minimum detectable effect thresholds (ignore tiny differences)
        - Classify each metric as improvement/regression/no_change
        - Make overall decision combining all metric classifications

        Thresholds:
        - TSR: Must not regress >2%, p < 0.05
        - Latency P95: Must not increase >20%, p < 0.05
        - Cost: Must not increase >10% unless TSR improves >5%
        - Allow trade-offs: higher cost acceptable if TSR improves significantly

        Multiple Comparison Correction:
        - Testing multiple metrics increases false positive risk
        - Apply Bonferroni correction: divide alpha by number of tests
        - Alternative: use Holm-Bonferroni for more power
        """
        # TODO: Your implementation
        #   Suggested approach:
        #   1. For each metric pair (new vs baseline):
        #      a. Calculate absolute difference
        #      b. Calculate percentage change
        #      c. Determine appropriate statistical test
        #      d. Compute p-value and effect size
        #      e. Compare against threshold
        #      f. Classify as improvement/regression/no_change
        #   2. Apply multiple comparison correction
        #   3. Check for acceptable trade-offs:
        #      - Higher cost OK if TSR improves
        #      - Higher latency OK if accuracy improves significantly
        #   4. Make overall decision:
        #      - "approve": no significant regressions, possible improvements
        #      - "reject": one or more critical regressions
        #      - "investigate": mixed results requiring human judgment
        #   5. Return detailed comparison with confidence intervals
        pass

# Testing and Validation

def test_evaluation_system():
    """
    Validation suite ensuring your evaluation system works correctly.

    Tests verify:
    - Offline evaluation completes successfully
    - Metrics computed correctly
    - Statistical tests function properly
    - A/B testing logic handles edge cases
    - MLflow integration logs complete data
    - Rollback triggers fire when appropriate

    Run these tests before considering your implementation complete.
    """
    evaluator = TravelAgentEvaluator("test_data.json", "http://localhost:5000")

    # Test 1: Offline evaluation basic functionality
    offline_results = evaluator.run_offline_evaluation("v2.0")
    assert "metrics" in offline_results
    assert "decision" in offline_results
    assert offline_results["decision"] in ["pass", "fail"]
    print("✅ Test 1 passed: Offline evaluation runs successfully")

    # Test 2: Metrics structure validation
    metrics = offline_results["metrics"]
    assert hasattr(metrics, "overall_tsr")
    assert hasattr(metrics, "tsr_by_complexity")
    assert 0.0 <= metrics.overall_tsr <= 1.0
    print("✅ Test 2 passed: Metrics structure valid")

    # Test 3: A/B testing with mock agents
    # (Your implementation should include mock agents for testing)
    # ab_results = evaluator.run_ab_test(control_v1, treatment_v2)
    # assert "statistical_tests" in ab_results
    # assert "rollout_decision" in ab_results
    # print("✅ Test 3 passed: A/B testing functions correctly")

    # Test 4: Statistical comparison
    # (Create two metrics objects with known differences)
    # comparison = evaluator.compare_to_baseline(new_metrics, baseline_metrics)
    # assert "overall_decision" in comparison
    # print("✅ Test 4 passed: Statistical comparison working")

    print("\n✅ All validation tests passed!")
    print("Your end-to-end evaluation system is ready for production use.")

if __name__ == "__main__":
    test_evaluation_system()
