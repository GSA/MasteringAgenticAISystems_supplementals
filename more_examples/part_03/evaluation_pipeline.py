"""
Code Example 3.1.1: Offline Evaluation Pipeline with MLflow

Purpose: Automated evaluation pipeline for agent testing

Concepts Demonstrated:
- Offline evaluation with test datasets
- Multiple metric computation (accuracy, latency, cost)
- MLflow experiment tracking
- Baseline comparison and regression detection
- Automated pass/fail gates

Author: NVIDIA Agentic AI Certification
Chapter: 3, Section: 3.1
Exam Skill: 3.1 - Implement evaluation pipelines and task benchmarks
"""

import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import statistics

# Simulated MLflow (in production, use real MLflow)
class MLflowSimulator:
    def __init__(self):
        self.runs = []
        self.current_run = None

    def start_run(self, run_name):
        self.current_run = {"name": run_name, "metrics": {}, "params": {}}
        print(f"MLflow: Started run '{run_name}'")

    def log_metric(self, key, value):
        if self.current_run:
            self.current_run["metrics"][key] = value

    def log_param(self, key, value):
        if self.current_run:
            self.current_run["params"][key] = value

    def end_run(self):
        if self.current_run:
            self.runs.append(self.current_run)
            print(f"MLflow: Ended run '{self.current_run['name']}'")
            self.current_run = None

mlflow = MLflowSimulator()


@dataclass
class TestCase:
    """Single evaluation test case"""
    query: str
    ground_truth: str
    intent: str


@dataclass
class EvaluationMetrics:
    """Evaluation results"""
    accuracy: float
    latency_p50: float
    latency_p95: float
    cost_per_query: float
    pass_threshold: bool


class EvaluationPipeline:
    """
    Automated evaluation pipeline for agent testing.

    Features:
    - Load test dataset
    - Run agent on all test cases
    - Compute metrics
    - Compare to baseline
    - Pass/fail decision
    """

    def __init__(self, baseline_metrics: Dict[str, float] = None):
        self.baseline_metrics = baseline_metrics or {
            "accuracy": 0.92,
            "latency_p95": 2000,  # ms
            "cost_per_query": 0.015  # USD
        }

        # Thresholds for regression detection
        self.thresholds = {
            "accuracy_drop_max": 0.02,  # -2% max drop
            "latency_increase_max": 1.2,  # 20% max increase
            "cost_increase_max": 1.3  # 30% max increase
        }

    def load_test_dataset(self, dataset_path: str) -> List[TestCase]:
        """Load evaluation dataset"""
        # Simulated - in production, load from JSON/CSV
        return [
            TestCase("Where is my order?", "Track your order at...", "order_status"),
            TestCase("What sizes available?", "Sizes: S, M, L, XL", "product_info"),
            TestCase("Return policy?", "30-day return policy", "policy_question"),
            # ... 97 more test cases ...
        ] * 33  # Simulate 100 test cases

    def evaluate_agent(self, agent, test_dataset: List[TestCase]) -> EvaluationMetrics:
        """
        Run agent on test dataset and compute metrics.

        Returns: EvaluationMetrics with pass/fail decision
        """
        print(f"\n{'='*60}")
        print(f"Running Evaluation Pipeline")
        print(f"{'='*60}")
        print(f"Test Dataset: {len(test_dataset)} queries\n")

        correct = 0
        latencies = []
        total_cost = 0.0

        for i, test_case in enumerate(test_dataset):
            # Run agent
            start_time = time.time()
            response = agent.query(test_case.query)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            # Check accuracy
            if self._is_correct(response, test_case.ground_truth):
                correct += 1

            # Estimate cost (simplified)
            total_cost += self._estimate_cost(test_case.query, response)

            if (i + 1) % 25 == 0:
                print(f"Progress: {i+1}/{len(test_dataset)} queries evaluated")

        # Compute metrics
        accuracy = correct / len(test_dataset)
        latency_p50 = statistics.median(latencies)
        latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        cost_per_query = total_cost / len(test_dataset)

        # Compare to baseline
        pass_threshold = self._check_thresholds(
            accuracy, latency_p95, cost_per_query
        )

        metrics = EvaluationMetrics(
            accuracy=accuracy,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            cost_per_query=cost_per_query,
            pass_threshold=pass_threshold
        )

        self._print_results(metrics)

        return metrics

    def _is_correct(self, response: str, ground_truth: str) -> bool:
        """Check if response matches ground truth (simplified)"""
        # In production: use semantic similarity, LLM-as-judge, etc.
        return ground_truth.lower() in response.lower()

    def _estimate_cost(self, query: str, response: str) -> float:
        """Estimate cost based on token usage (simplified)"""
        # In production: track actual API costs
        tokens = len(query.split()) + len(response.split())
        cost_per_1k_tokens = 0.002
        return (tokens / 1000) * cost_per_1k_tokens

    def _check_thresholds(self, accuracy: float, latency_p95: float, cost: float) -> bool:
        """Check if metrics meet quality thresholds"""
        baseline_accuracy = self.baseline_metrics["accuracy"]
        baseline_latency = self.baseline_metrics["latency_p95"]
        baseline_cost = self.baseline_metrics["cost_per_query"]

        # Check for regressions
        accuracy_drop = baseline_accuracy - accuracy
        latency_ratio = latency_p95 / baseline_latency
        cost_ratio = cost / baseline_cost

        pass_accuracy = accuracy_drop <= self.thresholds["accuracy_drop_max"]
        pass_latency = latency_ratio <= self.thresholds["latency_increase_max"]
        pass_cost = cost_ratio <= self.thresholds["cost_increase_max"]

        return pass_accuracy and pass_latency and pass_cost

    def _print_results(self, metrics: EvaluationMetrics):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics.accuracy:.2%}")
        print(f"  Baseline: {self.baseline_metrics['accuracy']:.2%}")
        print(f"  Change: {(metrics.accuracy - self.baseline_metrics['accuracy']):.2%}")
        print()
        print(f"Latency P50: {metrics.latency_p50:.1f}ms")
        print(f"Latency P95: {metrics.latency_p95:.1f}ms")
        print(f"  Baseline P95: {self.baseline_metrics['latency_p95']:.1f}ms")
        print()
        print(f"Cost per Query: ${metrics.cost_per_query:.4f}")
        print(f"  Baseline: ${self.baseline_metrics['cost_per_query']:.4f}")
        print()
        print(f"{'='*60}")

        if metrics.pass_threshold:
            print("✅ EVALUATION PASSED - Safe to deploy")
        else:
            print("❌ EVALUATION FAILED - Deployment BLOCKED")

        print(f"{'='*60}\n")


# Mock agent for demo
class MockAgent:
    def __init__(self, version: str, performance_factor: float = 1.0):
        self.version = version
        self.performance_factor = performance_factor

    def query(self, query: str) -> str:
        # Simulate processing time
        time.sleep(0.001 * self.performance_factor)

        # Simulate response
        if "order" in query.lower():
            return "Track your order at tracking.example.com with your order number."
        elif "size" in query.lower():
            return "Our products come in sizes: Small, Medium, Large, and XL."
        elif "return" in query.lower():
            return "We have a 30-day return policy. Contact support to initiate a return."
        else:
            return "I can help you with that. Please provide more details."


# Main execution
def main():
    print("\nOffline Evaluation Pipeline with MLflow Integration\n")

    # Initialize pipeline
    pipeline = EvaluationPipeline()

    # Load test dataset
    test_dataset = pipeline.load_test_dataset("test_data.json")

    # Evaluate baseline agent
    print("\n--- Evaluating Baseline Agent (v1.0) ---")
    mlflow.start_run("baseline-v1.0")
    mlflow.log_param("agent_version", "v1.0")
    mlflow.log_param("model", "gpt-3.5-turbo")

    baseline_agent = MockAgent("v1.0", performance_factor=1.0)
    baseline_metrics = pipeline.evaluate_agent(baseline_agent, test_dataset)

    mlflow.log_metric("accuracy", baseline_metrics.accuracy)
    mlflow.log_metric("latency_p95", baseline_metrics.latency_p95)
    mlflow.log_metric("cost_per_query", baseline_metrics.cost_per_query)
    mlflow.end_run()

    # Evaluate new agent (better accuracy, slower)
    print("\n--- Evaluating New Agent (v2.0) ---")
    mlflow.start_run("new-v2.0")
    mlflow.log_param("agent_version", "v2.0")
    mlflow.log_param("model", "gpt-4")

    new_agent = MockAgent("v2.0", performance_factor=1.5)  # Slower
    new_metrics = pipeline.evaluate_agent(new_agent, test_dataset)

    mlflow.log_metric("accuracy", new_metrics.accuracy)
    mlflow.log_metric("latency_p95", new_metrics.latency_p95)
    mlflow.log_metric("cost_per_query", new_metrics.cost_per_query)
    mlflow.end_run()

    print(f"\n{'='*60}")
    print("All experiments logged to MLflow")
    print("Review results at: http://localhost:5000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
