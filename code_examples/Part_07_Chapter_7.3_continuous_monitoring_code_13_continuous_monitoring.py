# Production monitoring configuration
evaluator = Evaluator(
    agent=optimized_agent,
    test_cases=gold_standard_tests,
    schedule="hourly",
    alert_on_regression=True,
    thresholds={
        "accuracy": 0.85,
        "latency_p95": 3000,
        "cost_per_query": 0.015
    }
)

evaluator.start_continuous_monitoring()
