evaluator = Evaluator(
    agent=agent,
    test_cases=test_cases,
    accuracy_threshold=0.85,  # Alert if accuracy drops below 85%
    latency_threshold=3000,   # Alert if latency exceeds 3000ms
    cost_threshold=0.020      # Alert if cost exceeds $0.02 per query
)
