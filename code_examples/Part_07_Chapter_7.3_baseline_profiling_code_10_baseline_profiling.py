# Initial profiling (pre-optimization)
profiler = AgentProfiler(agent)
baseline_metrics = profiler.run_suite(test_cases)

baseline_metrics.save("metrics/baseline_2024_11_10.json")
