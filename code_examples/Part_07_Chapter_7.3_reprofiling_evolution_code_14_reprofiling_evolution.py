# Re-profile after 30 days in production
profiler_v2 = AgentProfiler(production_agent)
current_metrics = profiler_v2.run_suite(test_cases)

# Identify new bottlenecks
new_bottlenecks = profiler_v2.compare_to_baseline(
    baseline="metrics/after_parallelization_2024_11_10.json",
    current=current_metrics
)
# Returns: [
#   {"component": "vector_search", "latency_increase": "+45%", "investigation_needed": true}
# ]
