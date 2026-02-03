# Wrap with NeMo Agent Toolkit profiler
profiler = AgentProfiler(agent)

# Execute workflow with profiling enabled
result = profiler.run("Find the latest AI research papers and summarize the top result")

# Access granular profiling metrics
metrics = profiler.get_metrics()
print(metrics)
