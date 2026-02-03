# Analyze bottlenecks
bottlenecks = profiler.identify_bottlenecks(threshold=0.30)
# Returns: [
#   {"component": "arxiv_search", "latency_pct": 53%, "optimization_potential": "HIGH"},
#   {"component": "summarize_paper", "latency_pct": 42%, "optimization_potential": "MEDIUM"}
# ]

# Prioritize by impact * feasibility
optimization_plan = profiler.recommend_optimizations(bottlenecks)
# Returns: [
#   {"strategy": "parallelize arxiv_search + database_query", "estimated_improvement": "40% latency"},
#   {"strategy": "cache arxiv_search results", "estimated_improvement": "62% cost"},
#   {"strategy": "reduce summarize_paper token usage", "estimated_improvement": "25% cost"}
# ]
