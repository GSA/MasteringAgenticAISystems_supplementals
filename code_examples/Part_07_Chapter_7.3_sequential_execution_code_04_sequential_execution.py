# Sequential execution (profiler baseline)
papers = agent.run("Search ArXiv for quantum computing papers")
related = agent.run("Search ArXiv for related quantum entanglement work")
# Total latency: 1234ms + 1234ms = 2468ms
