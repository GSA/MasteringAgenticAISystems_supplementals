import asyncio

async def optimized_workflow():
    """Parallel execution of independent tool calls"""
    papers, related = await asyncio.gather(
        agent.arun("Search ArXiv for quantum computing papers"),
        agent.arun("Search ArXiv for related quantum entanglement work")
    )
    return papers, related

# Execute parallelized workflow
result = asyncio.run(optimized_workflow())
