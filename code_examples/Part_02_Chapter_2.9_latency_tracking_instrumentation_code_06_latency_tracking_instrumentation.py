import time
from contextlib import asynccontextmanager

@asynccontextmanager
async def track_latency(stage: str):
    """Context manager to track individual stage latency"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{stage} latency: {elapsed:.3f}s")

async def generate_stream(question: str):
    metrics = {}

    async with track_latency("retrieval") as m:
        context = await retrieve_documents(question)

    async with track_latency("ttft") as m:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            # First chunk marks TTFT
            if not metrics.get('ttft_recorded'):
                metrics['ttft_recorded'] = True
                logger.info(f"TTFT achieved: {m.elapsed:.3f}s")
            yield f"data: {chunk.content}\n\n"
