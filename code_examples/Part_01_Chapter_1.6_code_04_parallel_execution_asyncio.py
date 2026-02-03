# Parallel execution (fast)
import asyncio

async def search_all(sub_questions):
    # Create coroutines for all searches
    tasks = [search_api_async(q) for q in sub_questions]

    # Execute all concurrently and wait for all to complete
    results = await asyncio.gather(*tasks)
    return results

# Total time: 5 seconds for 5 questions (limited by slowest search)
