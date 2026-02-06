import asyncio
from openai import AsyncOpenAI

# Initialize async client
client = AsyncOpenAI(
    api_key=os.environ.get("NIM_API_KEY"),
    base_url="http://nim.example.com/v1"
)

# Batch requests with concurrency
async def process_batch(requests):
    tasks = [
        client.chat.completions.create(
            model="meta-llama-2-7b",
            messages=[{"role": "user", "content": req}],
            temperature=0.7,
            max_tokens=256
        )
        for req in requests
    ]
    return await asyncio.gather(*tasks)

# Process 32 concurrent requests
requests = [f"Summarize topic {i}" for i in range(32)]
responses = asyncio.run(process_batch(requests))
