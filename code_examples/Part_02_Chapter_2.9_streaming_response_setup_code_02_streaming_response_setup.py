from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI(model="gpt-4", temperature=0.7, streaming=True)

async def generate_stream(question: str):
    """Async generator that yields tokens as they're produced"""
    # Simulate retrieval (still takes 2 seconds)
    await asyncio.sleep(2)
    context = "Return policy: 30 days for full refund..."

    # Stream tokens from LLM
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:  # Filter empty chunks
            # Format as SSE (Server-Sent Events)
            yield f"data: {chunk.content}\n\n"

@app.post("/ask/stream")
async def ask_stream(question: str):
    """Returns streaming response for progressive delivery"""
    return StreamingResponse(
        generate_stream(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
