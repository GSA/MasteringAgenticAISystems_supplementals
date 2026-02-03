from fastapi import Request

async def generate_stream(question: str, request: Request):
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        # Check if client disconnected
        if await request.is_disconnected():
            break
        yield f"data: {chunk.content}\n\n"
