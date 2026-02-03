async def generate_stream(question: str):
    try:
        # Retrieval and generation logic
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
    except Exception as e:
        # Send error as final SSE event
        yield f"data: [ERROR] {str(e)}\n\n"
        yield "event: error\ndata: Generation failed\n\n"
