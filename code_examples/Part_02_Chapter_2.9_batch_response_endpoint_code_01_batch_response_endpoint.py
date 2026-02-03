from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import time

app = FastAPI()
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

@app.post("/ask")
async def ask_batch(question: str):
    start = time.time()

    # Simulate retrieval delay
    await asyncio.sleep(2)
    context = "Return policy: 30 days for full refund..."

    # Call LLM - blocks until complete
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    elapsed = time.time() - start
    return {
        "answer": response.content,
        "latency": elapsed
    }
