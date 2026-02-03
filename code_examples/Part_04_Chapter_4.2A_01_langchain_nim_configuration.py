from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://llama-nim-service:8000/v1",
    model="meta/llama-3.1-70b-instruct",
    api_key="not-needed",  # NIM doesn't require API keys for internal access
    temperature=0.7,
    max_tokens=512
)
