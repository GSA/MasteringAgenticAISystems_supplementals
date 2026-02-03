from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from fastapi import FastAPI

# Define a simple chain
llm = ChatOpenAI(model="gpt-4", streaming=True)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant."),
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()

# Create FastAPI app and add streaming routes
app = FastAPI(title="Support Agent API")
add_routes(
    app,
    chain,
    path="/agent",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)
