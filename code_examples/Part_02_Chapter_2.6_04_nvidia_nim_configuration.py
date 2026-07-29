import os

from openai import OpenAI

# Standard OpenAI API; the client reads OPENAI_API_KEY from the environment.
openai_client = OpenAI(base_url="https://api.openai.com/v1")

# NVIDIA NIM endpoint (maintains OpenAI API compatibility).
# Set NIM_API_KEY and, when needed, NIM_BASE_URL in the environment.
nim_client = OpenAI(
    base_url=os.environ.get("NIM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ["NIM_API_KEY"],
)

# Function-calling code remains identical.
response = nim_client.chat.completions.create(
    model="meta/llama-3-70b-instruct",  # Select a model supported by your NIM deployment.
    messages=messages,
    tools=tool_schemas,
    tool_choice="auto",
)
