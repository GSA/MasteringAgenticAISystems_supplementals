# Standard OpenAI API
from openai import OpenAI

client = OpenAI(base_url="https://api.openai.com/v1")

# NVIDIA NIM endpoint (maintains OpenAI API compatibility)
client = OpenAI(
    base_url="https://your-nim-endpoint.nvidia.com/v1",
    api_key="your-nim-api-key"
)

# Function calling code remains identical
response = client.chat.completions.create(
    model="meta/llama-3-70b-instruct",  # NIM model
    messages=messages,
    tools=tool_schemas,
    tool_choice="auto"
)
