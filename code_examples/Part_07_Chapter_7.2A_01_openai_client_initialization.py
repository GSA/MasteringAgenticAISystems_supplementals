from openai import OpenAI
import os

# Initialize client with NIM endpoint
client = OpenAI(
    api_key=os.environ.get("NIM_API_KEY"),
    base_url="http://localhost:8000/v1"
)

# Send chat completion request (identical to OpenAI API)
response = client.chat.completions.create(
    model="meta-llama-2-7b",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is tensor parallelism?"}
    ],
    temperature=0.7,
    max_tokens=256
)

# Access response
print(response.choices[0].message.content)
