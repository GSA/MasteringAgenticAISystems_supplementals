# Stream responses for real-time display
stream = client.chat.completions.create(
    model="meta-llama-2-7b",
    messages=[{"role": "user", "content": "Write a Python function to calculate Fibonacci."}],
    temperature=0.5,
    max_tokens=512,
    stream=True  # Enable streaming
)

# Process streaming chunks
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
