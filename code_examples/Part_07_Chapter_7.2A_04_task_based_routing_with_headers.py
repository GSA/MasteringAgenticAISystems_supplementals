response = client.chat.completions.create(
    model="mistral-7b",
    messages=[{"role": "user", "content": "Write a Python function..."}],
    extra_headers={"x-task-type": "code-generation"}
)
