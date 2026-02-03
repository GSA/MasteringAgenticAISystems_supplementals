# Generate embeddings with NIM
response = client.embeddings.create(
    model="nvidia/embed-qa-4",
    input=[
        "NVIDIA NIM simplifies LLM deployment",
        "TensorRT optimizes inference performance"
    ]
)

# Access embeddings
embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
# Output: Generated 2 embeddings of dimension 768
