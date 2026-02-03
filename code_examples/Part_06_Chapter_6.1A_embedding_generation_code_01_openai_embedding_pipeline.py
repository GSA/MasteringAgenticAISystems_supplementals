# Example: Generate embeddings using OpenAI
from openai import OpenAI

client = OpenAI()

def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text chunk."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Batch processing for efficiency
def generate_embeddings_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings in batches for efficiency."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )
        embeddings.extend([data.embedding for data in response.data])
    return embeddings
