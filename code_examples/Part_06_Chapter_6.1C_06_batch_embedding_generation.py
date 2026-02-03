from typing import List
import numpy as np
from openai import OpenAI

client = OpenAI()

class SimpleRAGSystem:
    """RAG system with batch embedding generation."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.embeddings_matrix = None

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        # Batch processing for efficiency
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"  # Using small for cost efficiency
        )
        return [data.embedding for data in response.data]

    def _rebuild_embeddings_matrix(self):
        """Rebuild numpy matrix of embeddings for fast vector search."""
        embeddings = [chunk.embedding for chunk in self.chunks]
        self.embeddings_matrix = np.array(embeddings)
