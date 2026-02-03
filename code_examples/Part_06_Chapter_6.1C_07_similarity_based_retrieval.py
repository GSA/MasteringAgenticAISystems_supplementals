from typing import List
import numpy as np
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str
    source_doc: str
    chunk_index: int
    embedding: List[float] = None
    metadata: dict = None

class SimpleRAGSystem:
    """RAG system with similarity-based retrieval."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix = None

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve top-k most relevant chunks for query.

        Steps:
        1. Generate query embedding
        2. Compute cosine similarity with all chunks
        3. Return top-k chunks
        """
        print(f"\nRetrieving for query: {query}")

        # Step 1: Generate query embedding
        query_embedding = self._generate_embeddings([query])[0]
        query_vector = np.array(query_embedding)

        # Step 2: Compute cosine similarity
        # Normalize vectors for cosine similarity
        query_norm = query_vector / np.linalg.norm(query_vector)
        chunks_norm = self.embeddings_matrix / np.linalg.norm(
            self.embeddings_matrix, axis=1, keepdims=True
        )

        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(chunks_norm, query_norm)

        # Step 3: Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved_chunks = [self.chunks[idx] for idx in top_indices]

        print(f"  Retrieved {len(retrieved_chunks)} chunks")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"    [{i+1}] Similarity: {similarities[top_indices[i]]:.3f} | "
                  f"Source: {chunk.source_doc} | Chunk {chunk.chunk_index}")

        return retrieved_chunks

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        # Placeholder for embedding generation
        pass
