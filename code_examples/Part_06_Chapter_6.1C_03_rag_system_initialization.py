from typing import List
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
    """
    Complete RAG implementation.

    Components:
    1. Document ingestion and chunking
    2. Embedding generation
    3. Vector storage (in-memory for simplicity)
    4. Retrieval with similarity search
    5. Answer generation with LLM
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix = None
