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
    """RAG system with sliding window chunking."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_document(self, document: str, doc_id: str) -> List[DocumentChunk]:
        """
        Chunk document using sliding window with overlap.

        Simple tokenization: split by whitespace (production would use tiktoken)
        """
        tokens = document.split()
        chunks = []

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunk = DocumentChunk(
                text=chunk_text,
                source_doc=doc_id,
                chunk_index=chunk_idx,
                metadata={"num_tokens": len(chunk_tokens)}
            )
            chunks.append(chunk)

            # Slide window with overlap
            start += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1

            if end == len(tokens):
                break

        return chunks
