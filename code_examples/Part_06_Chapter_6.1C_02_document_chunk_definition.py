from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str              # Chunk content
    source_doc: str        # Origin document identifier
    chunk_index: int       # Position in source document
    embedding: List[float] = None  # Vector representation (1536-d for text-embedding-3-small)
    metadata: Dict[str, Any] = None  # Extensible metadata (timestamps, authors, sections)
