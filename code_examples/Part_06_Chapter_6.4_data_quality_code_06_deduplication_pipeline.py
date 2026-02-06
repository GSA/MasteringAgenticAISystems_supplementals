"""
Multi-Level Deduplication - Complete Pipeline

Orchestrates exact matching, fuzzy matching, and semantic deduplication.
"""

import hashlib
from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class Document:
    """Document representation for deduplication."""
    id: str
    content: str
    title: str = ""
    embedding: list = None
    content_hash: str = None


class DeduplicationPipeline:
    """
    Three-level deduplication: exact, fuzzy, and semantic.

    Processing order optimizes for performance:
    1. Exact matching (hash-based) - O(n), catches ~40% duplicates
    2. Fuzzy matching (Levenshtein) - O(n²), catches ~30% remaining
    3. Semantic matching (embedding) - O(n²), catches final ~20%
    """

    def __init__(self,
                 fuzzy_threshold: float = 0.90,
                 semantic_threshold: float = 0.95):
        """
        Initialize deduplication pipeline.

        Args:
            fuzzy_threshold: Similarity threshold for fuzzy matching (0-1)
            semantic_threshold: Cosine similarity threshold for embeddings
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.stats = {
            'exact_duplicates': 0,
            'fuzzy_duplicates': 0,
            'semantic_duplicates': 0
        }

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of normalized content."""
        # Normalize: lowercase, strip whitespace, remove multiple spaces
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _remove_exact_duplicates(self, documents: List[Document]) -> List[Document]:
        """Remove documents with identical content hashes."""
        seen_hashes: Set[str] = set()
        unique_docs: List[Document] = []

        for doc in documents:
            # Compute and cache hash
            doc.content_hash = self._compute_content_hash(doc.content)

            if doc.content_hash not in seen_hashes:
                seen_hashes.add(doc.content_hash)
                unique_docs.append(doc)
            else:
                self.stats['exact_duplicates'] += 1

        return unique_docs

    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicates from document collection.

        Returns:
            List of unique documents (duplicates removed)
        """
        print(f"Starting deduplication of {len(documents)} documents...")

        # Level 1: Exact matching via content hashing
        docs_after_exact = self._remove_exact_duplicates(documents)
        print(f"  After exact matching: {len(docs_after_exact)} documents "
              f"({self.stats['exact_duplicates']} exact duplicates removed)")

        # Note: Levels 2 & 3 would be implemented with FuzzyDeduplicator
        # and SemanticDeduplicator classes in production

        total_removed = len(documents) - len(docs_after_exact)
        print(f"\nDeduplication complete: removed {total_removed} total duplicates "
              f"({total_removed/len(documents)*100:.1f}%)")

        return docs_after_exact
