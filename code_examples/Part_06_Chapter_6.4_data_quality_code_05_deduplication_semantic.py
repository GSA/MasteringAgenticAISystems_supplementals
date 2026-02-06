"""
Multi-Level Deduplication - Semantic Matching Implementation

Detects semantic duplicates using embedding similarity.
"""

from typing import List, Dict, Set
import numpy as np


class SemanticDeduplicator:
    """Detects semantic duplicates using embedding similarity."""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize semantic deduplicator.

        Args:
            similarity_threshold: Minimum cosine similarity (0-1) to mark as duplicate
        """
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns:
            Similarity score from 0 (orthogonal) to 1 (identical)
        """
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def find_semantic_duplicates(self,
                                documents: List[Dict],
                                embedding_key: str = 'embedding') -> List[tuple]:
        """
        Find semantic duplicate pairs above threshold.

        Args:
            documents: List of dicts with 'id' and embedding
            embedding_key: Key in document dict containing embedding vector

        Returns:
            List of (doc1_id, doc2_id, similarity) tuples
        """
        duplicates = []
        doc_embeddings = [doc.get(embedding_key) for doc in documents]

        # Verify all documents have embeddings
        if any(emb is None for emb in doc_embeddings):
            return duplicates

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                emb1 = np.array(doc_embeddings[i])
                emb2 = np.array(doc_embeddings[j])

                similarity = self.cosine_similarity(emb1, emb2)

                if similarity >= self.similarity_threshold:
                    duplicates.append((
                        documents[i]['id'],
                        documents[j]['id'],
                        float(similarity)
                    ))

        return duplicates

    def remove_semantic_duplicates(self, documents: List[Dict]) -> List[Dict]:
        """
        Remove semantic duplicates keeping first occurrence.

        Args:
            documents: List of documents with embeddings

        Returns:
            List with duplicates removed
        """
        if not documents:
            return []

        doc_embeddings = [np.array(doc.get('embedding', [])) for doc in documents]
        marked_duplicate = set()

        for i in range(len(documents)):
            if i in marked_duplicate:
                continue

            for j in range(i + 1, len(documents)):
                if j in marked_duplicate:
                    continue

                similarity = self.cosine_similarity(doc_embeddings[i], doc_embeddings[j])
                if similarity >= self.similarity_threshold:
                    marked_duplicate.add(j)

        # Return non-duplicate documents
        return [doc for i, doc in enumerate(documents) if i not in marked_duplicate]
