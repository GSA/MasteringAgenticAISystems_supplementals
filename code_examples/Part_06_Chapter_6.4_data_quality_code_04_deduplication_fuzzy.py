"""
Multi-Level Deduplication - Fuzzy Matching Implementation

Uses Levenshtein distance to detect near-duplicate documents.
"""

from typing import List, Dict, Set


class FuzzyDeduplicator:
    """Detects near-duplicates using fuzzy string matching."""

    def __init__(self, similarity_threshold: float = 0.90):
        """
        Initialize fuzzy deduplicator.

        Args:
            similarity_threshold: Minimum similarity (0-1) to mark as duplicate
        """
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein distance between two strings.

        This is the minimum number of single-character edits
        (insertions, deletions, substitutions) needed to change
        one string into the other.
        """
        if len(s1) < len(s2):
            return FuzzyDeduplicator._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _normalized_similarity(self, s1: str, s2: str) -> float:
        """
        Compute normalized string similarity (0-1 scale).

        Returns:
            1.0 for identical strings, 0.0 for completely different
        """
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0

        distance = self._levenshtein_distance(s1, s2)
        similarity = 1.0 - (distance / max_len)
        return similarity

    def find_fuzzy_duplicates(self, documents: List[Dict[str, str]]) -> List[tuple]:
        """
        Find all fuzzy duplicate pairs above threshold.

        Args:
            documents: List of dicts with 'id' and 'content' keys

        Returns:
            List of (doc1_id, doc2_id, similarity) tuples
        """
        duplicates = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]

                similarity = self._normalized_similarity(
                    doc1['content'], doc2['content']
                )

                if similarity >= self.similarity_threshold:
                    duplicates.append((
                        doc1['id'],
                        doc2['id'],
                        similarity
                    ))

        return duplicates
