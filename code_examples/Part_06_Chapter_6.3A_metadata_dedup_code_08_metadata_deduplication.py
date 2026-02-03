    def compute_hash(self, text: str) -> str:
        """
        Compute SHA-256 content hash for deduplication.

        Content hashing enables O(1) duplicate detection via set lookups,
        scaling to millions of documents. SHA-256 provides strong collision
        resistance—probability of false positives is negligible.

        Args:
            text: Text to hash

        Returns:
            64-character hexadecimal hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def extract_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize metadata for filtered retrieval.

        Metadata enables queries like:
        - "Find documentation about authentication updated this month"
        - "Retrieve articles tagged 'troubleshooting' in category 'networking'"

        Args:
            doc: Source document with potential metadata fields

        Returns:
            Normalized metadata dictionary
        """
        return {
            "category": doc.get("category", "general"),
            "tags": doc.get("tags", []),
            "updated_at": doc.get("updated_at", datetime.now()).isoformat(),
            "source": doc.get("source", "unknown")
        }
