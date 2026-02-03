    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks for retrieval.

        Chunking strategy:
        - Target size: 512 tokens (~2048 characters)
        - Overlap: 50 tokens (~200 characters)
        - Split on semantic boundaries (paragraph → sentence → word)
        - Maintain minimum viable chunk size (100 tokens)

        Args:
            text: Cleaned, validated text to chunk

        Returns:
            List of text chunks ready for embedding
        """
        config = self.config["chunking"]

        # Simplified chunking (production would use tiktoken for accurate token counting)
        # Approximation: 1 token ≈ 4 characters

        char_chunk_size = config["chunk_size"] * 4
        char_overlap = config["overlap"] * 4

        chunks = []
        start = 0

        while start < len(text):
            end = start + char_chunk_size

            # Find good break point (paragraph, sentence, word boundary)
            if end < len(text):
                for separator in config["separators"]:
                    break_point = text.rfind(separator, start, end)
                    if break_point != -1 and break_point > start:
                        end = break_point + len(separator)
                        break

            chunk = text[start:end].strip()

            # Only keep chunks meeting minimum size
            if len(chunk) >= config["min_chunk_size"]:
                chunks.append(chunk)

            # Move start with overlap
            start = end - char_overlap

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks
