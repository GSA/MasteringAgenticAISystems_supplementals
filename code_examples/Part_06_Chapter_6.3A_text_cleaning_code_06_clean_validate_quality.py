    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for embedding.

        Cleaning operations:
        - Remove HTML tags (from web scraping)
        - Normalize whitespace (multiple spaces/newlines to single)
        - Remove control characters (formatting artifacts)
        - Trim leading/trailing whitespace

        Args:
            text: Raw text from extraction

        Returns:
            Cleaned text ready for quality validation
        """
        import re

        # Remove HTML tags (basic - use BeautifulSoup for production)
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())

        # Trim
        text = text.strip()

        return text

    def validate_quality(self, text: str) -> bool:
        """
        Validate document quality before chunking.

        Quality checks:
        - Length bounds (50-100,000 characters)
        - Minimum word count (10 words)
        - Boilerplate detection (legal disclaimers, copyright notices)
        - Content density (not just whitespace/punctuation)

        Args:
            text: Cleaned text to validate

        Returns:
            True if document meets quality standards, False otherwise
        """
        config = self.config["quality"]

        # Length check
        if len(text) < config["min_length"]:
            return False

        if len(text) > config["max_length"]:
            return False

        # Basic content check (not just whitespace/punctuation)
        word_count = len(text.split())
        if word_count < 10:
            return False

        # Boilerplate detection (simplified)
        if config["remove_boilerplate"]:
            boilerplate_phrases = [
                "This page was automatically generated",
                "Copyright © All rights reserved",
                "Terms and Conditions apply"
            ]

            for phrase in boilerplate_phrases:
                if phrase.lower() in text.lower():
                    return False

        return True
