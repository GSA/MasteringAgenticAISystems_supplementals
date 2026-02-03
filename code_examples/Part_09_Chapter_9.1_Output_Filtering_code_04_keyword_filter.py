import re
from typing import List, Tuple

class KeywordFilter:
    """Simple keyword-based output filter"""

    def __init__(self):
        # Deny list of prohibited terms
        self.deny_list = [
            r'\bhate\b',
            r'\bkill\b',
            r'\bharm\b',
            # Add investment advice patterns
            r'\byou should (buy|sell|invest)\b',
            r'\b(strong buy|strong sell)\b',
        ]

        # Compile patterns for efficiency
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.deny_list]

    def filter(self, text: str) -> Tuple[bool, List[str]]:
        """
        Filter text for prohibited keywords.
        Returns (is_safe, matched_patterns)
        """
        matches = []
        for pattern in self.patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)

        is_safe = len(matches) == 0
        return is_safe, matches

# Usage
filter = KeywordFilter()
output = "You should buy Tesla stock now!"
is_safe, matches = filter.filter(output)
if not is_safe:
    print(f"Output blocked. Matched patterns: {matches}")
