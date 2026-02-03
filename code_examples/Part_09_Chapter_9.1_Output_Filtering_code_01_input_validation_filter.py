# Example: Layer 1 Input Validation
import re
from typing import Tuple

class InputValidator:
    """Layer 1: Prevent harmful inputs from reaching the model"""

    def __init__(self):
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        self.profanity_list = ['badword1', 'badword2']  # Simplified

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input before passing to model.
        Returns (is_valid, reason)
        """
        # Check for PII
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return False, f"Contains {pii_type}"

        # Check for profanity
        text_lower = text.lower()
        for profanity in self.profanity_list:
            if profanity in text_lower:
                return False, "Contains prohibited language"

        return True, ""

# Usage
validator = InputValidator()
user_input = "My SSN is 123-45-6789"
is_valid, reason = validator.validate_input(user_input)
if not is_valid:
    print(f"Input blocked: {reason}")
