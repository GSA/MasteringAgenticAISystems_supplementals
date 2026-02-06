"""
PII Detection and Redaction - Pattern Definitions

Defines regex patterns for common PII types.
"""

import re
from enum import Enum
from dataclasses import dataclass


class PIIType(Enum):
    """Types of PII we detect."""
    SSN = "social_security_number"
    CREDIT_CARD = "credit_card"
    PHONE = "phone_number"
    EMAIL = "email_address"
    PERSON_NAME = "person_name"
    MEDICAL_ID = "medical_record_number"
    ACCOUNT_NUMBER = "account_number"


@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    pii_type: PIIType
    text: str
    start_pos: int
    end_pos: int
    confidence: float  # 0-1 confidence score
    detection_method: str  # 'pattern', 'ner', 'context'


class PIIPatterns:
    """Pattern library for common PII types."""

    PATTERNS = {
        PIIType.SSN: [
            r'\b\d{3}-\d{2}-\d{4}\b',  # 123-45-6789
            r'\b\d{9}\b'  # 123456789 (9 consecutive digits)
        ],
        PIIType.CREDIT_CARD: [
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'  # 16-digit cards
        ],
        PIIType.PHONE: [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 555-123-4567
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'  # (555) 123-4567
        ],
        PIIType.EMAIL: [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ],
        PIIType.MEDICAL_ID: [
            r'\bMRN[:\s#]*\d{6,10}\b',  # Medical Record Number
            r'\bPatient\s*ID[:\s#]*\d{6,10}\b'
        ]
    }

    # Context keywords that indicate sensitive data nearby
    CONTEXT_KEYWORDS = {
        PIIType.ACCOUNT_NUMBER: ['account number', 'account #', 'acct'],
        PIIType.MEDICAL_ID: ['patient id', 'medical record', 'mrn', 'patient number']
    }

    @staticmethod
    def find_pattern_matches(text: str, pii_type: PIIType) -> list:
        """
        Find all matches of a specific PII pattern in text.

        Returns:
            List of (match_text, start_pos, end_pos) tuples
        """
        matches = []
        patterns = PIIPatterns.PATTERNS.get(pii_type, [])

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                matches.append((
                    match.group(),
                    match.start(),
                    match.end()
                ))

        return matches

    @staticmethod
    def find_all_pii_patterns(text: str) -> list:
        """
        Find all PII patterns in text.

        Returns:
            List of (pii_type, match_text, start_pos, end_pos) tuples
        """
        all_matches = []

        for pii_type in PIIType:
            pattern_matches = PIIPatterns.find_pattern_matches(text, pii_type)
            for match_text, start_pos, end_pos in pattern_matches:
                all_matches.append((pii_type, match_text, start_pos, end_pos))

        # Sort by position
        all_matches.sort(key=lambda x: x[2])
        return all_matches
