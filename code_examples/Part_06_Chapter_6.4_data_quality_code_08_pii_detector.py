"""
PII Detection and Redaction - Detector Implementation

Combines pattern-based, NER-based, and context-aware detection.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


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
    confidence: float
    detection_method: str


class PIIDetector:
    """Multi-strategy PII detection and redaction."""

    PATTERNS = {
        PIIType.SSN: [r'\b\d{3}-\d{2}-\d{4}\b', r'\b\d{9}\b'],
        PIIType.CREDIT_CARD: [r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'],
        PIIType.PHONE: [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\(\d{3}\)\s*\d{3}[-.]?\d{4}'],
        PIIType.EMAIL: [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
        PIIType.MEDICAL_ID: [r'\bMRN[:\s#]*\d{6,10}\b', r'\bPatient\s*ID[:\s#]*\d{6,10}\b']
    }

    CONTEXT_KEYWORDS = {
        PIIType.ACCOUNT_NUMBER: ['account number', 'account #', 'acct'],
        PIIType.MEDICAL_ID: ['patient id', 'medical record', 'mrn', 'patient number']
    }

    def __init__(self, redaction_strategy: str = 'mask'):
        """
        Initialize PII detector.

        Args:
            redaction_strategy: How to redact PII ('mask', 'hash', or 'remove')
        """
        self.redaction_strategy = redaction_strategy
        self.stats = {pii_type: 0 for pii_type in PIIType}

    def _detect_patterns(self, text: str) -> List[PIIMatch]:
        """Detect PII using regex patterns."""
        matches = []

        for pii_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.95,
                        detection_method='pattern'
                    ))

        return matches

    def _detect_names_simple(self, text: str) -> List[PIIMatch]:
        """Simplified name detection using capitalization patterns."""
        matches = []
        name_pattern = r'\b(Dr\.|Mr\.|Ms\.|Mrs\.)?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b'

        for match in re.finditer(name_pattern, text):
            name = match.group()
            if not any(fp in name for fp in ['The ', 'This ', 'That ']):
                matches.append(PIIMatch(
                    pii_type=PIIType.PERSON_NAME,
                    text=name,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.75,
                    detection_method='ner'
                ))

        return matches

    def _detect_contextual(self, text: str) -> List[PIIMatch]:
        """Detect PII based on surrounding context keywords."""
        matches = []

        for pii_type, keywords in self.CONTEXT_KEYWORDS.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                text_lower = text.lower()

                pos = 0
                while True:
                    pos = text_lower.find(keyword_lower, pos)
                    if pos == -1:
                        break

                    search_end = min(pos + len(keyword) + 20, len(text))
                    context_text = text[pos:search_end]

                    number_match = re.search(r'[:\s#]*(\d{6,})', context_text)
                    if number_match:
                        number_start = pos + number_match.start(1)
                        number_end = pos + number_match.end(1)

                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            text=text[number_start:number_end],
                            start_pos=number_start,
                            end_pos=number_end,
                            confidence=0.85,
                            detection_method='context'
                        ))

                    pos += len(keyword)

        return matches

    def detect_and_redact(self, text: str) -> Tuple[str, List[PIIMatch]]:
        """
        Detect PII and return redacted text with detection details.

        Returns:
            (redacted_text, list of PII matches found)
        """
        matches: List[PIIMatch] = []

        # Strategy 1: Pattern-based detection
        matches.extend(self._detect_patterns(text))

        # Strategy 2: Named Entity Recognition (simplified)
        matches.extend(self._detect_names_simple(text))

        # Strategy 3: Context-aware detection
        matches.extend(self._detect_contextual(text))

        # Sort matches by position (reverse order for safe replacement)
        matches.sort(key=lambda m: m.start_pos, reverse=True)

        # Redact matches
        redacted_text = text
        for match in matches:
            redacted_text = self._redact_match(redacted_text, match)
            self.stats[match.pii_type] += 1

        return redacted_text, matches

    def _redact_match(self, text: str, match: PIIMatch) -> str:
        """Redact a single PII match from text."""
        if self.redaction_strategy == 'mask':
            replacement = f"[{match.pii_type.value.upper()}]"
        elif self.redaction_strategy == 'hash':
            import hashlib
            hash_val = hashlib.sha256(match.text.encode()).hexdigest()[:8]
            replacement = f"[{match.pii_type.value.upper()}_{hash_val}]"
        else:  # 'remove'
            replacement = ""

        return text[:match.start_pos] + replacement + text[match.end_pos:]

    def get_stats(self) -> Dict[PIIType, int]:
        """Get detection statistics."""
        return self.stats.copy()
