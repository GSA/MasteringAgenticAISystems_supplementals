"""
Data Quality Validation - Content Completeness Checking

Validates document completeness across field-level, document-level, and corpus-level dimensions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Container for validation outcomes."""
    is_valid: bool
    severity: ValidationSeverity
    field: str
    message: str
    value: Any = None


class CompletenessValidator:
    """Validates document completeness at multiple levels."""

    MIN_CONTENT_LENGTH = 50  # Minimum characters for meaningful content
    MAX_CONTENT_LENGTH = 100000  # Maximum reasonable content length

    @staticmethod
    def validate_content_length(content: str, field: str = 'content') -> List[ValidationResult]:
        """Validate that content has reasonable length."""
        results = []
        if isinstance(content, str):
            length = len(content)
            if length < CompletenessValidator.MIN_CONTENT_LENGTH:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field=field,
                    message=f"Content length {length} chars is suspiciously short (< {CompletenessValidator.MIN_CONTENT_LENGTH})"
                ))
            elif length > CompletenessValidator.MAX_CONTENT_LENGTH:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field=field,
                    message=f"Content length {length} chars exceeds recommended maximum"
                ))
        return results

    @staticmethod
    def validate_required_sections(content: str,
                                  required_sections: List[str]) -> List[ValidationResult]:
        """Validate that document contains required section headers."""
        results = []
        content_lower = content.lower()
        for section in required_sections:
            section_lower = section.lower()
            if section_lower not in content_lower:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field='content_structure',
                    message=f"Document appears to be missing section: '{section}'"
                ))
        return results

    @staticmethod
    def validate_field_completeness(document: Dict[str, Any],
                                   required_fields: List[str]) -> List[ValidationResult]:
        """Validate field-level completeness."""
        results = []
        missing_count = 0
        for field in required_fields:
            if field not in document or document[field] is None:
                missing_count += 1
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field=field,
                    message=f"Required field '{field}' is missing or null"
                ))
        return results
