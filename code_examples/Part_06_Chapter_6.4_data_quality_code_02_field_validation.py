"""
Data Quality Validation - Field Validation Implementation

Implements core field-level and type validation across document collections.
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


@dataclass
class DocumentValidationReport:
    """Comprehensive validation report for a document."""
    doc_id: str
    passed: bool
    errors: List[ValidationResult] = None
    warnings: List[ValidationResult] = None
    info: List[ValidationResult] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.info is None:
            self.info = []

    def add_result(self, result: ValidationResult):
        """Add a validation result to appropriate category."""
        if result.severity == ValidationSeverity.ERROR:
            self.errors.append(result)
            self.passed = False
        elif result.severity == ValidationSeverity.WARNING:
            self.warnings.append(result)
        else:
            self.info.append(result)

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (f"Document {self.doc_id}: {status} | "
                f"Errors: {len(self.errors)} | "
                f"Warnings: {len(self.warnings)} | "
                f"Info: {len(self.info)}")


class FieldValidator:
    """Field-level validation for document attributes."""

    @staticmethod
    def validate_required_fields(document: Dict[str, Any],
                                required_fields: List[str]) -> List[ValidationResult]:
        """Validate that all required fields are present and non-empty."""
        results = []
        for field in required_fields:
            if field not in document:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Required field '{field}' is missing"
                ))
            elif document[field] is None or document[field] == "":
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field=field,
                    message=f"Required field '{field}' is empty"
                ))
        return results

    @staticmethod
    def validate_field_types(document: Dict[str, Any],
                           field_types: Dict[str, type]) -> List[ValidationResult]:
        """Validate that fields match expected types."""
        results = []
        for field, expected_type in field_types.items():
            if field in document and document[field] is not None:
                if not isinstance(document[field], expected_type):
                    actual_type = type(document[field]).__name__
                    expected = expected_type.__name__
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Field '{field}' has type {actual_type}, expected {expected}",
                        value=document[field]
                    ))
        return results
