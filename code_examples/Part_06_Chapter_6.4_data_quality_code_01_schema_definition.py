"""
Comprehensive Data Quality Validation Pipeline - Part 1: Schema Definition

Demonstrates production-grade validation across schema, range, format,
and reference dimensions. Designed for extensibility and detailed error reporting.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from urllib.parse import urlparse

class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    ERROR = "error"      # Must fix - document rejected
    WARNING = "warning"  # Should review - document flagged
    INFO = "info"        # Optional improvement

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
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)

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
