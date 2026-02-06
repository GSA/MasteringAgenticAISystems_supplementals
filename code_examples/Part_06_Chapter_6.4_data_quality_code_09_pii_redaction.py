"""
PII Detection and Redaction - ETL Integration

Integrates PII detection into document processing pipelines.
"""

from typing import Dict, Any, List, Tuple
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


def process_document_with_pii_protection(
    document: Dict[str, Any],
    pii_detector
) -> Dict[str, Any]:
    """
    Process document with PII redaction.

    Args:
        document: Document with 'content', 'title', and optional 'id'
        pii_detector: Initialized PIIDetector instance

    Returns:
        Processed document with PII redacted and metadata added
    """
    # Redact content
    redacted_content, pii_matches = pii_detector.detect_and_redact(
        document.get('content', '')
    )

    # Redact title if needed
    redacted_title, title_matches = pii_detector.detect_and_redact(
        document.get('title', '')
    )

    # Log PII detections for audit
    total_pii = len(pii_matches) + len(title_matches)
    if total_pii > 0:
        print(f"WARNING: Document {document.get('id', 'unknown')} contained "
              f"{total_pii} PII instances (redacted)")

    # Collect PII types found
    all_matches = pii_matches + title_matches
    pii_types = list(set(m.pii_type.value for m in all_matches))

    return {
        **document,
        'content': redacted_content,
        'title': redacted_title,
        'pii_detected': total_pii,
        'pii_types': pii_types,
        'pii_redacted': total_pii > 0
    }


def batch_process_documents_with_pii_protection(
    documents: List[Dict[str, Any]],
    pii_detector
) -> Dict[str, Any]:
    """
    Process batch of documents with PII protection.

    Returns:
        Dict with processed documents and statistics
    """
    processed_docs = []
    total_pii_detections = 0
    docs_with_pii = 0
    pii_type_counts = {}

    for doc in documents:
        processed = process_document_with_pii_protection(doc, pii_detector)
        processed_docs.append(processed)

        pii_count = processed.get('pii_detected', 0)
        if pii_count > 0:
            docs_with_pii += 1
            total_pii_detections += pii_count

            # Track PII types
            for pii_type in processed.get('pii_types', []):
                pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1

    return {
        'processed_documents': processed_docs,
        'statistics': {
            'total_documents': len(documents),
            'documents_with_pii': docs_with_pii,
            'total_pii_instances': total_pii_detections,
            'pii_types_detected': pii_type_counts,
            'redaction_rate': total_pii_detections / len(documents) if documents else 0
        }
    }


def create_pii_audit_log(
    document_id: str,
    pii_matches: List[Any],
    title_matches: List[Any],
    redaction_strategy: str
) -> Dict[str, Any]:
    """
    Create audit log entry for PII detection.

    Args:
        document_id: ID of processed document
        pii_matches: Content PII matches
        title_matches: Title PII matches
        redaction_strategy: How PII was redacted

    Returns:
        Audit log entry
    """
    all_matches = pii_matches + title_matches

    return {
        'document_id': document_id,
        'timestamp': None,  # Would use datetime.now()
        'total_pii_instances': len(all_matches),
        'by_type': {},
        'by_field': {
            'content': len(pii_matches),
            'title': len(title_matches)
        },
        'redaction_strategy': redaction_strategy,
        'matches': [
            {
                'pii_type': m.pii_type.value,
                'confidence': m.confidence,
                'detection_method': m.detection_method,
                'field': 'content' if m in pii_matches else 'title'
            }
            for m in all_matches
        ]
    }
