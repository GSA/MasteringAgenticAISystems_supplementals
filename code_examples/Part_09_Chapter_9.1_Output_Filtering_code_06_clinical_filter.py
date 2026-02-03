import re
from typing import Tuple

class ClinicalOutputFilter:
    """HIPAA-compliant output filter for clinical AI"""

    def __init__(self):
        self.phi_patterns = {
            'mrn': r'\b[A-Z]{2,3}\d{6,8}\b',
            'dob': r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        self.high_risk_actions = [
            'prescribe medication',
            'modify treatment plan',
            'order diagnostic test'
        ]

    def filter_output(self, text: str, action: str) -> Tuple[bool, str]:
        """Filter clinical output for PHI and risk level"""
        # Check for PHI exposure
        for phi_type, pattern in self.phi_patterns.items():
            if re.search(pattern, text):
                return False, f"Contains {phi_type.upper()} - HIPAA violation"

        # High-risk actions require human approval
        if action in self.high_risk_actions:
            return False, "High-risk action requires physician approval"

        return True, ""
