import re
from typing import Dict, Any

def calculate_policy_compliance(
    response: str,
    policies: Dict[str, Any]
) -> float:
    """Check policy compliance"""
    response_lower = response.lower()

    # Extract mentioned discounts (e.g., "10% off", "$20 discount")
    discount_patterns = [
        r'(\d+)%\s*(?:off|discount)',  # Percentage discounts
        r'\$(\d+)\s*(?:off|discount)'  # Dollar amount discounts
    ]

    violations = []

    for pattern in discount_patterns:
        matches = re.findall(pattern, response_lower)
        for match in matches:
            discount_value = float(match)

            # Check against policy limit
            if pattern.startswith(r'(\d+)%'):
                # Percentage discount
                if discount_value > policies.get('max_discount_pct', 15):
                    violations.append(
                        f"Discount {discount_value}% exceeds policy limit"
                    )
            else:
                # Dollar discount
                if discount_value > policies.get('max_discount_dollar', 50):
                    violations.append(
                        f"Discount ${discount_value} exceeds policy limit"
                    )

    # Return 1.0 if compliant, 0.0 if violations found
    return 1.0 if len(violations) == 0 else 0.0
