reference_trajectory = [
    {
        "step": 1,
        "tool": "lookup_order",
        "context_conditions": {
            "user_provided_order_id": True,
            "order_id_format_valid": True
        }
    },
    {
        "step": 2,
        "tool": "check_return_eligibility",
        "context_conditions": {
            "order_found": True,
            "user_intent": "return"  # Not escalation
        }
    },
    {
        "step": 3,
        "tool": "escalate_to_human",  # Conditional action
        "context_conditions": {
            "return_not_eligible": True,  # Only escalate if ineligible
            "OR": {"user_requested_human": True}  # Or if explicitly requested
        }
    }
]