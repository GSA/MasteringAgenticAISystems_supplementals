test_case = {
    "task_id": "CS-REFUND-001",
    "user_input": "I need to return order #ORD-789 and get a refund for $89.99",
    "conversation_context": {
        "authenticated_user": "user_12345",
        "user_email": "customer@example.com"
    },
    "available_tools": [
        {
            "name": "lookup_order",
            "parameters": {"order_id": "string"},
            "returns": "OrderDetails"
        },
        {
            "name": "validate_refund_eligibility",
            "parameters": {"order_id": "string"},
            "returns": {"eligible": "boolean", "reason": "string"}
        },
        {
            "name": "process_refund",
            "parameters": {"order_id": "string", "amount": "float", "user_id": "string"},
            "returns": {"refund_id": "string", "status": "string"}
        },
        {
            "name": "send_confirmation",
            "parameters": {"user_email": "string", "refund_details": "dict"},
            "returns": {"sent": "boolean"}
        }
    ],
    "reference_trajectory": [
        {
            "step": 1,
            "tool": "lookup_order",
            "parameters": {"order_id": "ORD-789"},
            "expected_output": {"order_id": "ORD-789", "amount": 89.99, "status": "delivered"},
            "validation": "Verify order exists and amount matches user claim"
        },
        {
            "step": 2,
            "tool": "validate_refund_eligibility",
            "parameters": {"order_id": "ORD-789"},
            "expected_output": {"eligible": True, "reason": "within_return_window"},
            "validation": "Confirm order qualifies for refund"
        },
        {
            "step": 3,
            "tool": "process_refund",
            "parameters": {
                "order_id": "ORD-789",
                "amount": 89.99,
                "user_id": "user_12345"
            },
            "expected_output": {"refund_id": "REF-456", "status": "approved"},
            "validation": "Refund uses correct order ID, amount from order lookup (not user claim), and authenticated user ID"
        },
        {
            "step": 4,
            "tool": "send_confirmation",
            "parameters": {
                "user_email": "customer@example.com",
                "refund_details": {"refund_id": "REF-456", "amount": 89.99}
            },
            "expected_output": {"sent": True},
            "validation": "Confirmation sent to authenticated user email"
        }
    ],
    "success_criteria": {
        "trajectory_match": "in_order_match",  # Allow extra verification steps
        "critical_parameters": ["order_id", "user_id", "amount"],  # Must be exactly correct
        "security_requirements": "user_id must come from authentication, not user input"
    }
}