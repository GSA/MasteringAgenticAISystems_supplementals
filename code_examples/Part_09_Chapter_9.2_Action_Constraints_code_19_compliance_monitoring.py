{
    'request_id': 'APR_20240125_1847',
    'agent_id': 'refund_processing_agent',
    'action': 'refund: $4,500.00 for customer C_789012',
    'details': {
        'transaction_type': 'refund',
        'amount': 4500.00,
        'customer_id': 'C_789012',
        'customer_history': {
            'account_age_days': 730,
            'total_purchases': 24,
            'average_purchase': 850,
            'previous_disputes': 1,
            'dispute_resolution': 'resolved in customer favor'
        },
        'dispute_details': {
            'disputed_charge': 4500.00,
            'dispute_reason': 'service not delivered',
            'merchant': 'ACME Consulting',
            'transaction_date': '2024-01-15',
            'delivery_confirmation': 'none found',
            'customer_evidence': 'email thread showing non-delivery'
        },
        'agent_analysis':
            'Customer has 2-year account history with minimal disputes. '
            'Transaction for consulting services shows no delivery confirmation. '
            'Customer provided email evidence of non-delivery complaints to merchant. '
            'Merchant refund policy states "full refund if services not delivered." '
            'Recommend refund approval.',
        'recommendation': 'APPROVE',
        'merchant_policy': 'Full refund within 90 days if services not delivered'
    },
    'risk_level': 'medium',
    'approver_id': 'manager_finance_team_3',
    'requested_at': '2024-01-25T18:47:22Z',
    'timeout_at': '2024-01-25T22:47:22Z'  # 4-hour SLA for medium risk
}