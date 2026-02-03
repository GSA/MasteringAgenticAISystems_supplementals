# Configure restrictive defaults
owd_settings = {
    'Account': 'Private',      # No access by default
    'Contact': 'Private',
    'Opportunity': 'Private'
}

# Add precise sharing rules
sharing_rules = [
    SharingRule(
        object='Account',
        criteria='owner_id == agent_user_id',  # Only agent's own records
        access_level='Read'                     # Read-only access
    )
]