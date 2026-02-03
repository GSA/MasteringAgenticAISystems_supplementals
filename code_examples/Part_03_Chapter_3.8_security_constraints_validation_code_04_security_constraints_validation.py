def validate_security_constraints(user_id: str, action: str,
                                  params: dict, permissions: dict) -> tuple[bool, str]:
    """Validate action doesn't violate security constraints"""

    # Check authorization: can this user perform this action?
    if action not in permissions.get(user_id, {}).get('allowed_actions', []):
        return False, f"User {user_id} not authorized for action {action}"

    # Check resource scope: do parameters reference authorized resources?
    if 'customer_id' in params:
        requested_customer = params['customer_id']
        # Users can only access their own data unless they have admin role
        if requested_customer != user_id and 'admin' not in permissions[user_id]['roles']:
            return False, f"BOLA attempt: User {user_id} attempting to access {requested_customer}"

    # Check for injection attacks in string parameters
    for param_name, param_value in params.items():
        if isinstance(param_value, str):
            if contains_sql_injection_pattern(param_value):
                return False, f"SQL injection detected in parameter {param_name}"
            if contains_path_traversal_pattern(param_value):
                return False, f"Path traversal detected in parameter {param_name}"

    # Check rate limits: has user exceeded action frequency?
    if exceeds_rate_limit(user_id, action):
        return False, f"Rate limit exceeded for action {action}"

    return True, "Security validation passed"