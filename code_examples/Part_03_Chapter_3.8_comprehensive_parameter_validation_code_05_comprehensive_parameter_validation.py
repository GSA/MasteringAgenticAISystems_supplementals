def comprehensive_parameter_validation(
    action: str,
    params: dict,
    user_context: dict,
    conversation_history: list,
    tool_outputs: dict
) -> tuple[bool, dict]:
    """Multi-layer parameter validation pipeline"""

    validation_results = {
        "static_valid": False,
        "semantic_valid": False,
        "security_valid": False,
        "context_valid": False,
        "errors": []
    }

    # Layer 1: Static schema validation
    try:
        validated_params = validate_schema(action, params)
        validation_results["static_valid"] = True
    except ValidationError as e:
        validation_results["errors"].append(f"Schema validation failed: {e}")
        return False, validation_results  # Early exit on schema failure

    # Layer 2: Semantic grounding validation
    grounding_valid, grounding_errors = validate_semantic_grounding(
        validated_params, conversation_history, tool_outputs
    )
    validation_results["semantic_valid"] = grounding_valid
    if not grounding_valid:
        validation_results["errors"].extend(grounding_errors)

    # Layer 3: Security constraint validation
    security_valid, security_msg = validate_security_constraints(
        user_context['user_id'], action, validated_params, user_context['permissions']
    )
    validation_results["security_valid"] = security_valid
    if not security_valid:
        validation_results["errors"].append(security_msg)
        return False, validation_results  # Security failures block execution

    # Layer 4: Context consistency validation
    context_valid, context_errors = validate_context_consistency(
        validated_params, conversation_history, user_context
    )
    validation_results["context_valid"] = context_valid
    if not context_valid:
        validation_results["errors"].extend(context_errors)

    # All layers must pass for overall validation success
    all_valid = all([
        validation_results["static_valid"],
        validation_results["semantic_valid"],
        validation_results["security_valid"],
        validation_results["context_valid"]
    ])

    return all_valid, validation_results