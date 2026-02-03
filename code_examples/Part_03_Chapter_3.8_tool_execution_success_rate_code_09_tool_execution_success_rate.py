def calculate_tool_execution_success_rate(execution_log: list) -> dict:
    """
    Measure tool execution success rates with failure mode breakdown

    Returns:
        Dict with success rate and categorized failure modes
    """
    results = {
        "success_rate": 0.0,
        "total_invocations": len(execution_log),
        "successful": 0,
        "failures": {
            "parameter_validation_error": 0,  # Tool rejected parameters
            "execution_error": 0,              # Tool ran but crashed
            "timeout": 0,                       # Tool exceeded time limit
            "authorization_error": 0,           # Permission denied
            "not_found_error": 0                # Resource doesn't exist
        }
    }

    for log_entry in execution_log:
        if log_entry["status"] == "success":
            results["successful"] += 1
        else:
            # Categorize failure mode
            error_type = classify_error(log_entry["error"])
            results["failures"][error_type] += 1

    results["success_rate"] = results["successful"] / results["total_invocations"]
    return results