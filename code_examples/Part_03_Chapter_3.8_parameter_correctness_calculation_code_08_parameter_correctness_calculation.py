def calculate_parameter_correctness(actual_params: dict,
                                   expected_params: dict,
                                   strict: bool = True) -> dict:
    """
    Calculate parameter correctness with per-parameter breakdown

    Args:
        actual_params: Parameters agent actually used
        expected_params: Expected parameter values
        strict: If True, require exact match; if False, allow semantic equivalence

    Returns:
        Dict with overall rate and per-parameter accuracy
    """
    results = {
        "overall_rate": 0.0,
        "per_parameter": {},
        "missing_parameters": [],
        "extra_parameters": [],
        "incorrect_parameters": []
    }

    all_param_names = set(actual_params.keys()) | set(expected_params.keys())
    correct_count = 0

    for param_name in all_param_names:
        if param_name not in expected_params:
            # Agent provided parameter not in reference
            results["extra_parameters"].append(param_name)
            results["per_parameter"][param_name] = 0.0
        elif param_name not in actual_params:
            # Agent missed required parameter
            results["missing_parameters"].append(param_name)
            results["per_parameter"][param_name] = 0.0
        else:
            # Compare parameter values
            actual_val = actual_params[param_name]
            expected_val = expected_params[param_name]

            if strict:
                params_match = actual_val == expected_val
            else:
                params_match = semantically_equivalent(actual_val, expected_val)

            if params_match:
                correct_count += 1
                results["per_parameter"][param_name] = 1.0
            else:
                results["incorrect_parameters"].append({
                    "parameter": param_name,
                    "expected": expected_val,
                    "actual": actual_val
                })
                results["per_parameter"][param_name] = 0.0

    results["overall_rate"] = correct_count / len(all_param_names) if all_param_names else 0.0
    return results