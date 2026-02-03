def evaluate_error_recovery(execution_trace: list) -> dict:
    """
    Analyze error detection and recovery patterns in execution traces

    Returns metrics for recovery capability beyond first-attempt accuracy
    """
    results = {
        "first_attempt_success": 0,
        "failures_detected": 0,
        "failures_missed": 0,  # Agent didn't recognize failure
        "recovery_attempted": 0,
        "recovery_succeeded": 0,
        "recovery_failed": 0,
        "avg_attempts_to_success": 0.0
    }

    attempt_counts = []

    for action_sequence in execution_trace:
        attempts = action_sequence["attempts"]

        # First attempt
        if attempts[0]["status"] == "success":
            results["first_attempt_success"] += 1
            attempt_counts.append(1)
        else:
            # First attempt failed - did agent detect it?
            if len(attempts) > 1:
                # Agent tried again, so it detected failure
                results["failures_detected"] += 1
                results["recovery_attempted"] += 1

                # Did recovery succeed eventually?
                if attempts[-1]["status"] == "success":
                    results["recovery_succeeded"] += 1
                    attempt_counts.append(len(attempts))
                else:
                    results["recovery_failed"] += 1
            else:
                # Agent didn't try again after failure
                results["failures_missed"] += 1

    # Calculate derived metrics
    total_failures = (results["failures_detected"] + results["failures_missed"])
    results["error_detection_rate"] = (
        results["failures_detected"] / total_failures if total_failures > 0 else 0
    )
    results["recovery_success_rate"] = (
        results["recovery_succeeded"] / results["recovery_attempted"]
        if results["recovery_attempted"] > 0 else 0
    )
    results["avg_attempts_to_success"] = (
        np.mean(attempt_counts) if attempt_counts else 0
    )

    return results