def routing_with_quality_check(state: DebugAgentState) -> str:
    """Enhanced routing: continue iteration if tests pass but code quality is poor."""
    # Always finalize if out of iterations
    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    # Tests failed: always analyze errors
    if state["test_status"] == TestStatus.FAILED:
        return "analyze"

    # Tests passed: check code quality
    # (In real implementation, this would run linters, complexity analysis, etc.)
    code = state["generated_code"]

    # Simple heuristic: if code is too long, try to refactor
    if len(code.split("\n")) > 20:
        return "refactor"  # New node for code refactoring

    # If recursion depth exceeds threshold, try iterative approach
    if "fibonacci" in code and code.count("fibonacci(") > 3:
        return "refactor"

    # Code quality acceptable
    return "finalize"
