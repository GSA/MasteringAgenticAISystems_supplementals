def analyze_failures_node(state: DebugAgentState) -> DebugAgentState:
    """Analyze test failures and extract actionable error context.

    This node processes pytest output to identify what went wrong,
    helping the next generation attempt fix specific issues.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    prompt = f"""Analyze these test failures and identify the core problems:

                Test Output:
                {state["test_output"]}

                Generated Code:
                {state["generated_code"]}

                Provide a concise analysis (2-3 sentences) focusing on:
                1. What errors occurred (imports, logic, edge cases)
                2. Why the code failed to meet requirements
                3. Specific fixes needed for the next iteration

                Keep analysis brief and actionable."""

    messages = [
        SystemMessage(content="You are a code reviewer analyzing test failures."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    analysis = response.content.strip()

    # Extract error patterns (simplified pattern detection)
    patterns = []
    test_output_lower = state["test_output"].lower()
    if "importerror" in test_output_lower or "modulenotfounderror" in test_output_lower:
        patterns.append("missing_imports")
    if "assertionerror" in test_output_lower:
        patterns.append("logic_error")
    if "typeerror" in test_output_lower:
        patterns.append("type_mismatch")

    return {
        "error_analysis": analysis,
        "error_patterns": state["error_patterns"] + patterns
    }
