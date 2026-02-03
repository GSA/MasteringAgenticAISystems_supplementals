from langgraph.graph import StateGraph, END

def should_continue_iteration(state: DebugAgentState) -> str:
    """Conditional edge: route based on test results and iteration limits.

    Returns:
        "finalize" if tests passed or max iterations reached
        "analyze" if tests failed and iterations remain
    """
    # Success: tests passed
    if state["test_status"] == TestStatus.PASSED:
        return "finalize"

    # Gave up: reached iteration limit without success
    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    # Continue: tests failed but iterations remain
    return "analyze"

def finalize_node(state: DebugAgentState) -> DebugAgentState:
    """Final node: mark workflow complete and store final code if successful."""
    return {
        "is_complete": True,
        "final_code": state["generated_code"] if state["test_status"] == TestStatus.PASSED else None
    }

# Build the state graph
workflow = StateGraph(DebugAgentState)

# Add nodes to graph
workflow.add_node("generate", generate_code_node)
workflow.add_node("test", run_tests_node)
workflow.add_node("analyze", analyze_failures_node)
workflow.add_node("finalize", finalize_node)

# Define edges
workflow.set_entry_point("generate")

# Static edges (always proceed)
workflow.add_edge("generate", "test")
workflow.add_edge("analyze", "generate")  # Create the cycle!
workflow.add_edge("finalize", END)

# Conditional edge (routing decision)
workflow.add_conditional_edges(
    "test",
    should_continue_iteration,
    {
        "analyze": "analyze",    # Tests failed, continue iteration
        "finalize": "finalize"   # Tests passed or gave up, finish
    }
)

# Compile graph into executable
app = workflow.compile()
