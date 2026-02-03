from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import logging

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State with error tracking."""
    input: str
    output: str
    errors: list[str]
    retry_count: int
    max_retries: int


def analysis_node(state: AgentState) -> AgentState:
    """Node that may fail, with retry tracking."""
    try:
        # Perform analysis (may raise exceptions)
        result = perform_llm_analysis(state["input"])
        state["output"] = result
        state["retry_count"] = 0  # Reset on success
        return state

    except Exception as e:
        # Record error
        state["errors"].append(f"Analysis failed: {str(e)}")
        state["retry_count"] += 1
        logger.error(f"Analysis node error (retry {state['retry_count']}): {str(e)}")
        return state


def fallback_node(state: AgentState) -> AgentState:
    """Fallback to simpler analysis when primary fails."""
    logger.warning("Using fallback analysis")
    state["output"] = f"[Fallback analysis] Basic summary for: {state['input']}"
    return state


def should_retry(state: AgentState) -> Literal["retry", "fallback", "end"]:
    """Conditional routing based on errors and retry count."""
    if state["output"]:
        # Success - end workflow
        return "end"

    if state["retry_count"] < state["max_retries"]:
        # Transient failure - retry
        return "retry"

    # Persistent failure - fallback
    return "fallback"


# Build graph with error recovery paths
def create_error_recovery_workflow():
    """Create LangGraph workflow with error recovery paths."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("fallback", fallback_node)

    # Add conditional edges for error recovery
    workflow.add_conditional_edges(
        "analysis",
        should_retry,
        {
            "retry": "analysis",  # Loop back for retry
            "fallback": "fallback",  # Fallback strategy
            "end": END  # Success
        }
    )

    workflow.add_edge("fallback", END)
    workflow.set_entry_point("analysis")

    # Compile and return
    return workflow.compile()


# Example usage function
def perform_llm_analysis(input_text: str) -> str:
    """Placeholder for LLM analysis function."""
    # In real implementation, this would call an LLM
    return f"Analysis of: {input_text}"


def execute_workflow_with_error_recovery(input_prompt: str) -> dict:
    """Execute workflow with error recovery."""
    app = create_error_recovery_workflow()

    result = app.invoke({
        "input": input_prompt,
        "output": "",
        "errors": [],
        "retry_count": 0,
        "max_retries": 3
    })

    print(f"Result: {result['output']}")
    if result['errors']:
        print(f"Errors encountered: {result['errors']}")

    return result
