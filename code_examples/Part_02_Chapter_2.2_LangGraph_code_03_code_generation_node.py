from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Note: LangGraph nodes use LangChain's LLM components for model interactions.
# LangGraph provides the graph structure; LangChain provides the LLM integration.

def generate_code_node(state: DebugAgentState) -> DebugAgentState:
    """Generate Python code based on requirements and error feedback.

    This node calls an LLM to produce code, incorporating lessons
    from previous failed attempts if iteration > 0.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)

    # Build prompt based on iteration context
    if state["iteration"] == 0:
        # First attempt: just the requirements
        prompt = f"""Generate a Python function that satisfies these requirements:

{state["requirements"]}

Return ONLY the Python code, no explanations. Include necessary imports."""
    else:
        # Refinement attempt: include error feedback
        prompt = f"""The previous code failed tests with these errors:

                {state[`error_analysis`]}

                Requirements:
                {state[`requirements`]}

                Previous attempt:
                {state["generated_code"]}

                Generate improved code that fixes these errors. Return ONLY the Python code."""

    messages = [
        SystemMessage(content="You are an expert Python developer focused on correctness."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    new_code = response.content.strip()

    # Remove markdown code fences if present
    if new_code.startswith("```python"):
        new_code = new_code.split("```python")[1].split("```")[0].strip()
    elif new_code.startswith("```"):
        new_code = new_code.split("```")[1].split("```")[0].strip()

    # Update state with new code
    return {
        "generated_code": new_code,
        "code_history": state["code_history"] + [new_code],
        "iteration": state["iteration"]
    }
