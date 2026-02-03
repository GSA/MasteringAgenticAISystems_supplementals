# ====================================================================
# DECISION NODES (LOGIC TREE BRANCHES)
# ====================================================================

def categorize_query(state: SupportAgentState) -> SupportAgentState:
    """
    First node: Classify the user query into a category.

    This is the root of the logic tree - all subsequent routing
    depends on this classification.
    """
    query = state["user_query"]

    # Use NVIDIA NIM for fast, high-quality classification
    classification_prompt = f"""
    Classify this customer support query into exactly ONE category:
    - billing: payment issues, refunds, invoices, charges
    - technical: bugs, errors, performance problems
    - general: FAQs, account info, product questions
    - escalation: angry customer, complex issue, legal matter

    Query: {query}

    Respond with ONLY the category name, nothing else.
    """

    response = llm.invoke(classification_prompt)
    category = response.content.strip().lower()

    # Update state with classification
    new_state = state.copy()
    new_state["query_category"] = category
    new_state["metadata"]["classification_confidence"] = response.response_metadata.get("confidence", 0.0)

    return new_state
