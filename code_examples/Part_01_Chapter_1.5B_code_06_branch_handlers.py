def handle_billing(state: SupportAgentState) -> SupportAgentState:
    """
    Billing branch: Process payment-related queries.

    In production, this would integrate with payment processing systems,
    invoice databases, and refund APIs.
    """
    query = state["user_query"]

    billing_prompt = f"""
    You are a billing specialist. Address this customer's concern:

    Query: {query}

    Provide:
    1. Immediate clarification on their billing question
    2. Steps to resolve if it's an issue
    3. Contact info for billing department if needed
    """

    response = llm.invoke(billing_prompt)

    new_state = state.copy()
    new_state["agent_responses"].append(response.content)
    new_state["resolution_status"] = "resolved"

    return new_state


def handle_technical(state: SupportAgentState) -> SupportAgentState:
    """
    Technical branch: Troubleshoot technical issues.

    In production, this would query knowledge bases, run diagnostics,
    and potentially trigger automated fixes.
    """
    query = state["user_query"]

    # Check conversation history for context (multi-turn support)
    history_context = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in state["conversation_history"][-3:]  # Last 3 turns
    ])

    technical_prompt = f"""
    You are a technical support specialist.

    Conversation history:
    {history_context}

    Current issue: {query}

    Provide:
    1. Diagnostic questions (if needed)
    2. Step-by-step troubleshooting
    3. Expected resolution timeline
    """

    response = llm.invoke(technical_prompt)

    new_state = state.copy()
    new_state["agent_responses"].append(response.content)

    # Technical issues might require follow-up
    if "needs_follow_up" in response.content.lower():
        new_state["resolution_status"] = "pending"
    else:
        new_state["resolution_status"] = "resolved"

    return new_state


def handle_general(state: SupportAgentState) -> SupportAgentState:
    """
    General branch: Answer common questions via FAQ retrieval.

    In production, this would use RAG (Retrieval-Augmented Generation)
    to fetch relevant documentation and generate answers.
    """
    query = state["user_query"]

    # Simulate FAQ retrieval (in production, use vector DB)
    faq_context = retrieve_faqs(query)  # Pseudo-function

    general_prompt = f"""
    Answer this general question using the FAQ content:

    FAQ Content:
    {faq_context}

    Question: {query}

    Provide a clear, concise answer with links to relevant documentation.
    """

    response = llm.invoke(general_prompt)

    new_state = state.copy()
    new_state["agent_responses"].append(response.content)
    new_state["resolution_status"] = "resolved"

    return new_state


def handle_escalation(state: SupportAgentState) -> SupportAgentState:
    """
    Escalation branch: Route to human agent.

    In production, this would create a ticket, notify on-call support,
    and provide the human with full context.
    """
    new_state = state.copy()

    escalation_message = f"""
    This inquiry has been escalated to a human support specialist.

    Your case number: {state['metadata']['case_id']}
    Expected response time: 30 minutes

    A support specialist will review your conversation history and
    reach out shortly.
    """

    new_state["agent_responses"].append(escalation_message)
    new_state["resolution_status"] = "escalated"
    new_state["metadata"]["escalation_time"] = get_timestamp()

    # Trigger human notification (simulated)
    notify_human_agent(state)

    return new_state
