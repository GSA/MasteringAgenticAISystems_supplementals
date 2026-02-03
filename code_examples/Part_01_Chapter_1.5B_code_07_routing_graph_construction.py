# ====================================================================
# ROUTING LOGIC (CONDITIONAL EDGES)
# ====================================================================

def route_query(state: SupportAgentState) -> Literal["billing", "technical", "general", "escalation"]:
    """
    Routing function that determines which branch to follow.

    This implements the conditional logic of the decision tree.
    LangGraph uses this to create dynamic control flow.
    """
    category = state["query_category"]

    # Map categories to node names
    routing_map = {
        "billing": "billing",
        "technical": "technical",
        "general": "general",
        "escalation": "escalation"
    }

    return routing_map.get(category, "general")  # Default to general if unknown


# ====================================================================
# GRAPH CONSTRUCTION (LOGIC TREE STRUCTURE)
# ====================================================================

def build_support_agent_graph():
    """
    Construct the LangGraph representing the decision tree.

    Graph structure:
                     START
                       ↓
                  categorize_query
                       ↓
                [route decision]
                   ↙  ↓  ↓  ↘
            billing tech general escalation
                   ↘  ↓  ↓  ↙
                      END
    """
    # Initialize graph with state schema
    graph = StateGraph(SupportAgentState)

    # Add nodes (decision tree branches)
    graph.add_node("categorize", categorize_query)
    graph.add_node("billing", handle_billing)
    graph.add_node("technical", handle_technical)
    graph.add_node("general", handle_general)
    graph.add_node("escalation", handle_escalation)

    # Set entry point
    graph.set_entry_point("categorize")

    # Add conditional edges (routing logic)
    graph.add_conditional_edges(
        "categorize",          # From this node
        route_query,           # Use this function to decide
        {                      # Map outputs to target nodes
            "billing": "billing",
            "technical": "technical",
            "general": "general",
            "escalation": "escalation"
        }
    )

    # All terminal nodes go to END
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)
    graph.add_edge("escalation", END)

    # Compile the graph
    app = graph.compile()

    return app
