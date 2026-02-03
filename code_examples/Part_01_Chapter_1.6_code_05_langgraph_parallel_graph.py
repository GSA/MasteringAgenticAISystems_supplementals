from langgraph.graph import StateGraph

def build_parallel_search_graph():
    graph = StateGraph(ResearchState)

    # Define parallel search node that processes all sub-questions concurrently
    graph.add_node(
        "search_all",
        parallel_search_handler,  # Function that handles parallel execution
        parallel=True  # Signals LangGraph to execute this node's operations concurrently
    )

    graph.add_edge("decompose", "search_all")
    graph.add_edge("search_all", "synthesize")

    return graph.compile()
