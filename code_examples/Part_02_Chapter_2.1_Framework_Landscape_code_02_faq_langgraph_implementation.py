class FAQState(TypedDict):
    question: str
    query_embedding: List[float]
    retrieved_articles: List[dict]
    answer: str

def embed_node(state: FAQState) -> FAQState:
    state["query_embedding"] = embed_text(state["question"])
    return state

def search_node(state: FAQState) -> FAQState:
    state["retrieved_articles"] = vector_db.search(state["query_embedding"])
    return state

def generate_node(state: FAQState) -> FAQState:
    state["answer"] = llm.generate(state["question"], state["retrieved_articles"])
    return state

graph = StateGraph(FAQState)
graph.add_node("embed", embed_node)
graph.add_node("search", search_node)
graph.add_node("generate", generate_node)
graph.add_edge("embed", "search")
graph.add_edge("search", "generate")
graph.set_entry_point("embed")
graph.set_finish_point("generate")
