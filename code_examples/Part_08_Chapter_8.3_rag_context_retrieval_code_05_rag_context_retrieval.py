# RAG-Optimized Context Retrieval
async def optimized_market_context(client_profile: dict, query: str) -> str:
    """Retrieve only relevant market data based on client holdings and query intent"""

    # Construct retrieval query combining portfolio holdings and user question
    # This focuses search on market data relevant to client's investments
    retrieval_query = f"""
    Client holdings: {', '.join(client_profile['holdings'])}
    Query: {query}
    """

    # Vector search for semantically relevant market data
    relevant_data = await vector_search(
        query=retrieval_query,
        top_k=5,  # Retrieve top 5 most relevant documents
        collection="market_data"
    )

    # Assemble retrieved snippets into context
    # This typically yields ~1,200 tokens vs 3,000 for full market data
    context = "\n\n".join(doc.content for doc in relevant_data)

    return context
