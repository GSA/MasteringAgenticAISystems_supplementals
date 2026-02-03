def hybrid_search(client, query: str, limit: int = 10, alpha: float = 0.5):
    """
    Hybrid search combining vector similarity and BM25.

    Args:
        query: Search query
        limit: Number of results
        alpha: Weight (0=pure BM25, 1=pure vector, 0.5=balanced)
    """

    result = (
        client.query
        .get("Document", ["content", "source_doc", "chunk_index", "metadata"])
        .with_hybrid(
            query=query,
            alpha=alpha,  # Fusion weight
            properties=["content"]  # Fields to search
        )
        .with_limit(limit)
        .with_additional(["score", "explainScore"])  # Include relevance scores
        .do()
    )

    return result["data"]["Get"]["Document"]
