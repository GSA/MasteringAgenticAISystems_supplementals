# Multi-tier caching configuration for RAG system

# Layer 1: Document Embedding Cache (Vector Database)
vector_db_config = {
    "index_type": "HNSW",  # Fast approximate retrieval
    "cache_size": "unlimited",  # All documents cached
    "invalidation": "on_model_update"  # Only when embedding model changes
}

# Layer 2: Query Embedding Cache (Semantic Similarity)
query_cache_config = {
    "cache_size": 10000,  # 10K most frequent queries
    "similarity_threshold": 0.93,  # Semantic matching tolerance
    "eviction_policy": "LRU",  # Least recently used
    "ttl_seconds": 3600  # 1 hour expiration
}

# Layer 3: Tool Call Result Cache (API Responses)
tool_cache_config = {
    "cache_size": 50000,  # Cache all recent tool results
    "ttl_seconds": 300,  # 5-minute expiration
    "invalidation": "on_write",  # Invalidate on data updates
    "include_tools": ["get_api_reference", "search_docs"],  # READ operations only
    "exclude_tools": ["submit_feedback", "log_analytics"]  # WRITE operations
}
