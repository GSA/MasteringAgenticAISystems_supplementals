# Milvus HNSW configuration
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 24,              # Bi-directional links
        "efConstruction": 300  # Build-time quality
    }
}

search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 100  # Query-time accuracy
    }
}
