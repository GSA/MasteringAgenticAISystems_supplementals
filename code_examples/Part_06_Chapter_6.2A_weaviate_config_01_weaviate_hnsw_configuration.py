# Weaviate HNSW configuration
collection_config = {
    "class": "Document",
    "vectorizer": "none",  # Using external embeddings
    "vectorIndexConfig": {
        "distance": "cosine",
        "efConstruction": 300,  # High quality index
        "maxConnections": 24,   # M parameter
        "ef": 100              # Query-time search depth
    }
}
