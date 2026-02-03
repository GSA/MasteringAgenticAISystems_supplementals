# Create schema (collection)
schema = {
    "class": "Document",
    "description": "Technical documentation chunks",
    "vectorizer": "text2vec-openai",  # Built-in vectorization
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "type": "text"
        }
    },
