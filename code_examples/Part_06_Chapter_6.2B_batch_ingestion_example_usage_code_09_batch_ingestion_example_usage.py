# Example usage
documents = [
    {
        "content": "NVIDIA H100 provides 3x faster training...",
        "source_doc": "H100_specs",
        "chunk_index": 0,
        "metadata": {"section": "Performance", "timestamp": "2024-01-15"}
    },
    # ... more documents
]

ingest_documents_batch(client, documents)
