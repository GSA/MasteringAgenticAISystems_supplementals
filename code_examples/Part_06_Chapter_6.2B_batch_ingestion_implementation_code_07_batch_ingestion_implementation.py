def ingest_documents_batch(client, documents: List[Dict], batch_size: int = 100):
    """
    Batch ingest documents for optimal performance.

    Weaviate best practices:
    - Use batch API (not individual inserts)
    - Batch size 100-500 optimal
    - Monitor for errors
    """

    print(f"Ingesting {len(documents)} documents in batches of {batch_size}")

    with client.batch as batch:
        batch.batch_size = batch_size

        # Configure batch behavior
        batch.dynamic = True  # Auto-adjust batch size
        batch.timeout_retries = 3
        batch.callback = lambda results: print(f"Batch complete: {len(results)} objects")
