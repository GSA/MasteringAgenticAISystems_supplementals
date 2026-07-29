"""Contextual document-processing helper used with batch setup example 07."""

from __future__ import annotations


def add_documents_to_batch(batch, documents):
    """Add prepared document objects to an already configured batch."""
    for i, doc in enumerate(documents):
        # Add object to batch
        batch.add_data_object(
            data_object={
                "content": doc["content"],
                "source_doc": doc["source_doc"],
                "chunk_index": doc["chunk_index"],
                "metadata": doc.get("metadata", {})
            },
            class_name="Document",
            vector=doc.get("embedding")  # Optional: provide pre-computed embedding
        )

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(documents)} documents")

    print(f"Ingestion complete: {len(documents)} documents indexed")
