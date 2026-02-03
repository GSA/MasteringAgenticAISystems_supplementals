from typing import List

class SimpleRAGSystem:
    """RAG system with document ingestion pipeline."""

    def ingest_document(self, document: str, doc_id: str):
        """
        Ingest a document: chunk and generate embeddings.

        Steps:
        1. Split document into chunks
        2. Generate embeddings for each chunk
        3. Store chunks and embeddings
        """
        print(f"Ingesting document: {doc_id}")

        # Step 1: Chunk document
        chunks = self._chunk_document(document, doc_id)
        print(f"  Created {len(chunks)} chunks")

        # Step 2: Generate embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self._generate_embeddings(texts)
        print(f"  Generated {len(embeddings)} embeddings")

        # Step 3: Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.chunks.append(chunk)

        # Rebuild embeddings matrix for fast search
        self._rebuild_embeddings_matrix()

        print(f"  Total chunks in knowledge base: {len(self.chunks)}")
