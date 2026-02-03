class DataTransformer:
    """
    Data transformation with quality validation and chunking.

    This class implements the Transform phase of ETL, converting raw
    extracted documents into clean, semantically chunked units ready
    for embedding and vector database insertion.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def transform_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Complete transformation pipeline from raw documents to embeddings-ready chunks.

        Pipeline stages:
        1. Clean text (remove HTML, normalize whitespace)
        2. Validate quality (length, content checks)
        3. Detect duplicates (content hashing)
        4. Chunk documents (semantic boundaries)
        5. Extract metadata (category, tags, timestamps)

        Args:
            documents: Raw documents from extraction phase

        Returns:
            List of transformed chunks with text and metadata
        """
        logger.info(f"Transforming {len(documents)} documents...")

        transformed_chunks = []
        stats = {
            "input_docs": len(documents),
            "quality_rejected": 0,
            "duplicates_removed": 0,
            "chunks_created": 0
        }

        seen_hashes = set()

        for doc in documents:
            # Step 1: Clean text
            cleaned_text = self.clean_text(doc["content"])

            # Step 2: Quality validation
            if not self.validate_quality(cleaned_text):
                stats["quality_rejected"] += 1
                logger.debug(f"Rejected low-quality doc: {doc.get('id')}")
                continue

            # Step 3: Duplicate detection
            doc_hash = self.compute_hash(cleaned_text)
            if doc_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                logger.debug(f"Removed duplicate doc: {doc.get('id')}")
                continue

            seen_hashes.add(doc_hash)

            # Step 4: Chunk document
            chunks = self.chunk_text(cleaned_text)

            # Step 5: Create chunk objects with metadata
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "text": chunk_text,
                    "source_id": doc.get("id"),
                    "title": doc.get("title", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "metadata": self.extract_metadata(doc),
                    "hash": self.compute_hash(chunk_text)
                }

                transformed_chunks.append(chunk)
                stats["chunks_created"] += 1

        logger.info(f"✓ Transformation complete:")
        logger.info(f"  Input documents: {stats['input_docs']}")
        logger.info(f"  Quality rejected: {stats['quality_rejected']}")
        logger.info(f"  Duplicates removed: {stats['duplicates_removed']}")
        logger.info(f"  Chunks created: {stats['chunks_created']}")

        return transformed_chunks
