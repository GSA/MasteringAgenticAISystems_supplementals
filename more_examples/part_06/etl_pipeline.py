"""
Code Example 6.3.1: Enterprise RAG ETL Pipeline

Purpose: Demonstrate complete Extract-Transform-Load pipeline for knowledge bases

Concepts Demonstrated:
- Multi-source extraction: SQL databases, REST APIs, file systems
- Data transformation: Cleaning, validation, chunking, deduplication
- Vector database loading: Milvus with proper indexing
- Incremental updates: State tracking and change detection

Prerequisites:
- SQL basics (SELECT queries)
- Understanding of vector databases
- Familiarity with data quality concepts

Author: NVIDIA Certified Generative AI LLM Course
Chapter: 6, Section: 6.3
Exam Skill: 6.3 - Build ETL Pipelines to Integrate Enterprise Data Sources
"""

# ============================================================================
# IMPORTS
# ============================================================================

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib
import json
import time

# Data processing
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "sources": {
        "database": {
            "connection": "postgresql://user:pass@localhost/kb",
            "table": "knowledge_base",
            "timestamp_col": "updated_at"
        }
    },
    "chunking": {
        "chunk_size_tokens": 512,
        "overlap_tokens": 50,
        "min_chunk_chars": 100
    },
    "quality": {
        "min_length": 50,
        "max_length": 100000,
        "detect_duplicates": True
    },
    "incremental": {
        "enabled": True,
        "state_file": "etl_state.json",
        "lookback_hours": 24
    }
}

# ============================================================================
# EXTRACTION
# ============================================================================

class DataExtractor:
    """Extract data from enterprise sources."""

    def __init__(self, config: Dict):
        self.config = config

    def extract_from_database(
        self,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract documents from SQL database.

        Demonstrates incremental extraction using timestamps.
        """
        logger.info("Extracting from database...")

        db_config = self.config["sources"]["database"]

        # Build query
        query = f"SELECT * FROM {db_config['table']}"

        # Add incremental filter
        if since:
            query += f" WHERE {db_config['timestamp_col']} > :since"
            logger.info(f"  Incremental: extracting since {since}")

        # Execute query (simplified - production would use actual connection)
        # engine = create_engine(db_config["connection"])
        # with engine.connect() as conn:
        #     df = pd.read_sql(text(query), conn, params={"since": since})
        # documents = df.to_dict('records')

        # Simulated documents
        documents = [
            {
                "id": 1,
                "title": "NVIDIA NIM Deployment Guide",
                "content": "Complete guide to deploying NVIDIA NIM microservices...",
                "updated_at": datetime.now()
            },
            {
                "id": 2,
                "title": "Triton Inference Server Configuration",
                "content": "Best practices for configuring Triton...",
                "updated_at": datetime.now()
            }
        ]

        logger.info(f"✓ Extracted {len(documents)} documents")
        return documents


# ============================================================================
# TRANSFORMATION
# ============================================================================

class DataTransformer:
    """Transform raw data into AI-ready format."""

    def __init__(self, config: Dict):
        self.config = config
        self.seen_hashes = set()

    def transform(self, documents: List[Dict]) -> List[Dict]:
        """
        Complete transformation pipeline.

        Steps:
        1. Clean text
        2. Validate quality
        3. Detect duplicates
        4. Chunk documents
        5. Add metadata
        """
        logger.info(f"Transforming {len(documents)} documents...")

        chunks = []
        stats = {
            "input": len(documents),
            "quality_rejected": 0,
            "duplicates": 0,
            "chunks_created": 0
        }

        for doc in documents:
            # Clean
            text = self.clean_text(doc["content"])

            # Validate quality
            if not self.validate_quality(text):
                stats["quality_rejected"] += 1
                continue

            # Detect duplicates
            doc_hash = self.compute_hash(text)
            if doc_hash in self.seen_hashes:
                stats["duplicates"] += 1
                continue

            self.seen_hashes.add(doc_hash)

            # Chunk
            doc_chunks = self.chunk_text(text)

            # Create chunk objects
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "text": chunk_text,
                    "source_id": doc["id"],
                    "title": doc["title"],
                    "chunk_index": i,
                    "metadata": {
                        "updated_at": doc["updated_at"].isoformat()
                    }
                })
                stats["chunks_created"] += 1

        logger.info(f"✓ Transformation complete:")
        logger.info(f"  Created {stats['chunks_created']} chunks")
        logger.info(f"  Rejected {stats['quality_rejected']} low-quality docs")
        logger.info(f"  Removed {stats['duplicates']} duplicates")

        return chunks

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters
        text = ''.join(c for c in text if c.isprintable() or c.isspace())

        return text.strip()

    def validate_quality(self, text: str) -> bool:
        """Validate text quality."""
        config = self.config["quality"]

        # Length checks
        if len(text) < config["min_length"]:
            return False
        if len(text) > config["max_length"]:
            return False

        # Word count
        if len(text.split()) < 10:
            return False

        return True

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text at semantic boundaries.

        Strategy: Fixed size with overlap, split on paragraph/sentence boundaries.
        """
        config = self.config["chunking"]

        # Approximate: 1 token ≈ 4 characters
        chunk_size_chars = config["chunk_size_tokens"] * 4
        overlap_chars = config["overlap_tokens"] * 4

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size_chars

            # Find semantic boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    break_point = text.rfind(sep, start, end)
                    if break_point > start:
                        end = break_point + len(sep)
                        break

            chunk = text[start:end].strip()

            if len(chunk) >= config["min_chunk_chars"]:
                chunks.append(chunk)

            start = end - overlap_chars

            if start >= len(text):
                break

        return chunks

    def compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()


# ============================================================================
# LOADING
# ============================================================================

class VectorDatabaseLoader:
    """Load data into vector database."""

    def __init__(self, config: Dict):
        self.config = config

    def load(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Load chunks and embeddings into vector DB.

        In production: Would use actual Milvus/Pinecone/Weaviate client.
        This demonstrates the loading pattern.
        """
        logger.info(f"Loading {len(chunks)} chunks to vector database...")

        # Simulated loading (production would use actual vector DB)
        # from pymilvus import connections, Collection
        # connections.connect(host="localhost", port=19530)
        # collection = Collection("knowledge_base")
        # collection.insert([embeddings, texts, metadata])

        batch_size = 1000
        loaded = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_emb = embeddings[i:i+batch_size]

            # Simulated batch insert
            time.sleep(0.1)  # Simulate network latency

            loaded += len(batch)
            logger.info(f"  Loaded batch {i//batch_size + 1}: {loaded} total")

        logger.info(f"✓ Loaded {loaded} chunks")
        return loaded


# ============================================================================
# ORCHESTRATION
# ============================================================================

class ETLPipeline:
    """Complete ETL pipeline orchestrator."""

    def __init__(self, config: Dict):
        self.config = config
        self.extractor = DataExtractor(config)
        self.transformer = DataTransformer(config)
        self.loader = VectorDatabaseLoader(config)

    def run(self, incremental: bool = True):
        """Execute full ETL pipeline."""
        logger.info("="*70)
        logger.info("ETL PIPELINE START")
        logger.info("="*70)

        start_time = time.time()

        # Get extraction window
        since = None
        if incremental and self.config["incremental"]["enabled"]:
            since = self._get_last_run_time()

        # EXTRACT
        logger.info("\n[PHASE 1] EXTRACTION")
        documents = self.extractor.extract_from_database(since)

        # TRANSFORM
        logger.info("\n[PHASE 2] TRANSFORMATION")
        chunks = self.transformer.transform(documents)

        # GENERATE EMBEDDINGS (simulated)
        logger.info("\n[PHASE 3] EMBEDDING")
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        # embeddings = embedding_model.encode([c["text"] for c in chunks])
        embeddings = [[0.1] * 768 for _ in chunks]  # Simulated
        logger.info(f"✓ Generated {len(embeddings)} embeddings")

        # LOAD
        logger.info("\n[PHASE 4] LOADING")
        loaded = self.loader.load(chunks, embeddings)

        # Save state
        if incremental:
            self._save_state(datetime.now())

        duration = time.time() - start_time

        logger.info("\n" + "="*70)
        logger.info("ETL PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Documents: {len(documents)}")
        logger.info(f"Chunks loaded: {loaded}")
        logger.info("="*70)

    def _get_last_run_time(self) -> datetime:
        """Get last successful ETL run time."""
        state_file = Path(self.config["incremental"]["state_file"])

        if state_file.exists():
            state = json.loads(state_file.read_text())
            return datetime.fromisoformat(state["last_run"])

        # Default lookback
        return datetime.now() - timedelta(
            hours=self.config["incremental"]["lookback_hours"]
        )

    def _save_state(self, timestamp: datetime):
        """Save ETL state."""
        state_file = Path(self.config["incremental"]["state_file"])

        state = {
            "last_run": timestamp.isoformat(),
            "status": "success"
        }

        state_file.write_text(json.dumps(state, indent=2))
        logger.info(f"✓ Saved state to {state_file}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Run ETL pipeline example."""
    print("\n" + "="*70)
    print("Enterprise RAG ETL Pipeline Example")
    print("="*70)

    # Create pipeline
    pipeline = ETLPipeline(CONFIG)

    # Run pipeline
    pipeline.run(incremental=True)

    print("\n✅ ETL pipeline complete!")


if __name__ == "__main__":
    main()
