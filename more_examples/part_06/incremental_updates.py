"""
Code Example 6.3.2: Incremental ETL with Change Detection

Purpose: Demonstrate efficient incremental updates for large knowledge bases

Concepts Demonstrated:
- State tracking: Persist last successful run information
- Change detection: Query only new/modified data
- Upsert operations: Update existing + insert new records
- File-based change detection: Monitor modification times
- Performance optimization: Process only deltas, not full dataset

Prerequisites:
- Understanding of timestamps and datetime comparisons
- Basic file system operations
- SQL WHERE clause filtering

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
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class ETLStateManager:
    """
    Manage ETL state for incremental processing.

    Tracks:
    - Last successful run timestamp
    - Document counts for validation
    - Processing status
    - Error information
    """

    def __init__(self, state_file: str = "etl_state.json"):
        """
        Initialize state manager.

        Args:
            state_file (str): Path to JSON state file
        """
        self.state_file = Path(state_file)

    def get_last_run(self) -> Optional[datetime]:
        """
        Get timestamp of last successful ETL run.

        Returns:
            datetime: Last run timestamp, or None if first run
        """
        if not self.state_file.exists():
            logger.info("No previous state found - first run")
            return None

        state = json.loads(self.state_file.read_text())

        if state.get("status") != "success":
            logger.warning(
                f"Last run status: {state.get('status')} - "
                "may need full refresh"
            )

        last_run = state.get("last_run")
        if last_run:
            return datetime.fromisoformat(last_run)

        return None

    def save_run_state(
        self,
        timestamp: datetime,
        doc_count: int,
        chunk_count: int,
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Save ETL run state.

        Args:
            timestamp (datetime): Run timestamp
            doc_count (int): Documents processed
            chunk_count (int): Chunks created
            status (str): Run status (success/failed/partial)
            error (str, optional): Error message if failed
        """
        state = {
            "last_run": timestamp.isoformat(),
            "status": status,
            "documents_processed": doc_count,
            "chunks_created": chunk_count,
            "error": error
        }

        self.state_file.write_text(json.dumps(state, indent=2))

        logger.info(f"✓ Saved state: {status}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Documents: {doc_count}")
        logger.info(f"  Chunks: {chunk_count}")

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state information."""
        if not self.state_file.exists():
            return {}

        return json.loads(self.state_file.read_text())


# ============================================================================
# INCREMENTAL EXTRACTION
# ============================================================================

class IncrementalExtractor:
    """Extract only changed/new documents."""

    def __init__(self, state_manager: ETLStateManager):
        self.state_manager = state_manager

    def extract_changed_documents(
        self,
        source_config: Dict[str, Any],
        default_lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Extract documents changed since last run.

        Demonstrates incremental extraction pattern:
        1. Get last run timestamp from state
        2. Query with WHERE timestamp > last_run
        3. Return only changed/new documents

        Args:
            source_config (Dict): Data source configuration
            default_lookback_hours (int): Default lookback if no state

        Returns:
            List[Dict]: Changed documents
        """
        # Get extraction window
        last_run = self.state_manager.get_last_run()

        if last_run is None:
            # First run - use default lookback
            since = datetime.now() - timedelta(hours=default_lookback_hours)
            logger.info(
                f"First run: extracting last {default_lookback_hours} hours"
            )
        else:
            since = last_run
            logger.info(f"Incremental: extracting since {since}")

        # Build query with timestamp filter
        # Production SQL example:
        # query = f"""
        #     SELECT * FROM {source_config['table']}
        #     WHERE {source_config['timestamp_col']} > :since
        #     ORDER BY {source_config['timestamp_col']} ASC
        # """

        # Simulated extraction
        documents = [
            {
                "id": 101,
                "title": "New Document",
                "content": "This is new content added after last run...",
                "updated_at": datetime.now()
            },
            {
                "id": 52,  # Updated existing document
                "title": "Updated Document",
                "content": "This content was modified...",
                "updated_at": datetime.now()
            }
        ]

        logger.info(f"✓ Extracted {len(documents)} changed documents")

        return documents

    def extract_changed_files(
        self,
        directory: Path,
        pattern: str = "**/*.txt"
    ) -> List[Dict[str, Any]]:
        """
        Extract files modified since last run.

        Demonstrates file-based change detection using modification time.

        Args:
            directory (Path): Root directory
            pattern (str): Glob pattern

        Returns:
            List[Dict]: Changed files with metadata
        """
        last_run = self.state_manager.get_last_run()

        if last_run is None:
            # First run - process all files
            since = datetime.min
            logger.info("First run: processing all files")
        else:
            since = last_run
            logger.info(f"Incremental: files modified since {since}")

        documents = []

        for file_path in directory.glob(pattern):
            # Check modification time
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

            if mtime <= since:
                continue  # Skip unchanged file

            try:
                content = file_path.read_text(encoding='utf-8')

                doc = {
                    "id": str(file_path),
                    "title": file_path.stem,
                    "content": content,
                    "updated_at": mtime,
                    "source": "filesystem",
                    "path": str(file_path)
                }

                documents.append(doc)

            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

        logger.info(f"✓ Found {len(documents)} changed files")

        return documents


# ============================================================================
# UPSERT OPERATIONS
# ============================================================================

class IncrementalLoader:
    """
    Load data with upsert capability.

    Upsert = Update if exists, Insert if new
    """

    def __init__(self):
        self.existing_ids = set()  # Track existing document IDs

    def upsert_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Dict[str, int]:
        """
        Upsert chunks into vector database.

        Logic:
        - If chunk exists (by source_id + chunk_index): UPDATE
        - If chunk is new: INSERT

        Args:
            chunks (List[Dict]): Document chunks
            embeddings (List[List[float]]): Vector embeddings

        Returns:
            Dict: Counts of inserts and updates
        """
        logger.info(f"Upserting {len(chunks)} chunks...")

        stats = {"inserted": 0, "updated": 0}

        for chunk, embedding in zip(chunks, embeddings):
            # Create unique identifier
            chunk_id = self._create_chunk_id(
                chunk["source_id"],
                chunk["chunk_index"]
            )

            if chunk_id in self.existing_ids:
                # UPDATE existing chunk
                self._update_chunk(chunk_id, chunk, embedding)
                stats["updated"] += 1
            else:
                # INSERT new chunk
                self._insert_chunk(chunk_id, chunk, embedding)
                self.existing_ids.add(chunk_id)
                stats["inserted"] += 1

        logger.info(f"✓ Upsert complete:")
        logger.info(f"  Inserted: {stats['inserted']}")
        logger.info(f"  Updated: {stats['updated']}")

        return stats

    def _create_chunk_id(self, source_id: str, chunk_index: int) -> str:
        """Create unique chunk identifier."""
        return f"{source_id}_{chunk_index}"

    def _insert_chunk(
        self,
        chunk_id: str,
        chunk: Dict,
        embedding: List[float]
    ):
        """
        Insert new chunk.

        Production would execute:
        INSERT INTO vector_db (id, embedding, text, metadata)
        VALUES (:id, :embedding, :text, :metadata)
        """
        logger.debug(f"Inserting chunk: {chunk_id}")

    def _update_chunk(
        self,
        chunk_id: str,
        chunk: Dict,
        embedding: List[float]
    ):
        """
        Update existing chunk.

        Production would execute:
        UPDATE vector_db
        SET embedding = :embedding, text = :text, metadata = :metadata
        WHERE id = :id
        """
        logger.debug(f"Updating chunk: {chunk_id}")


# ============================================================================
# INCREMENTAL ETL PIPELINE
# ============================================================================

class IncrementalETLPipeline:
    """Complete incremental ETL pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.state_manager = ETLStateManager(
            config.get("state_file", "etl_state.json")
        )

        self.extractor = IncrementalExtractor(self.state_manager)
        self.loader = IncrementalLoader()

    def run(self):
        """
        Execute incremental ETL pipeline.

        Workflow:
        1. Load state from previous run
        2. Extract only changed documents
        3. Transform changed documents
        4. Upsert to vector database
        5. Save new state
        """
        logger.info("="*70)
        logger.info("INCREMENTAL ETL PIPELINE")
        logger.info("="*70)

        run_timestamp = datetime.now()

        try:
            # Display previous state
            prev_state = self.state_manager.get_full_state()
            if prev_state:
                logger.info("\nPrevious state:")
                logger.info(f"  Last run: {prev_state.get('last_run')}")
                logger.info(f"  Status: {prev_state.get('status')}")
                logger.info(f"  Documents: {prev_state.get('documents_processed')}")

            # EXTRACT (incremental)
            logger.info("\n[PHASE 1] INCREMENTAL EXTRACTION")

            documents = self.extractor.extract_changed_documents(
                self.config.get("source", {}),
                default_lookback_hours=24
            )

            if not documents:
                logger.info("✓ No changes detected - skipping transformation/loading")
                return

            # TRANSFORM (simplified for this example)
            logger.info("\n[PHASE 2] TRANSFORMATION")

            chunks = []
            for doc in documents:
                # Simple chunking
                chunks.append({
                    "text": doc["content"],
                    "source_id": str(doc["id"]),
                    "chunk_index": 0,
                    "metadata": {"title": doc["title"]}
                })

            logger.info(f"✓ Created {len(chunks)} chunks")

            # GENERATE EMBEDDINGS (simulated)
            logger.info("\n[PHASE 3] EMBEDDING")
            embeddings = [[0.1] * 768 for _ in chunks]
            logger.info(f"✓ Generated {len(embeddings)} embeddings")

            # LOAD (upsert)
            logger.info("\n[PHASE 4] UPSERT TO VECTOR DB")

            upsert_stats = self.loader.upsert_documents(chunks, embeddings)

            # SAVE STATE
            self.state_manager.save_run_state(
                timestamp=run_timestamp,
                doc_count=len(documents),
                chunk_count=len(chunks),
                status="success"
            )

            logger.info("\n" + "="*70)
            logger.info("INCREMENTAL ETL COMPLETE")
            logger.info("="*70)
            logger.info(f"Documents processed: {len(documents)}")
            logger.info(f"Chunks inserted: {upsert_stats['inserted']}")
            logger.info(f"Chunks updated: {upsert_stats['updated']}")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"ETL failed: {e}")

            # Save failure state
            self.state_manager.save_run_state(
                timestamp=run_timestamp,
                doc_count=0,
                chunk_count=0,
                status="failed",
                error=str(e)
            )

            raise


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

def compare_full_vs_incremental():
    """Compare full refresh vs incremental update performance."""
    print("\n" + "="*70)
    print("Performance Comparison: Full vs Incremental ETL")
    print("="*70)

    scenarios = {
        "Small KB (100K docs)": {
            "full_refresh_time": 45,  # minutes
            "incremental_time": 3,
            "daily_change_rate": 0.02  # 2%
        },
        "Medium KB (1M docs)": {
            "full_refresh_time": 480,  # 8 hours
            "incremental_time": 12,
            "daily_change_rate": 0.01  # 1%
        },
        "Large KB (10M docs)": {
            "full_refresh_time": 4800,  # 80 hours
            "incremental_time": 35,
            "daily_change_rate": 0.005  # 0.5%
        }
    }

    print("\n| Scenario | Full Refresh | Incremental | Speedup |")
    print("|----------|-------------|-------------|---------|")

    for scenario, metrics in scenarios.items():
        full_time = metrics["full_refresh_time"]
        incr_time = metrics["incremental_time"]
        speedup = full_time / incr_time

        # Convert to appropriate units
        if full_time < 60:
            full_str = f"{full_time}m"
        else:
            full_str = f"{full_time/60:.1f}h"

        if incr_time < 60:
            incr_str = f"{incr_time}m"
        else:
            incr_str = f"{incr_time/60:.1f}h"

        print(f"| {scenario} | {full_str} | {incr_str} | {speedup:.1f}x |")

    print("\nKey Takeaway:")
    print("  ✓ Incremental updates enable frequent refreshes (hourly vs daily)")
    print("  ✓ 10-100x faster for typical change rates (<5% daily)")
    print("  ✓ Reduced infrastructure costs (shorter compute time)")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Run incremental ETL example."""
    print("\n" + "="*70)
    print("Incremental ETL Pipeline Example")
    print("="*70)

    config = {
        "state_file": "incremental_etl_state.json",
        "source": {
            "table": "knowledge_base",
            "timestamp_col": "updated_at"
        }
    }

    # Create pipeline
    pipeline = IncrementalETLPipeline(config)

    # Run incremental ETL
    pipeline.run()

    # Show performance comparison
    compare_full_vs_incremental()

    print("\n✅ Incremental ETL demonstration complete!")


if __name__ == "__main__":
    main()
