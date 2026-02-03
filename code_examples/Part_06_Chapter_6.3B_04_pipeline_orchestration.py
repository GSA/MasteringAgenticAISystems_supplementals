import json
from datetime import datetime, timedelta
from pathlib import Path


class ETLPipeline:
    """
    Complete ETL pipeline orchestrator.

    Coordinates extraction, transformation, and loading phases
    with incremental update support and state management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extractor = DataExtractor(config)
        self.transformer = DataTransformer(config)
        self.loader = VectorDatabaseLoader(config)

    def run(self, incremental: bool = True):
        """
        Execute full ETL pipeline.

        Phases:
        1. Extract - Pull data from sources (DB, API, files)
        2. Transform - Clean, validate, chunk, deduplicate
        3. Embed - Generate vector representations
        4. Load - Insert into vector database with indexing

        Args:
            incremental (bool): Process only changes since last run
        """
        logger.info("="*70)
        logger.info("STARTING ETL PIPELINE")
        logger.info("="*70)

        start_time = datetime.now()

        # Determine extraction window for incremental updates
        since = None
        if incremental and self.config["incremental"]["enabled"]:
            since = self._get_last_run_time()
            logger.info(f"Incremental mode: extracting since {since}")

        # PHASE 1: EXTRACT
        logger.info("\n[PHASE 1] EXTRACTION")
        documents = []

        documents.extend(self.extractor.extract_from_postgres(since))
        # Add other sources as needed:
        # documents.extend(self.extractor.extract_from_api(..., since))
        # documents.extend(self.extractor.extract_from_files(..., since))

        logger.info(f"Total extracted: {len(documents)} documents")

        if len(documents) == 0:
            logger.info("No new documents to process. Exiting.")
            return

        # PHASE 2: TRANSFORM
        logger.info("\n[PHASE 2] TRANSFORMATION")
        chunks = self.transformer.transform_documents(documents)

        if len(chunks) == 0:
            logger.warning(
                "No chunks survived transformation. "
                "Check quality filters."
            )
            return

        # PHASE 3: EMBEDDING GENERATION
        logger.info("\n[PHASE 3] EMBEDDING GENERATION")
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Production implementation would use actual embedding model:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # embeddings = model.encode([c["text"] for c in chunks])

        # Placeholder for demonstration:
        embeddings = [[0.1] * 1024 for _ in chunks]

        logger.info(f"✓ Generated {len(embeddings)} embeddings")

        # PHASE 4: LOAD
        logger.info("\n[PHASE 4] LOADING")
        loaded_count = self.loader.load_documents(chunks, embeddings)

        # Update state for next incremental run
        if incremental:
            self._save_run_state(datetime.now())

        duration = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*70)
        logger.info("ETL PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks loaded: {loaded_count}")
        logger.info("="*70)

    def _get_last_run_time(self) -> datetime:
        """
        Get timestamp of last successful ETL run.

        Enables incremental extraction by tracking processing windows.
        Defaults to configured lookback period if no state exists.

        Returns:
            datetime: Timestamp to use for incremental extraction
        """
        state_file = Path(self.config["incremental"]["state_file"])

        if state_file.exists():
            state = json.loads(state_file.read_text())
            return datetime.fromisoformat(state["last_run"])

        # Default: look back configured hours for first run
        return datetime.now() - timedelta(
            hours=self.config["incremental"]["lookback_hours"]
        )

    def _save_run_state(self, timestamp: datetime):
        """
        Save ETL run state for incremental tracking.

        Records timestamp of successful completion to enable
        next run to process only subsequent changes.

        Args:
            timestamp (datetime): Completion timestamp to record
        """
        state_file = Path(self.config["incremental"]["state_file"])

        state = {
            "last_run": timestamp.isoformat(),
            "status": "success"
        }

        state_file.write_text(json.dumps(state, indent=2))
