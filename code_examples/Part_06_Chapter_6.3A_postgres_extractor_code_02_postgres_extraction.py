class DataExtractor:
    """
    Multi-source data extraction for enterprise RAG.

    This class demonstrates extraction patterns applicable across
    SQL databases, REST APIs, and file systems. Each method handles
    source-specific concerns while presenting a uniform interface
    to downstream transformation logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def extract_from_postgres(self, since: datetime = None) -> List[Dict[str, Any]]:
        """
        Extract documents from PostgreSQL database with incremental update support.

        Args:
            since: Extract only records updated after this timestamp.
                   If None, performs full extraction.

        Returns:
            List of document dictionaries with content and metadata.
        """
        logger.info("Extracting from PostgreSQL database...")

        source_config = self.config["sources"]["postgres"]
        engine = create_engine(source_config["connection_string"])

        # Build query with optional incremental filter
        query = f"""
        SELECT id, title, content, updated_at, category, tags
        FROM {source_config['table']}
        """

        if since:
            query += f"""
            WHERE {source_config['timestamp_column']} > :since
            """

        # Execute query with proper parameter binding
        with engine.connect() as conn:
            if since:
                df = pd.read_sql(text(query), conn, params={"since": since})
            else:
                df = pd.read_sql(text(query), conn)

        # Convert DataFrame to list of dictionaries
        documents = df.to_dict('records')

        logger.info(f"✓ Extracted {len(documents)} documents from PostgreSQL")

        return documents
