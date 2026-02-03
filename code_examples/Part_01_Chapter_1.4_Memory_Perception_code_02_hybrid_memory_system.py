from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sqlite3

class HybridMemorySystem:
    def __init__(self, qdrant_client: QdrantClient, db_path: str):
        # Vector store for semantic memory
        self.vector_store = qdrant_client
        self.collection_name = "knowledge_base"

        # Initialize semantic memory collection
        self.vector_store.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        # SQL database for episodic and procedural memory
        self.db_conn = sqlite3.connect(db_path)
        self._initialize_schemas()

    def _initialize_schemas(self):
        """Create tables for episodic and procedural memory"""

        # Episodic memory: time-ordered event log
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                issue_category TEXT,
                entities TEXT,  -- JSON serialized list
                resolution TEXT,
                satisfaction_score INTEGER
            )
        """)

        # Procedural memory: learned resolution patterns
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS procedural_memory (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_pattern TEXT,  -- category + entity combination
                successful_resolution TEXT,
                success_count INTEGER DEFAULT 0,
                total_attempts INTEGER DEFAULT 0
            )
        """)

        self.db_conn.commit()
