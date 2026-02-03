from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import connections, utility
import logging

logger = logging.getLogger(__name__)


class VectorDatabaseLoader:
    """
    Load transformed data into vector database (Milvus).

    Handles collection creation, batch insertion, indexing,
    and state management for production RAG systems.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False

    def connect(self):
        """Establish connection to Milvus vector database."""
        if self.connected:
            return

        connections.connect(
            alias="default",
            host=self.config["vector_db"]["host"],
            port=self.config["vector_db"]["port"]
        )

        self.connected = True
        logger.info("✓ Connected to Milvus")

    def _get_or_create_collection(self, name: str) -> Collection:
        """Get existing collection or create new one with proper schema."""
        if utility.has_collection(name):
            logger.info(f"Using existing collection: {name}")
            return Collection(name)

        # Create new collection with explicit schema
        logger.info(f"Creating new collection: {name}")

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config["vector_db"]["embedding_dim"]
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name="source_id",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=65535
            )
        ]

        schema = CollectionSchema(
            fields,
            description="Support knowledge base"
        )

        collection = Collection(name, schema)

        logger.info(f"✓ Created collection: {name}")

        return collection
