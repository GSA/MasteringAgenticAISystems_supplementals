from neo4j import GraphDatabase

class KnowledgeGraphBuilder:
    """
    Constructs knowledge graph from unstructured text documents.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str
    ):
        # Initialize extractors
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()

        # Neo4j connection
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )

    def process_documents(self, documents: List[str]) -> None:
        """
        Process documents and populate knowledge graph.

        Args:
            documents: List of text documents

        Example:
            >>> builder.process_documents([
            ...     "Elon Musk founded SpaceX in 2002.",
            ...     "SpaceX developed the Falcon 9 reusable rocket."
            ... ])
        """
        for doc_text in documents:
            # Extract entities and relationships
            entities = self.entity_extractor.extract_entities(doc_text)
            relationships = self.relationship_extractor.extract_relationships(doc_text)

            # Populate graph
            self._populate_graph(entities, relationships)

    def _populate_graph(
        self,
        entities: Set[Tuple[str, str]],
        relationships: List[Tuple[str, str, str]]
    ) -> None:
        """
        Populate Neo4j graph with extracted entities and relationships.

        Args:
            entities: Set of (entity_text, entity_type) tuples
            relationships: List of (subject, relation, object) triples
        """
        with self.driver.session() as session:
            # Create entity nodes using MERGE for idempotency
            for entity_text, entity_type in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    """,
                    name=entity_text,
                    type=entity_type
                )

            # Create relationships between entities
            for subject, relation, obj in relationships:
                session.run(
                    f"""
                    MATCH (s:Entity {{name: $subject}})
                    MATCH (o:Entity {{name: $object}})
                    MERGE (s)-[r:{relation}]->(o)
                    """,
                    subject=subject,
                    object=obj
                )

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
