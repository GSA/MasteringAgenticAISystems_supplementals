from neo4j import GraphDatabase
from typing import List, Dict, Any

class CompanyKnowledgeGraph:
    """Manages company and investor knowledge graph in Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
