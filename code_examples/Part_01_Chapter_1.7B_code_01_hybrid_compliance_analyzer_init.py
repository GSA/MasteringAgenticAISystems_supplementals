"""
Hybrid RAG + Knowledge Graph Compliance System

Demonstrates:
- Combining vector retrieval with graph expansion
- Entity linking from documents to graph
- Context enrichment using graph relationships
- Decision logic for when to use which approach
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from typing import List, Dict, Any
import spacy

class HybridComplianceAnalyzer:
    """
    Combines vector-based RAG with knowledge graph reasoning
    for compliance analysis requiring both semantic understanding
    and precise relationship tracking.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        documents: List[str] = None
    ):
        """
        Initialize hybrid retrieval system.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Database username
            neo4j_password: Database password
            documents: Initial corpus for vector indexing
        """
        # Initialize vector store for semantic search
        self.embeddings = OpenAIEmbeddings()
        if documents:
            self.vector_store = FAISS.from_texts(
                documents,
                self.embeddings
            )
        else:
            self.vector_store = None

        # Initialize knowledge graph for relationship traversal
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password
        )

        # Initialize LLM for synthesis
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Entity extraction for linking documents to graph
        self.nlp = spacy.load("en_core_web_sm")
