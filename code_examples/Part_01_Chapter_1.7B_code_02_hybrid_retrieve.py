    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        expand_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Perform hybrid retrieval: vector search + graph expansion.

        This implements the retrieval-augmented knowledge graph pattern:
        1. Vector search finds semantically relevant documents
        2. Entity extraction links document mentions to graph nodes
        3. Graph traversal expands context around those entities

        Args:
            query: User question requiring hybrid retrieval
            top_k: Number of documents to retrieve via vector search
            expand_hops: Number of graph hops for context expansion

        Returns:
            Combined context from vectors and graph with entity links
        """
        # Step 1: Vector similarity search for semantic relevance
        vector_docs = self.vector_store.similarity_search(
            query,
            k=top_k
        )

        # Step 2: Extract entities from retrieved documents
        # This links unstructured text to structured graph nodes
        entities = self._extract_entities_from_docs(vector_docs)

        # Step 3: Expand context using knowledge graph
        # For each entity mentioned in retrieved docs, explore
        # surrounding graph structure to find related entities
        # and relationships that provide additional context
        graph_context = []
        for entity in entities:
            subgraph = self.graph.query(
                f"""
                MATCH (e {{name: $entity}})-[r*1..{expand_hops}]-(related)
                RETURN e.name AS center,
                       collect(DISTINCT related.name) AS connected_entities,
                       collect(DISTINCT type(r[0])) AS relationship_types
                LIMIT 20
                """,
                params={"entity": entity}
            )
            graph_context.extend(subgraph)

        return {
            "vector_results": [doc.page_content for doc in vector_docs],
            "entities": entities,
            "graph_context": graph_context
        }
