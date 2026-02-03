    def answer_with_hybrid_context(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Answer question using hybrid RAG + KG approach.

        Demonstrates complete flow:
        - Semantic retrieval finds relevant documents
        - Entity linking identifies graph starting points
        - Graph expansion provides relationship context
        - LLM synthesis combines both information sources

        Args:
            query: User question (e.g., compliance analysis query)

        Returns:
            Answer with both semantic and relational support
        """
        # Retrieve hybrid context
        context = self.hybrid_retrieve(query)

        # Format document context from vector search
        vector_context = "\n\n".join([
            f"Document {i+1}: {doc}"
            for i, doc in enumerate(context["vector_results"])
        ])

        # Format graph context from relationship expansion
        graph_context = "\n".join([
            f"- {item['center']} is connected to: "
            f"{', '.join(item['connected_entities'])} "
            f"(via relationships: {', '.join(item['relationship_types'])})"
            for item in context["graph_context"]
        ])

        # Synthesize answer using both sources
        prompt = f"""
        Answer the following compliance question using both document
        context (for policy rules and definitions) and knowledge graph
        relationships (for entity connections and potential conflicts).

        Question: {query}

        Document Context (semantic retrieval):
        {vector_context}

        Knowledge Graph Relationships (traversal context):
        {graph_context}

        Instructions:
        - Use document content for policy rules and compliance criteria
        - Use graph relationships to identify entity connections
        - Cite specific relationships when identifying potential issues
        - Combine both sources for complete compliance analysis

        Answer:
        """

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "vector_documents": len(context["vector_results"]),
            "entities_linked": len(context["entities"]),
            "graph_nodes_explored": len(context["graph_context"])
        }
