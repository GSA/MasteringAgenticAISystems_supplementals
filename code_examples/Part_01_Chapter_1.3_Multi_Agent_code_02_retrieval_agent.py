# Retrieval Agent performs semantic search
class RetrievalAgent:
    def retrieve_context(self, ticket_context: TicketContext) -> RetrievedDocs:
        """Searches knowledge base for relevant documentation."""
        # Embed customer query using MCP-connected embedding service
        query_embedding = self.mcp_client.invoke_tool(
            "embedding://generate",
            {"text": ticket_context.message}
        )

        # Vector similarity search against knowledge base
        docs = self.vector_db.similarity_search(
            query_embedding,
            top_k=5,
            filters={"product": ticket_context.account.product_tier}
        )

        # Return structured retrieval results
        return RetrievedDocs(
            documents=docs,
            relevance_scores=[d.score for d in docs],
            retrieval_method="semantic_search"
        )