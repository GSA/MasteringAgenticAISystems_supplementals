# Initialize hybrid system with compliance documents
compliance_docs = [
    "Company policy prohibits executives from having financial interests in competitors.",
    "Conflicts of interest include indirect interests through family members.",
    "Competitors are defined as companies operating in the same market segment.",
    "Disclosure is required for all potential conflicts, direct or indirect."
]

analyzer = HybridComplianceAnalyzer(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    documents=compliance_docs
)

# Query requiring both semantic understanding and relationship traversal
result = analyzer.answer_with_hybrid_context(
    "Has anyone with family connections to executives invested in companies "
    "operating in similar markets to ours?"
)

print(f"Answer: {result['answer']}")
print(f"\nRetrieval Statistics:")
print(f"- Documents retrieved: {result['vector_documents']}")
print(f"- Entities linked to graph: {result['entities_linked']}")
print(f"- Graph nodes explored: {result['graph_nodes_explored']}")
