# Initialize knowledge graph builder
builder = KnowledgeGraphBuilder(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Process business news corpus
documents = [
    "Elon Musk founded SpaceX in 2002 to revolutionize space technology.",
    "SpaceX developed the Falcon 9 reusable rocket, reducing launch costs.",
    "Tesla Motors was founded by Martin Eberhard and Marc Tarpenning in 2003.",
    "Elon Musk joined Tesla as chairman in 2004 and became CEO in 2008.",
    "Google invested $300 million in OpenAI in 2016 to advance AI research.",
    "Microsoft invested $10 billion in OpenAI in 2023, their largest AI investment."
]

builder.process_documents(documents)
builder.close()

print("Knowledge graph construction complete!")
