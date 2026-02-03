# Example query
results = hybrid_search(client, "What is the memory bandwidth of H100?", limit=5, alpha=0.7)

for i, doc in enumerate(results):
    print(f"\n[{i+1}] Score: {doc['_additional']['score']:.3f}")
    print(f"Source: {doc['source_doc']} (chunk {doc['chunk_index']})")
    print(f"Content: {doc['content'][:200]}...")
