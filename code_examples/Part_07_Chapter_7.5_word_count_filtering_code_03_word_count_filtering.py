# Remove documents that are too short to be meaningful
dataset = dataset.filter(
    WordCountFilter(
        min_words=50,
        max_words=10000  # Also remove extremely long documents (likely corrupted)
    )
)

print(f"After word count filter: {len(dataset)} documents")
# Expected: ~2.7 billion documents (eliminated 15% fragments and corrupted docs)