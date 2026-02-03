# Remove near-duplicates (documents with substantial overlapping content)
fuzzy_dedup = FuzzyDuplicates(
    id_field="id",
    text_field="text",
    seed=42,
    num_hashes=128,              # More hashes = higher precision, slower
    similarity_threshold=0.85    # Keep documents <85% similar
)

dataset = fuzzy_dedup(dataset)

print(f"After fuzzy deduplication: {len(dataset)} documents")
# Expected: ~1.5 billion documents (eliminated additional 17% near-duplicates)