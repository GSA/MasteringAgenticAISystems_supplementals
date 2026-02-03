# Remove exact duplicates (identical text content)
exact_dedup = ExactDuplicates(
    id_field="id",               # Document identifier for tracking
    text_field="text",           # Field containing document text
    hash_method="md5"            # Fast hash for exact matching
)

dataset = exact_dedup(dataset)

print(f"After exact deduplication: {len(dataset)} documents")
# Expected: ~1.8 billion documents (eliminated 22% exact duplicates)