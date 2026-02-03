# Filter to English-only content
dataset = dataset.filter(
    LanguageIdentificationFilter(
        language="en",
        score_threshold=0.95  # High confidence threshold to avoid false positives
    )
)

print(f"After language filter: {len(dataset)} documents")
# Expected: ~3.2 billion documents (64% English in web crawls)