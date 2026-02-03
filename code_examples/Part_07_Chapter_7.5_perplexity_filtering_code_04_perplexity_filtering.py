# Remove low-quality/corrupted documents using perplexity
dataset = dataset.filter(
    PerplexityFilter(
        max_perplexity=1500,
        model="gpt2",  # Reference model for scoring
        stride=512     # Compute perplexity on 512-token chunks for efficiency
    )
)

print(f"After perplexity filter: {len(dataset)} documents")
# Expected: ~2.3 billion documents (eliminated 15% noisy/corrupted content)