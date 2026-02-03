# Measure acceptance rate on production query distribution
from spec_decode import measure_acceptance_rate

draft_candidates = ["tinyllama-1b", "llama-3b-distilled", "self-spec-head"]
for draft in draft_candidates:
    acceptance_rate = measure_acceptance_rate(
        draft_model=draft,
        target_model="llama-70b",
        queries=production_query_sample  # 1000+ real queries
    )
    print(f"{draft}: acceptance_rate={acceptance_rate:.3f}")

# Select draft with highest acceptance rate, not highest accuracy
