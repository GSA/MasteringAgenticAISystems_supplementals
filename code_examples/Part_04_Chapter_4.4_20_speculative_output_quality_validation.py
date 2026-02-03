# Generate summaries with both configurations
outputs_baseline = llm_target.generate(
    test_contracts[:50],
    max_new_tokens=300,
    temperature=0.0
)

outputs_spec = llm_speculative.generate(
    test_contracts[:50],
    max_new_tokens=300,
    temperature=0.0
)

# Compare token-level exact match
exact_matches = sum([
    out_base.outputs[0].text == out_spec.outputs[0].text
    for out_base, out_spec in zip(outputs_baseline, outputs_spec)
])

print(f"Exact Output Matches: {exact_matches}/50 ({100*exact_matches/50:.1f}%)")

# Evaluate summary quality with automated metrics
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_baseline = []
rouge_spec = []

for contract, out_base, out_spec in zip(test_contracts[:50], outputs_baseline, outputs_spec):
    reference_summary = contract.gold_summary  # Ground truth from dataset

    scores_base = scorer.score(reference_summary, out_base.outputs[0].text)
    scores_spec = scorer.score(reference_summary, out_spec.outputs[0].text)

    rouge_baseline.append(scores_base['rougeL'].fmeasure)
    rouge_spec.append(scores_spec['rougeL'].fmeasure)

print(f"Baseline ROUGE-L: {np.mean(rouge_baseline):.3f}")
print(f"Speculative ROUGE-L: {np.mean(rouge_spec):.3f}")
print(f"Quality Difference: {np.mean(rouge_spec) - np.mean(rouge_baseline):.3f}")

# Results:
# Exact Output Matches: 48/50 (96.0%)
# Baseline ROUGE-L: 0.642
# Speculative ROUGE-L: 0.641
# Quality Difference: -0.001 (negligible)
