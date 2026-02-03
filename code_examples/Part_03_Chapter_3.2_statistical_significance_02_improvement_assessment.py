from scipy import stats
import numpy as np

def assess_improvement_significance(baseline_results, improved_results,
                                   min_practical_improvement=0.03):
    """
    Assess whether observed improvement is both statistically and practically significant.

    baseline_results: List of binary outcomes (1=success, 0=failure) for baseline agent
    improved_results: List of binary outcomes for improved agent on same examples
    min_practical_improvement: Minimum improvement threshold for practical significance

    Returns: (decision, stats_dict)
    """
    n = len(baseline_results)
    baseline_accuracy = np.mean(baseline_results)
    improved_accuracy = np.mean(improved_results)
    improvement = improved_accuracy - baseline_accuracy

    # Perform McNemar's test for paired binary data
    # Counts cases where agents disagree
    b10 = sum((b == 1 and i == 0) for b, i in zip(baseline_results, improved_results))
    b01 = sum((b == 0 and i == 1) for b, i in zip(baseline_results, improved_results))

    if b10 + b01 < 25:
        # Use exact binomial test for small disagreement counts
        p_value = stats.binom_test(b01, n=b10+b01, p=0.5, alternative='greater')
    else:
        # Use McNemar's test for larger samples
        chi2 = (abs(b01 - b10) - 1)**2 / (b01 + b10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Calculate confidence interval using Wilson score interval
    from statsmodels.stats.proportion import proportion_confint
    ci_low, ci_high = proportion_confint(
        sum(improved_results), n, method='wilson', alpha=0.05
    )

    stats_dict = {
        'baseline_accuracy': baseline_accuracy,
        'improved_accuracy': improved_accuracy,
        'improvement': improvement,
        'p_value': p_value,
        'ci_95': (ci_low, ci_high),
        'sample_size': n,
        'disagreements': {'improved_won': b01, 'baseline_won': b10}
    }

    # Decision logic: require both statistical and practical significance
    statistically_significant = p_value < 0.05
    practically_significant = improvement >= min_practical_improvement

    if statistically_significant and practically_significant:
        decision = "DEPLOY"
        rationale = f"Improvement of {improvement:.1%} is both statistically significant (p={p_value:.4f}) and exceeds practical threshold ({min_practical_improvement:.1%})"
    elif statistically_significant and not practically_significant:
        decision = "INSUFFICIENT_GAIN"
        rationale = f"While statistically significant (p={p_value:.4f}), improvement of {improvement:.1%} does not meet practical threshold ({min_practical_improvement:.1%})"
    elif not statistically_significant and practically_significant:
        decision = "INSUFFICIENT_EVIDENCE"
        rationale = f"Improvement of {improvement:.1%} meets practical threshold but lacks statistical significance (p={p_value:.4f}). Collect more samples."
    else:
        decision = "KEEP_BASELINE"
        rationale = f"Improvement of {improvement:.1%} is neither statistically significant (p={p_value:.4f}) nor practically significant"

    stats_dict['decision'] = decision
    stats_dict['rationale'] = rationale

    return decision, stats_dict

# Example: Evaluating whether an improved reasoning prompt helps
baseline_results = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 100  # 70% accuracy
improved_results = [1, 0, 1, 1, 1, 1, 1, 1, 0, 1] * 100  # 80% accuracy

decision, stats = assess_improvement_significance(
    baseline_results, improved_results, min_practical_improvement=0.05
)

print(f"Decision: {decision}")
print(f"Rationale: {stats['rationale']}")
print(f"Baseline: {stats['baseline_accuracy']:.1%}, Improved: {stats['improved_accuracy']:.1%}")
print(f"95% CI for improved agent: [{stats['ci_95'][0]:.1%}, {stats['ci_95'][1]:.1%}]")
