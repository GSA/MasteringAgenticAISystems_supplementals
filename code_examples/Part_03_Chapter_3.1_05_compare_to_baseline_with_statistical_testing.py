from scipy import stats
from typing import Dict, Any, Optional, Sequence

def compare_to_baseline(
    new_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    thresholds: Dict[str, float],
    new_outcomes: Optional[Sequence[int]] = None,
    baseline_outcomes: Optional[Sequence[int]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Statistical comparison to baseline with pass/fail decision.

    new_outcomes / baseline_outcomes: per-test-case results (1 = pass, 0 = fail)
    for the same test cases in the same order. When given, an accuracy drop past
    the threshold only blocks if an exact McNemar test finds it significant.
    """

    results = {
        'passed': True,
        'comparisons': {},
        'alerts': []
    }

    # Accuracy comparison
    accuracy_diff = new_metrics['accuracy'] - baseline_metrics['accuracy']
    accuracy_pct_change = (accuracy_diff / baseline_metrics['accuracy']) * 100  # relative change

    # Exact McNemar test: only the cases where the two versions disagree carry information
    p_value = None
    if new_outcomes is not None and baseline_outcomes is not None:
        if len(new_outcomes) != len(baseline_outcomes):
            raise ValueError("new_outcomes and baseline_outcomes must cover the same test cases")
        fixed = sum(1 for n, b in zip(new_outcomes, baseline_outcomes) if n and not b)
        broken = sum(1 for n, b in zip(new_outcomes, baseline_outcomes) if b and not n)
        p_value = stats.binomtest(broken, fixed + broken, 0.5).pvalue if fixed + broken else 1.0

    if accuracy_pct_change < -thresholds['accuracy_regression_pct'] and p_value is not None and p_value >= alpha:
        results['alerts'].append(
            f"Accuracy change {accuracy_pct_change:.1f}% is not significant (McNemar p={p_value:.3f}); not blocking"
        )
    elif accuracy_pct_change < -thresholds['accuracy_regression_pct']:
        results['passed'] = False
        results['alerts'].append(
            f"⚠️ Accuracy regression: {accuracy_pct_change:.1f}% "
            f"(threshold: {-thresholds['accuracy_regression_pct']}%)"
        )

    results['comparisons']['accuracy'] = {
        'baseline': baseline_metrics['accuracy'],
        'new': new_metrics['accuracy'],
        'change_pct': accuracy_pct_change,
        'p_value': p_value
    }

    # Latency comparison
    latency_diff = new_metrics['p95_latency_ms'] - baseline_metrics['p95_latency_ms']
    latency_pct_change = (latency_diff / baseline_metrics['p95_latency_ms']) * 100

    if latency_pct_change > thresholds['latency_increase_pct']:
        results['passed'] = False
        results['alerts'].append(
            f"⚠️ P95 latency increase: {latency_pct_change:.1f}% "
            f"(threshold: {thresholds['latency_increase_pct']}%)"
        )

    results['comparisons']['p95_latency'] = {
        'baseline': baseline_metrics['p95_latency_ms'],
        'new': new_metrics['p95_latency_ms'],
        'change_pct': latency_pct_change
    }

    # Error rate comparison
    if new_metrics['error_rate'] > thresholds['max_error_rate']:
        results['passed'] = False
        results['alerts'].append(
            f"⚠️ Error rate: {new_metrics['error_rate']*100:.1f}% "
            f"(threshold: {thresholds['max_error_rate']*100}%)"
        )

    return results
