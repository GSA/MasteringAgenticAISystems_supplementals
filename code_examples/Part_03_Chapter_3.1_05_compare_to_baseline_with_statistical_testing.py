from scipy import stats
from typing import Dict, Any

def compare_to_baseline(
    new_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Statistical comparison to baseline with pass/fail decision"""

    results = {
        'passed': True,
        'comparisons': {},
        'alerts': []
    }

    # Accuracy comparison
    accuracy_diff = new_metrics['accuracy'] - baseline_metrics['accuracy']
    accuracy_pct_change = (accuracy_diff / baseline_metrics['accuracy']) * 100

    if accuracy_pct_change < -thresholds['accuracy_regression_pct']:
        results['passed'] = False
        results['alerts'].append(
            f"⚠️ Accuracy regression: {accuracy_pct_change:.1f}% "
            f"(threshold: {-thresholds['accuracy_regression_pct']}%)"
        )

    results['comparisons']['accuracy'] = {
        'baseline': baseline_metrics['accuracy'],
        'new': new_metrics['accuracy'],
        'change_pct': accuracy_pct_change
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
