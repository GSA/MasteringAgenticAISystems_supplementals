import json
from typing import Dict, Any

def compare_metrics(current: Dict[str, Any], baseline: Dict[str, Any], thresholds_path: str = 'config/evaluation_thresholds.json') -> Dict[str, Any]:
    """Compare current metrics to baseline using configured thresholds"""

    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)

    results = {
        'passed': True,
        'comparisons': {},
        'alerts': []
    }

    # Check accuracy
    accuracy_diff = current['accuracy'] - baseline['accuracy']
    accuracy_pct_change = (accuracy_diff / baseline['accuracy']) * 100

    accuracy_thresholds = thresholds['accuracy']
    accuracy_passed = True

    if current['accuracy'] < accuracy_thresholds['min_absolute']:
        accuracy_passed = False
        results['alerts'].append(
            f"Accuracy {current['accuracy']:.1%} below minimum {accuracy_thresholds['min_absolute']:.1%}"
        )

    if accuracy_pct_change < accuracy_thresholds['max_regression_pct']:
        accuracy_passed = False
        results['alerts'].append(
            f"Accuracy regressed {accuracy_pct_change:.1f}% (threshold: {accuracy_thresholds['max_regression_pct']}%)"
        )

    results['comparisons']['accuracy'] = {
        'current': current['accuracy'],
        'baseline': baseline['accuracy'],
        'change_pct': accuracy_pct_change,
        'passed': accuracy_passed
    }

    if not accuracy_passed:
        results['passed'] = False

    # Repeat for other metrics...

    return results
