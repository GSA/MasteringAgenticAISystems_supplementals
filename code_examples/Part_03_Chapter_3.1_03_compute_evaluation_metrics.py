import numpy as np
from difflib import SequenceMatcher
from typing import Dict, Any

def compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate accuracy, latency, and cost metrics"""
    test_results = results['test_results']

    # Accuracy: fuzzy matching to handle minor wording differences
    correct = 0
    for result in test_results:
        if not result['success']:
            continue

        similarity = SequenceMatcher(
            None,
            result['expected'].lower(),
            result['actual'].lower()
        ).ratio()

        # Count as correct if >80% similar (handles paraphrasing)
        if similarity > 0.8:
            correct += 1

    accuracy = correct / len(test_results)

    # Latency percentiles
    latencies = [r['latency_ms'] for r in test_results]
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)

    # Error rate
    errors = sum(1 for r in test_results if not r['success'])
    error_rate = errors / len(test_results)

    # Cost calculation (example: $0.002 per query for GPT-4)
    cost_per_query = 0.002  # This would come from actual API usage
    total_cost = cost_per_query * len(test_results)

    return {
        'accuracy': accuracy,
        'p50_latency_ms': p50_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'error_rate': error_rate,
        'total_cost': total_cost,
        'test_count': len(test_results)
    }
