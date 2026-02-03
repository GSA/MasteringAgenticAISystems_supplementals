import time
from typing import Optional, Any, List, Dict

def evaluate_agent(
    agent: Any,
    test_cases: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run agent against all test cases and collect results"""
    results = []

    for idx, case in enumerate(test_cases):
        print(f"Processing query {idx+1}/{len(test_cases)}: {case['query'][:50]}...")

        start_time = time.time()
        try:
            response = agent.run(case['query'])
            latency_ms = (time.time() - start_time) * 1000

            results.append({
                'query': case['query'],
                'expected': case['expected_answer'],
                'actual': response,
                'intent': case['intent'],
                'difficulty': case['difficulty'],
                'latency_ms': latency_ms,
                'error': None,
                'success': True
            })
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            results.append({
                'query': case['query'],
                'expected': case['expected_answer'],
                'actual': None,
                'intent': case['intent'],
                'difficulty': case['difficulty'],
                'latency_ms': latency_ms,
                'error': str(e),
                'success': False
            })

    return {'test_results': results}
