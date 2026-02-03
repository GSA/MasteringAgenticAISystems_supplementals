import time
import numpy as np
from typing import List, Dict, Any

def evaluate_agent_with_custom_metrics(
    self,
    agent: Any
) -> Dict[str, Any]:
    """Run evaluation with standard + custom metrics"""

    results = []
    empathy_scores = []
    compliance_scores = []
    efficiency_scores = []

    for test_case in self.test_dataset:
        # Run agent and time response
        start_time = time.time()

        try:
            response = agent.run(test_case['query'])
            latency_ms = (time.time() - start_time) * 1000

            # Standard metrics
            accuracy = self._check_accuracy(
                response,
                test_case['expected_answer']
            )

            # Custom metrics (computed immediately with context)
            empathy = calculate_empathy_score(
                response,
                test_case['query']
            )
            compliance = calculate_policy_compliance(
                response,
                self.policies
            )

            # Tool efficiency (if agent provides tool call log)
            if hasattr(agent, 'get_tool_calls'):
                tool_calls = agent.get_tool_calls()
                efficiency = calculate_tool_efficiency(tool_calls)
            else:
                efficiency = None  # Not all agents expose tool calls

            results.append({
                'query': test_case['query'],
                'response': response,
                'accuracy': accuracy,
                'latency_ms': latency_ms,
                'empathy': empathy,
                'compliance': compliance,
                'efficiency': efficiency,
                'success': True
            })

            # Collect for aggregation
            empathy_scores.append(empathy)
            compliance_scores.append(compliance)
            if efficiency is not None:
                efficiency_scores.append(efficiency)

        except Exception as e:
            # Record failure but continue evaluation
            results.append({
                'query': test_case['query'],
                'error': str(e),
                'success': False
            })

    # Aggregate metrics
    metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results if r['success']]),
        'latency_p95': np.percentile(
            [r['latency_ms'] for r in results if r['success']],
            95
        ),
        'empathy_mean': np.mean(empathy_scores),
        'empathy_p50': np.percentile(empathy_scores, 50),
        'compliance_rate': np.mean(compliance_scores),  # 1.0 = 100% compliant
        'efficiency_mean': np.mean(efficiency_scores) if efficiency_scores else None,
        'test_count': len(results)
    }

    return metrics
