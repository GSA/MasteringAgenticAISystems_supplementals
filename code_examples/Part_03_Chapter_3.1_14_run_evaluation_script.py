import argparse
import json
import sys
from pathlib import Path
from evaluation_pipeline import CustomEvaluationPipeline
from agent import load_agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Test dataset path')
    parser.add_argument('--output', required=True, help='Results output path')
    parser.add_argument('--baseline', required=True, help='Baseline metrics path')
    args = parser.parse_args()

    try:
        # Load agent from current code
        agent = load_agent()

        # Run evaluation
        pipeline = CustomEvaluationPipeline(args.dataset)
        metrics = pipeline.evaluate_agent_with_custom_metrics(agent)

        # Load baseline for comparison
        with open(args.baseline, 'r') as f:
            baseline_metrics = json.load(f)

        # Compare to baseline
        comparison = compare_metrics(metrics, baseline_metrics)

        # Save results
        output = {
            'metrics': metrics,
            'baseline': baseline_metrics,
            'comparison': comparison,
            'passed': comparison['passed']
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        # Exit with appropriate code
        sys.exit(0 if comparison['passed'] else 1)

    except Exception as e:
        # Save error information
        error_output = {
            'error': str(e),
            'passed': False
        }
        with open(args.output, 'w') as f:
            json.dump(error_output, f, indent=2)

        # Exit with failure code
        sys.exit(2)  # Different code for pipeline errors vs metric failures

if __name__ == '__main__':
    main()
