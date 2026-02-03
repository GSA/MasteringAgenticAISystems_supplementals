import mlflow
from pathlib import Path
import json
from typing import Dict, Any

def log_to_mlflow(
    agent_config: Dict[str, Any],
    metrics: Dict[str, float],
    results: Dict[str, Any]
):
    """Log evaluation run to MLflow for tracking and comparison"""

    with mlflow.start_run(run_name=f"eval_{agent_config['model']}"):
        # Log agent configuration as parameters
        mlflow.log_params({
            'model': agent_config['model'],
            'temperature': agent_config['temperature'],
            'prompt_template': agent_config['prompt_template'],
            'max_tokens': agent_config['max_tokens']
        })

        # Log computed metrics
        mlflow.log_metrics(metrics)

        # Log detailed results as artifact
        results_path = Path('results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(results_path)

        print(f"Logged run to MLflow: {mlflow.active_run().info.run_id}")
