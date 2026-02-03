import mlflow
from pathlib import Path
from typing import Dict, Any

def log_evaluation_with_custom_metrics(
    agent_config: Dict[str, Any],
    metrics: Dict[str, float]
):
    """Log comprehensive evaluation to MLflow"""

    with mlflow.start_run(run_name=f"eval_{agent_config['model']}"):
        # Log agent configuration
        mlflow.log_params({
            'model': agent_config['model'],
            'temperature': agent_config['temperature'],
            'prompt_template': agent_config['prompt_template']
        })

        # Log standard metrics
        mlflow.log_metric('accuracy', metrics['accuracy'])
        mlflow.log_metric('latency_p95_ms', metrics['latency_p95'])

        # Log custom metrics with consistent naming
        mlflow.log_metric('custom/empathy_mean', metrics['empathy_mean'])
        mlflow.log_metric('custom/empathy_p50', metrics['empathy_p50'])
        mlflow.log_metric('custom/compliance_rate', metrics['compliance_rate'])

        if metrics['efficiency_mean'] is not None:
            mlflow.log_metric('custom/efficiency_mean', metrics['efficiency_mean'])

        # Tag run with custom metric categories for filtering
        mlflow.set_tag('has_custom_metrics', 'true')
        mlflow.set_tag('custom_metric_types', 'empathy,compliance,efficiency')

        print(f"Logged run to MLflow: {mlflow.active_run().info.run_id}")
