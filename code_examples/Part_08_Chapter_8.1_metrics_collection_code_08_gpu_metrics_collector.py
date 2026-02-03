# Collect GPU metrics every 10 seconds
def collect_gpu_metrics():
    metrics = dcgm_agent.get_latest_values()
    for gpu_id, values in metrics.items():
        gpu_utilization.labels(gpu_id=gpu_id).set(values['utilization'])
        gpu_memory_used.labels(gpu_id=gpu_id).set(values['memory_used'])
        gpu_power_usage.labels(gpu_id=gpu_id).set(values['power_usage'])
