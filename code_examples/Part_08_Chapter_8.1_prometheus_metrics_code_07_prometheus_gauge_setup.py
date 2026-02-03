# Prometheus metrics for GPU utilization
gpu_utilization = prom.Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = prom.Gauge('gpu_memory_used_mb', 'GPU memory used', ['gpu_id'])
gpu_power_usage = prom.Gauge('gpu_power_usage_watts', 'GPU power consumption', ['gpu_id'])
