# Pseudo-code for monitoring auto-scaling effectiveness
monitoring_queries = {
    "replica_count": "count(kube_pod_info{deployment='customer-service-agent'})",
    "cpu_utilization": "avg(rate(container_cpu_usage_seconds_total{pod=~'customer-service-agent.*'}[5m]))",
    "memory_utilization": "avg(container_memory_working_set_bytes{pod=~'customer-service-agent.*'} / container_spec_memory_limit_bytes{pod=~'customer-service-agent.*'})",
    "request_rate": "rate(agent_requests_total[1m])",
    "p95_latency": "histogram_quantile(0.95, rate(agent_request_duration_seconds_bucket[5m]))",
    "scaling_events": "count_over_time(kube_horizontalpodautoscaler_status_desired_replicas{horizontalpodautoscaler='customer-service-agent-hpa'}[1h])"
}
