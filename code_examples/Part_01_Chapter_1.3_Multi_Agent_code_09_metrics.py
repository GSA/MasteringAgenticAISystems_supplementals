# Key metrics for multi-agent system health
class MultiAgentMetrics:
    def __init__(self):
        self.metrics = {
            # Workflow-level metrics
            "ticket_processing_latency_p50": Histogram(),
            "ticket_processing_latency_p95": Histogram(),
            "ticket_processing_latency_p99": Histogram(),
            "tickets_processed_per_minute": Counter(),
            "escalation_rate": Gauge(),

            # Per-agent metrics
            "intake_latency": Histogram(),
            "retrieval_latency": Histogram(),
            "classification_latency": Histogram(),
            "response_generation_latency": Histogram(),
            "escalation_evaluation_latency": Histogram(),

            # Error metrics
            "agent_error_rate": Counter(),  # Tagged by agent type
            "fallback_activation_count": Counter(),  # How often fallbacks trigger
            "critical_error_count": Counter(),

            # Resource metrics
            "llm_tokens_consumed": Counter(),  # Tagged by agent and model
            "vector_db_queries": Counter(),
            "cache_hit_rate": Gauge(),

            # Quality metrics
            "classification_confidence_avg": Gauge(),
            "response_confidence_avg": Gauge(),
            "customer_satisfaction_score": Gauge()  # From post-ticket surveys
        }