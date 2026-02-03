# Distributed tracing for workflow visibility
import opentelemetry.trace as trace

class TicketOrchestrator:
    def process_ticket(self, customer_message: str, metadata: dict) -> TicketOutcome:
        """Orchestrates ticket processing with full tracing."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("ticket_processing") as span:
            # Add correlation ID for cross-agent tracing
            correlation_id = str(uuid.uuid4())
            span.set_attribute("correlation_id", correlation_id)
            span.set_attribute("account_id", metadata['account_id'])

            # Each agent call becomes a traced span
            with tracer.start_as_current_span("intake") as intake_span:
                ticket_context = self.intake_agent.process_ticket(
                    customer_message, metadata, correlation_id=correlation_id
                )
                intake_span.set_attribute("ticket_id", ticket_context.ticket_id)

            with tracer.start_as_current_span("retrieval") as retrieval_span:
                retrieved_docs = self.retrieval_agent.retrieve_context(
                    ticket_context, correlation_id=correlation_id
                )
                retrieval_span.set_attribute("docs_found", len(retrieved_docs.documents))
                retrieval_span.set_attribute("avg_relevance",
                    sum(retrieved_docs.relevance_scores) / len(retrieved_docs.relevance_scores)
                )

            # ... similar tracing for classification, response, escalation

            # Record workflow metrics
            self.metrics.record_workflow_latency(
                span.duration_ms,
                tags={"category": classification.category, "escalated": escalation_decision.escalate}
            )
            self.metrics.increment_counter(
                "tickets_processed",
                tags={"status": final_outcome.status}
            )

            return final_outcome