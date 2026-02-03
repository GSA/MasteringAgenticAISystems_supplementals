# Centralized error handling with fallback strategies
def _handle_agent_failure(self, error: AgentError, ticket_context: TicketContext):
    """Implements fallback strategies for agent failures."""
    if isinstance(error, RetrievalError):
        # Retrieval failure: use cached recent documents
        fallback_docs = self.cache.get_recent_docs(
            category=ticket_context.preliminary_category
        )
        # Continue workflow with cached documents
        return self._continue_with_fallback_docs(fallback_docs, ticket_context)

    elif isinstance(error, ClassificationError):
        # Classification failure: use rule-based classification
        fallback_category = self.rule_classifier.classify(
            ticket_context.message
        )
        # Continue with rule-based classification
        return self._continue_with_fallback_classification(
            fallback_category, ticket_context
        )

    elif isinstance(error, ResponseError):
        # Response generation failure: use template response
        template_response = self.template_library.get_template(
            classification.category
        )
        # Send template with escalation flag
        self._send_template_and_escalate(template_response, ticket_context)

    else:
        # Unknown error: escalate immediately
        self.event_bus.publish(
            topic="ticket.error.critical",
            event=ErrorEvent(
                ticket_id=ticket_context.ticket_id,
                error=error,
                requires_immediate_attention=True
            )
        )