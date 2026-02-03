# Escalation Agent determines human review necessity
class EscalationAgent:
    def evaluate_escalation(
        self,
        classification: ClassificationResult,
        response: Response,
        ticket_context: TicketContext
    ) -> EscalationDecision:
        """Determines if ticket requires human review."""
        # Check multiple escalation criteria
        needs_escalation = (
            classification.escalation_required or  # Low classification confidence
            response.confidence < 0.7 or           # Low response confidence
            self._contains_sensitive_content(ticket_context.message) or
            self._exceeds_authority_threshold(classification.category)
        )

        if needs_escalation:
            # Publish escalation event for downstream consumers
            self.event_bus.publish(
                topic="ticket.escalation.required",
                event=EscalationEvent(
                    ticket_id=ticket_context.ticket_id,
                    reason=self._escalation_reason(classification, response),
                    priority=self._calculate_priority(ticket_context),
                    assigned_team="customer_support_l2"
                )
            )

            return EscalationDecision(
                escalate=True,
                reason=self._escalation_reason(classification, response),
                hold_response=True  # Don't send automated response
            )
        else:
            # Ticket fully handled by automation
            return EscalationDecision(
                escalate=False,
                send_response=True
            )