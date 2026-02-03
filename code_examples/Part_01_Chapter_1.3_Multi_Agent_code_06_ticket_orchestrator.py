# TicketOrchestrator coordinates agent chain
class TicketOrchestrator:
    def process_ticket(self, customer_message: str, metadata: dict) -> TicketOutcome:
        """Orchestrates multi-agent ticket processing workflow."""
        try:
            # Step 1: Intake and enrichment
            ticket_context = self.intake_agent.process_ticket(
                customer_message, metadata
            )

            # Step 2: Knowledge retrieval
            retrieved_docs = self.retrieval_agent.retrieve_context(ticket_context)

            # Step 3: Classification
            classification = self.classification_agent.classify_intent(
                ticket_context, retrieved_docs
            )

            # Step 4: Response generation
            response = self.response_agent.generate_response(
                ticket_context, retrieved_docs, classification
            )

            # Step 5: Escalation evaluation
            escalation_decision = self.escalation_agent.evaluate_escalation(
                classification, response, ticket_context
            )

            # Determine final outcome
            if escalation_decision.escalate:
                # Hold automated response, await human handling
                return TicketOutcome(
                    status="escalated",
                    response=None,
                    assigned_to="human"
                )
            else:
                # Send automated response
                self._send_response(ticket_context.ticket_id, response)
                return TicketOutcome(
                    status="resolved",
                    response=response.text,
                    assigned_to="automation"
                )

        except AgentError as e:
            # Agent failure triggers fallback
            self._handle_agent_failure(e, ticket_context)
            return TicketOutcome(
                status="error",
                error=str(e),
                assigned_to="error_queue"
            )