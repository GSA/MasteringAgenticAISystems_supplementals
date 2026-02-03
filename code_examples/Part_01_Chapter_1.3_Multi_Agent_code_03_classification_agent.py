# Classification Agent determines ticket category
class ClassificationAgent:
    def classify_intent(
        self,
        ticket_context: TicketContext,
        retrieved_docs: RetrievedDocs
    ) -> ClassificationResult:
        """Classifies ticket intent for routing."""
        # Build classification prompt with retrieved context
        prompt = self._build_prompt(ticket_context, retrieved_docs)

        # Invoke classification LLM
        classification = self.llm.generate(
            prompt,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=50
        )

        # Parse structured classification output
        return ClassificationResult(
            category=classification.category,  # "refund", "technical", "billing"
            confidence=classification.confidence,
            routing_decision=self._determine_routing(classification),
            escalation_required=classification.confidence < 0.75
        )