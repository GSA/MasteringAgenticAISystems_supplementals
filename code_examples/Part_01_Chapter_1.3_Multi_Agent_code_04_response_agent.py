# Response Agent generates customer reply
class ResponseAgent:
    def generate_response(
        self,
        ticket_context: TicketContext,
        retrieved_docs: RetrievedDocs,
        classification: ClassificationResult
    ) -> Response:
        """Generates customer-facing response using retrieved context."""
        # Build response prompt with full context
        prompt = self._build_response_prompt(
            customer_message=ticket_context.message,
            retrieved_context=retrieved_docs.documents,
            category=classification.category,
            conversation_history=ticket_context.conversation_history
        )

        # Generate response with appropriate tone
        response_text = self.llm.generate(
            prompt,
            temperature=0.7,  # Higher temperature for natural responses
            max_tokens=500,
            stop_sequences=["Customer:", "Agent:"]
        )

        # Validate response quality
        if not self._meets_quality_standards(response_text):
            # Regenerate with stricter constraints
            response_text = self._regenerate_with_constraints(prompt)

        return Response(
            text=response_text,
            confidence=self._assess_confidence(response_text),
            citations=retrieved_docs.documents[:3]  # Include source references
        )