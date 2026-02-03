    @kernel_function(
        description="Analyze customer sentiment and concerns from support conversation history. "
                    "Returns sentiment classification, identified concerns, urgency assessment, "
                    "and recommended actions. Use when you need to understand customer emotional "
                    "state or prioritize support tickets.",
        name="analyze_support_conversation_sentiment"
    )
    async def analyze_support_conversation_sentiment(
        self,
        conversation_history: str,
        customer_profile: str
    ) -> dict:
        """
        Semantic function: Analyze sentiment using LLM reasoning.

        Args:
            conversation_history: Full conversation transcript
            customer_profile: JSON string of customer profile for context

        Returns:
            {
                "sentiment": str,  # "positive", "neutral", "negative"
                "concerns": list[str],  # Identified customer concerns
                "urgency": str,  # "low", "medium", "high"
                "recommended_actions": list[str]  # Support team next steps
            }
        """
        prompt = f"""
Analyze this customer support conversation for sentiment and key concerns.

Conversation:
{conversation_history}

Customer Context:
{customer_profile}

Provide analysis as JSON with these fields:
- sentiment: "positive", "neutral", or "negative"
- concerns: list of specific concerns mentioned (be concrete)
- urgency: "low", "medium", or "high" based on tone and issue severity
- recommended_actions: list of specific next steps for support team

Focus on customer emotional state, specific problems mentioned, and appropriate support escalation.
"""

        result = await self.kernel.invoke_prompt(prompt)
        return json.loads(str(result))

    @kernel_function(
        description="Generate personalized response to customer inquiry using their profile, "
                    "order history, and conversation context. Creates empathetic, specific "
                    "responses addressing customer concerns with relevant account details. "
                    "Use when drafting customer communications.",
        name="generate_personalized_customer_response"
    )
    async def generate_personalized_customer_response(
        self,
        customer_inquiry: str,
        customer_profile: str,
        additional_context: str = ""
    ) -> str:
        """
        Semantic function: Generate natural language response.

        Args:
            customer_inquiry: What the customer asked or communicated
            customer_profile: JSON string with customer details
            additional_context: Any relevant order/account context

        Returns:
            Personalized response text ready to send to customer
        """
        prompt = f"""
You are a professional customer service representative. Generate a helpful, empathetic response
to this customer inquiry using specific details from their account.

Customer Inquiry:
{customer_inquiry}

Customer Profile:
{customer_profile}

Additional Context:
{additional_context}

Response Guidelines:
- Address customer by name
- Reference specific account details when relevant
- Acknowledge concerns empathetically
- Provide actionable next steps
- Maintain professional but friendly tone
- Keep response concise (2-3 paragraphs maximum)

Generate the response:
"""

        result = await self.kernel.invoke_prompt(prompt)
        return str(result)
