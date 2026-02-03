from semantic_kernel.prompt_template import PromptTemplate

# Semantic function: Analyze customer sentiment from support conversation
analyze_sentiment_prompt = PromptTemplate(
    template="""
Analyze the sentiment and key concerns from this customer support conversation:

Conversation:
{{$conversation_history}}

Customer Profile:
{{$customer_profile}}

Provide:
1. Overall sentiment (Positive/Neutral/Negative)
2. Specific concerns or frustrations mentioned
3. Urgency level (Low/Medium/High)
4. Recommended next actions for support team

Format as JSON.
""",
    template_format="semantic-kernel"
)

@kernel_function(
    description="Analyze customer sentiment and concerns from support conversation history",
    name="analyze_customer_sentiment"
)
async def analyze_customer_sentiment(
    conversation_history: str,
    customer_profile: dict
) -> dict:
    """Semantic function using LLM for sentiment analysis"""
    result = await kernel.invoke_prompt_function(
        analyze_sentiment_prompt,
        conversation_history=conversation_history,
        customer_profile=str(customer_profile)
    )
    return result
