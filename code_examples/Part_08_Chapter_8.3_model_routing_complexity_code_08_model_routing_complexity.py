# Model Routing Based on Query Complexity
async def route_to_model(query: str, complexity: str):
    """Route query to appropriately-sized model"""

    # Simple heuristic-based complexity assessment
    simple_patterns = [
        "balance", "transaction", "history", "what is",
        "define", "explain", "current value", "holdings"
    ]

    complex_patterns = [
        "optimize", "tax-loss harvesting", "rebalance",
        "maximize", "minimize", "scenario", "compare strategies"
    ]

    query_lower = query.lower()

    # Classification logic
    if any(pattern in query_lower for pattern in complex_patterns):
        return "gpt-4o"
    elif any(pattern in query_lower for pattern in simple_patterns):
        return "gpt-4o-mini"
    else:
        # Default to moderate complexity → gpt-4o for safety
        return "gpt-4o"

async def generate_recommendation_with_routing(client_id: str, query: str):
    """Generate recommendation with complexity-based model routing"""

    # Determine appropriate model
    model = await route_to_model(query, assess_complexity(query))

    # Generate response with selected model
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content
