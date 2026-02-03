from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate_recommendation_with_caching(client_id: str, query: str):
    """Generate portfolio recommendation with optimized prompt caching"""

    # Structure prompt to maximize cache hits
    # CRITICAL: Place static/repetitive content first for prefix matching
    prompt = f"""[SYSTEM INSTRUCTIONS - CACHEABLE]
You are a financial advisor providing portfolio recommendations based on client profiles and current market conditions. Follow SEC and FINRA guidelines for suitability. Provide specific, actionable recommendations with risk disclosures.

[CLIENT PROFILE - CACHEABLE FOR SESSION]
Client ID: {client_id}
{await get_client_profile(client_id)}  # 5000 tokens, stable during conversation

[MARKET CONTEXT - UPDATES DAILY]
{await get_market_data()}  # 3000 tokens, refreshed daily

[USER QUERY - UNIQUE PER REQUEST]
{query}  # ~200 tokens, varies per request
"""

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        # OpenAI automatically caches matching prefixes
        # Cache TTL typically 5-60 minutes depending on traffic
    )

    # Track usage including cache hit information
    usage = response.usage
    tracker.track_request(
        request_id=response.id,
        feature="portfolio_recommendation",
        user_id=client_id,
        model="gpt-4o",
        prompt=prompt,
        completion=response.choices[0].message.content,
        cached_tokens=getattr(usage, 'prompt_tokens_cached', 0),
        latency_ms=response.response_ms
    )

    return response.choices[0].message.content
