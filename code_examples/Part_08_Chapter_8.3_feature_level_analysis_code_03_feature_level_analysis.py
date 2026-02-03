# Feature-Level Token Analysis
async def analyze_token_usage(tracker: TokenTracker, days: int = 7):
    """Analyze token usage patterns to identify optimization opportunities"""
    usage_data = await tracker.get_usage(days=days)

    # Group usage by feature type
    feature_analysis = {}
    for feature in ["portfolio_recommendation", "market_analysis", "risk_assessment"]:
        feature_data = [u for u in usage_data if u.feature == feature]

        total_requests = len(feature_data)
        total_cost = sum(u.cost_usd for u in feature_data)
        avg_input_tokens = sum(u.input_tokens for u in feature_data) / total_requests
        avg_output_tokens = sum(u.output_tokens for u in feature_data) / total_requests

        # Identify repetitive prompt prefixes (caching candidates)
        prompts = [get_prompt(u.request_id) for u in feature_data]
        common_prefix_length = find_common_prefix_length(prompts)

        feature_analysis[feature] = {
            "total_requests": total_requests,
            "total_cost_usd": total_cost,
            "avg_cost_per_request": total_cost / total_requests,
            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "cacheable_tokens": common_prefix_length,
            "cache_savings_potential": common_prefix_length * self.pricing["input"] * 0.5 * total_requests
        }

    return feature_analysis
