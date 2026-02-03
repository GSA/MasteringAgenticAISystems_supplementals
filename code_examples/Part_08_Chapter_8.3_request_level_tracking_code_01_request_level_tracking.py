# Pseudocode for request-level tracking
def track_llm_call(prompt, response, model, user_id, feature):
    metrics.emit({
        'timestamp': now(),
        'user_id': user_id,
        'feature': feature,
        'model': model,
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
        'cost': calculate_cost(response.usage, model),
        'latency_ms': response.latency
    })
