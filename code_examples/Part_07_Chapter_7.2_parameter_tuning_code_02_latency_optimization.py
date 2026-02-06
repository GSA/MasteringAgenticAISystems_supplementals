# Optimize parameters for minimum latency
response = client.chat.completions.create(
    model="meta-llama-2-7b",
    messages=[{"role": "user", "content": "Quick response?"}],
    temperature=0.3,    # Lower temperature = faster sampling
    max_tokens=50,      # Shorter responses = lower latency
    top_p=0.8,          # Reduce search space for faster decoding
    presence_penalty=0  # Simplify logit processing
)
