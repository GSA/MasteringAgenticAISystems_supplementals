# Output Constraint Implementation
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=500,  # Limit output to 500 tokens
    # Quality testing confirmed minimal degradation at this threshold
)
