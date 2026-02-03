from vllm import LLM
import numpy as np

# Generate summaries and analyze token probability entropy
llm_target = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="float16",
    max_model_len=8192
)

# Sample contracts and generate summaries with probability logging
test_contracts = load_contracts_sample(100)
outputs = llm_target.generate(
    test_contracts,
    max_new_tokens=300,
    temperature=0.0,
    logprobs=1  # Return top token probabilities
)

# Analyze predictability: what fraction of tokens have >70% probability?
high_confidence_tokens = []
for output in outputs:
    logprobs = output.outputs[0].logprobs
    for token_logprobs in logprobs:
        top_prob = np.exp(token_logprobs[list(token_logprobs.keys())[0]])
        high_confidence_tokens.append(top_prob > 0.7)

predictability = np.mean(high_confidence_tokens)
print(f"Predictable Token Fraction: {predictability:.1%}")

# Results:
# Predictable Token Fraction: 68.4%
