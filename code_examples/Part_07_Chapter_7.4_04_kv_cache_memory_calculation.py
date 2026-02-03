# Calculate maximum blocks
gpu_memory_gb = 40  # A100 40GB
model_weights_gb = 14  # Llama 2 7B FP16
activation_memory_gb = 2
available_for_kv = (gpu_memory_gb - model_weights_gb - activation_memory_gb) * 1024  # MB

tokens_per_block = 64
kv_cache_per_token_mb = 0.00048  # Depends on hidden_size, num_layers
max_blocks = int(available_for_kv / (tokens_per_block * kv_cache_per_token_mb))

print(f"Maximum KV cache blocks: {max_blocks}")
# Expected: ~40,000 blocks = 2.5M tokens total capacity
