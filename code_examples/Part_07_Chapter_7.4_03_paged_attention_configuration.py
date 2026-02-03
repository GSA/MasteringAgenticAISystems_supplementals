# TensorRT-LLM PagedAttention configuration
from tensorrt_llm import BuildConfig

build_config = BuildConfig()
build_config.plugin_config.set_paged_kv_cache(
    tokens_per_block=64  # Block size in tokens
)
build_config.plugin_config.set_context_fmha(True)  # Enable fused attention
build_config.max_num_tokens = 8192  # Total token budget

# Build engine with paged KV cache
engine = build(
    model,
    build_config,
    max_batch_size=32,
    max_input_len=2048,
    max_output_len=512
)
