from tensorrt_llm import ModelConfig, SpeculativeConfig
from tensorrt_llm.runtime import Session

# Build draft model engine
draft_config = ModelConfig(
    model_dir="./llama-3b-draft",
    dtype="float16",
    max_batch_size=32
)
draft_engine = build_engine(draft_config)

# Build target model engine
target_config = ModelConfig(
    model_dir="./llama-70b-target",
    dtype="float16",
    max_batch_size=32
)
target_engine = build_engine(target_config)

# Configure speculative decoding
spec_config = SpeculativeConfig(
    draft_engine=draft_engine,
    target_engine=target_engine,
    num_draft_tokens=4,  # K=4
    enable_dynamic_k=True,  # Adjust K based on acceptance
    acceptance_threshold=0.5  # Minimum acceptance for speedup
)

# Create speculative session
session = Session(spec_config)

# Run inference with automatic speculation
output = session.generate(
    input_ids=input_tokens,
    max_length=256
)
