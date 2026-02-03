import tensorrt_llm
from tensorrt_llm.models import GPT

# Configure optimization settings
builder_config = tensorrt_llm.Builder().create_builder_config(
    name='gpt2',
    precision='float16',  # FP16 optimization
    tensor_parallel=1,    # Single GPU
    max_batch_size=8,
    max_input_len=512,
    max_output_len=512
)

# Build optimized engine
engine = tensorrt_llm.build(
    model,
    builder_config,
    output_dir='gpt2_fp16_engine'
)
