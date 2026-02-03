production_config = tensorrt_llm.Builder().create_builder_config(
    name='gpt2_production',
    precision='int8',
    kv_cache_dtype='int8',  # Quantize KV cache to INT8
    enable_paged_kv_cache=True,  # Use paged memory allocation
    max_batch_size=32,  # Support dynamic batching
    max_beam_width=4
)

production_engine = tensorrt_llm.build(model, production_config,
                                       output_dir='gpt2_production_engine')
