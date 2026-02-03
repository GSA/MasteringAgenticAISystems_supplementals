# Prepare calibration dataset (CRITICAL for accuracy)
calibration_dataset = load_domain_specific_data(
    source='customer_support_conversations',  # Match production distribution
    num_samples=512,
    max_length=512
)

# Configure INT8 quantization with entropy calibration
int8_config = tensorrt_llm.Builder().create_builder_config(
    name='gpt2_int8',
    precision='int8',
    calibration_method='entropy',  # Minimize KL divergence
    calibration_data=calibration_dataset,
    quantization_mode='per_channel',  # Balance accuracy and speed
    max_batch_size=8
)

# Build INT8 engine (includes calibration phase)
int8_engine = tensorrt_llm.build(
    model,
    int8_config,
    output_dir='gpt2_int8_engine'
)
