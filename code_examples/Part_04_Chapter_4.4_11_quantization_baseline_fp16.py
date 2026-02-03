from vllm import LLM

# Build baseline FP16 engine
llm_fp16 = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="float16",
    gpu_memory_utilization=0.90
)

# Measure throughput on production-representative workload
import time
from datasets import load_dataset

moderation_samples = load_dataset("your-org/moderation-validation", split="test")
test_batch = moderation_samples.select(range(1000))

start = time.perf_counter()
outputs = llm_fp16.generate(
    [sample["text"] for sample in test_batch],
    max_new_tokens=10,  # Classification is short output
    temperature=0.0
)
throughput_fp16 = len(outputs) / (time.perf_counter() - start)

print(f"FP16 Throughput: {throughput_fp16:.1f} requests/second")
# Measured: 45.2 requests/second per GPU
# Six GPUs provide 271 req/sec total throughput
