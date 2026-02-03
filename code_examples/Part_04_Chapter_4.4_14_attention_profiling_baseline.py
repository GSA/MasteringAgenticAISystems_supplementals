import torch
from vllm import LLM
import time

# Build baseline engine with standard attention
llm_baseline = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    dtype="float16",
    max_model_len=16384,
    gpu_memory_utilization=0.90,
    disable_custom_all_reduce=True
)

# Profile single-request latency breakdown
test_contract = load_contract_sample(12000)  # 12K token input

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    output = llm_baseline.generate(
        test_contract,
        max_new_tokens=800,
        temperature=0.0
    )

# Analyze kernel time breakdown
stats = prof.key_averages()
attention_time = sum([s.cuda_time_total for s in stats if 'attention' in s.key])
total_time = sum([s.cuda_time_total for s in stats])

print(f"Total Inference Time: {total_time/1e6:.1f}ms")
print(f"Attention Kernels: {attention_time/1e6:.1f}ms ({100*attention_time/total_time:.1f}%)")

# Results:
# Total Inference Time: 3420ms
# Attention Kernels: 2640ms (77.2%)
