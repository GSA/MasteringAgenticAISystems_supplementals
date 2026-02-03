import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model in FP32 precision
model = GPT2LMHeadModel.from_pretrained("gpt2-large")  # 1.5B params
model.eval().cuda()

# Measure baseline latency and memory
input_ids = tokenizer.encode("Explain quantum computing:", return_tensors="pt").cuda()

import time
start = time.time()
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100, do_sample=False)
baseline_latency = time.time() - start

print(f"Baseline FP32 latency: {baseline_latency:.2f}s for 100 tokens")
print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
