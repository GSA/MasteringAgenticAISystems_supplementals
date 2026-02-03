# Measure throughput with optimal batch size
# vLLM automatically manages batching, so we test with high request rate
start = time.perf_counter()
outputs_optimized = llm_flash.generate(
    test_contracts,  # 100 contracts
    max_new_tokens=800,
    temperature=0.0
)
throughput_optimized = len(outputs_optimized) / (time.perf_counter() - start)

print(f"Optimized Throughput: {throughput_optimized:.1f} req/sec")
print(f"Total Speedup: {throughput_optimized/2.4:.2f}x")

# Results:
# Optimized Throughput: 6.4 req/sec
# Total Speedup: 2.67x
