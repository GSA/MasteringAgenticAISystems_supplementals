# TensorRT Best Practices for Performance Optimization

**Source:** https://docs.nvidia.com/deeplearning/tensorrt/10.8.0/performance/best-practices.html

**Version:** TensorRT 10.8.0+
**Focus:** Performance measurement, optimization techniques, and deployment strategies
**Audience:** Engineers optimizing inference performance

## Overview

NVIDIA TensorRT best practices guide provides comprehensive guidance for achieving optimal inference performance. The core philosophy emphasizes practical measurement-driven optimization: "The most important optimization is to compute as many results in parallel as possible using batching."

## Performance Measurement Fundamentals

### Key Metrics

**Latency** - Time from input presentation to output availability
- **Units:** Milliseconds (ms)
- **Importance:** Critical for safety-critical and user-facing applications
- **Optimization Goal:** Lower is better
- **Use Case:** Real-time systems, interactive applications

**Throughput** - Number of inferences completed in a fixed time period
- **Units:** Inferences per second (IPS)
- **Importance:** Critical for batch processing and data centers
- **Optimization Goal:** Higher is better
- **Use Case:** Batch jobs, high-volume services

**Trade-off:** Increasing throughput typically increases latency; optimization strategy depends on application requirements

### Measurement Tools

**trtexec** (Recommended Primary Tool)
- Built-in command-line benchmarking tool
- Per-layer timing analysis
- End-to-end metrics
- Usage: `trtexec --engine=model.trt --warmUp=100 --duration=10`

**Nsight Systems**
- GPU-level kernel analysis
- Timeline visualization
- Context switching analysis
- Detailed system-level metrics

**Nsight Deep Learning Designer**
- ONNX-specific profiling
- Layer-to-kernel correlation
- Visual performance analysis

## Critical Optimization Strategies

### 1. Batching (Highest Impact)

**Principle:** Larger batches improve GPU efficiency by amortizing layer overhead across more instances

**Batch Size Guidelines:**
- **Multiples of 32**: Optimize Tensor Core utilization in FP16/INT8 inference
- **Powers of 2**: Often perform better due to GPU scheduling
- **Model-specific**: Test actual batch sizes relevant to deployment

**Example Performance Impact:**
- Batch 1: Baseline latency
- Batch 32: 8-12x throughput improvement
- Batch 64: 12-18x throughput improvement
- Batch 128: Diminishing returns, increased latency

**Implementation:**
```bash
trtexec --onnx=model.onnx --batch=32 --fp16
```

### 2. Precision Selection

TensorRT supports multiple precision levels with trade-offs between speed and accuracy:

| Precision | Speed | Memory | Accuracy | Use Case |
|---|---|---|---|---|
| FP32 | Baseline (1x) | Baseline | Full precision | Baseline/validation |
| FP16 | 2-4x faster | 50% | Minimal loss | Default optimization |
| TF32 | 4-6x faster | Baseline | Minimal loss | CUDA Compute 8.0+ |
| INT8 | 4-8x faster | 75% | 0.5-1% loss | Calibration required |
| FP8 | 8-16x faster | 75% | 1-2% loss | Modern GPUs (Ada+) |

**Selection Strategy:**
1. Start with FP16 (good balance)
2. Test INT8 with calibration
3. Evaluate accuracy impact
4. Use FP8 for extreme optimization if available

### 3. GPU Configuration

**Clock Management**
- Lock GPU clocks at consistent frequencies for deterministic measurements
- Trade-off: Consistency vs. average performance
- Verification: `nvidia-smi -lgc <frequency>`

**Power Throttling Monitoring**
- Monitor with: `nvidia-smi dmon`
- Detect power limitations affecting clock stability
- Ensure adequate power supply (PCIe power connectors)

**Thermal Considerations**
- Maintain proper cooling
- Monitor temperature: `nvidia-smi -l 1` (continuous monitoring)
- Prevent thermal throttling (avoid >85°C)
- Degrade both performance and stability above thermal limits

## Profiling and Diagnostics

### Profiling Workflow

**Step 1: Baseline Measurement**
```bash
trtexec --engine=model.trt --batch=32 --warmUp=100 --duration=10
```

**Step 2: Per-Layer Analysis**
```bash
trtexec --engine=model.trt --batch=32 --dumpProfile
# Examine layer-wise latencies
```

**Step 3: GPU-Level Analysis**
```bash
nsys profile -o profile.nsys-rep \
  python inference.py --batch=32
```

### Key Profiling Insight

Run benchmarks with `--noDataTransfers` to isolate GPU computation performance from host-device data movement overhead. This reveals whether optimization targets are compute-bound or memory-bound.

### Bottleneck Identification

**Compute-Bound Operations:**
- GPU cores fully utilized
- Optimization: Better algorithms, Tensor Cores, reduced precision
- Tools: FLOPS/peak analysis

**Memory-Bound Operations:**
- Memory bandwidth limitation
- Optimization: Layer fusion, reduced precision, batching
- Tools: Bandwidth analysis, cache efficiency

## Advanced Performance Techniques

### CUDA Graphs

**Concept:** Capture kernel sequences for reduced CPU overhead

**Benefits:**
- Reduces CPU overhead in enqueue operations
- Particularly beneficial for small-batch or "enqueue-bound" workloads
- Kernel launch overhead amortization

**When to Use:**
- Small batches (<4)
- High inference rates (>1000 IPS)
- CPU bottleneck detected

### Multi-Streaming

**Within-Inference Streams**
- Use auxiliary streams via `setMaxAuxStreams()`
- Enables parallel execution of independent layers
- Reduces pipeline stalls

**Cross-Inference Streams**
- Run multiple contexts simultaneously
- Improves device utilization
- Better hardware utilization at scale

**Configuration:**
```cpp
IRuntime* runtime = createInferRuntime(logger);
IExecutionContext* context1 = engine->createExecutionContext();
IExecutionContext* context2 = engine->createExecutionContext();
// Execute contexts on different CUDA streams
```

### Layer Fusion

**Automatic Fusion:**
- TensorRT automatically fuses compatible patterns
- Examples: convolution+ReLU, activation sequences
- Reduces kernel launches and memory traffic

**Impact:**
- 10-30% throughput improvement typical
- Reduced memory bandwidth pressure
- Better latency characteristics

## Quantization Workflow (INT8)

### Complete INT8 Process

**Step 1: Export Model**
```bash
# Export from framework (PyTorch, TensorFlow, etc.)
torch.onnx.export(model, dummy_input, "model.onnx")
```

**Step 2: Quantize with ModelOptimizer**
```bash
python3 -m modelopt.onnx.quantization \
  --onnx_path model.onnx \
  --quantize_mode int8
```

**Step 3: Build TensorRT Engine**
```bash
trtexec --onnx=model_quantized.onnx \
  --int8 \
  --stronglyTyped \
  --saveEngine=model_int8.trt
```

**Step 4: Profile Results**
```bash
trtexec --engine=model_int8.trt \
  --dumpProfile \
  --batch=32
```

### Quantization Performance

**Typical Results:**
- Throughput improvement: 60% vs. FP16 baseline
- Accuracy loss: <1% on most models
- Memory reduction: 75%

**Calibration:** Requires representative data samples for INT8 calibration

## Engine Building Optimization

### Resource-Aware Building

Limit available compute resources during engine building via CUDA MPS when the GPU will face contention at runtime.

**Benefit:** Produces engines optimized for realistic deployment conditions rather than peak theoretical performance

**Configuration:**
```bash
# Set MPS thread percentage limit
nvidia-smi -i 0 -X EXCLUSIVE_PROCESS

# Build with limited resources
trtexec --onnx=model.onnx \
  --cudaGraphs \
  --useSpinWait=False
```

## Deployment Optimization

### Model Caching

Cache serialized engines to avoid rebuild overhead:
```cpp
std::ifstream cache("model.trt", std::ios::binary);
std::vector<char> trtModelStream(
    (std::istreambuf_iterator<char>(cache)),
    std::istreambuf_iterator<char>()
);
IRuntime* runtime = createInferRuntime(logger);
ICudaEngine* engine = runtime->deserializeCudaEngine(
    trtModelStream.data(), trtModelStream.size(), nullptr
);
```

### Memory Optimization

**Workspace Memory:**
- Configured via builder settings
- Typical: 1-4 GB
- Trade-off: More memory = potentially faster execution

**Weight Memory:**
- Reduced via quantization
- Shared across inference contexts

## Best Practices Checklist

### Pre-Deployment
- [ ] Establish baseline performance (latency, throughput)
- [ ] Test quantization and precision options
- [ ] Profile with realistic batch sizes
- [ ] Verify GPU cooling and power delivery
- [ ] Test on target hardware

### Optimization
- [ ] Implement batching for throughput optimization
- [ ] Evaluate precision levels (start with FP16)
- [ ] Test INT8 quantization
- [ ] Profile per-layer timing
- [ ] Identify compute vs. memory bottlenecks

### Deployment
- [ ] Use engine caching for fast startup
- [ ] Monitor GPU utilization and temperature
- [ ] Implement CUDA graphs for low-latency paths
- [ ] Use multi-streaming for better resource utilization
- [ ] Plan for inference scaling

## Conclusion

TensorRT best practices emphasize measurement-driven optimization with focus on practical deployment. By systematically addressing batching, precision selection, GPU configuration, and profiling, organizations can achieve 3-10x performance improvements, enabling efficient inference at scale with optimal latency and throughput characteristics.

The key insight: **Measure first, optimize second**—without baseline metrics, optimization decisions are guesses; with proper profiling, significant performance gains become systematic and reproducible.
