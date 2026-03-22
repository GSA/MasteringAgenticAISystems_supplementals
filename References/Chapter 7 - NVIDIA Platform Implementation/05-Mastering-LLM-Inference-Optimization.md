# Mastering LLM Inference Optimization Techniques

**Source:** https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

**Publisher:** NVIDIA Technical Blog
**Focus:** Comprehensive LLM inference optimization strategies
**Audience:** ML engineers and inference optimization specialists

## Overview

LLM inference presents unique optimization challenges distinct from training. While training maximizes GPU utilization through batching, inference must handle dynamic workloads with varying token generation requirements. This guide explores optimization techniques addressing the fundamental challenge: **LLM inference is memory-bandwidth bound despite computational capacity.**

## Core Inference Phases

### Prefill Phase

**Definition:** Process input tokens to compute intermediate states

**Characteristics:**
- Highly parallelized matrix-matrix operations
- Saturates GPU compute capacity
- Benefits from batching and parallelization
- Compute-bound operations

**Duration:** Typically 10-20% of total inference time

**Optimization Focus:**
- GPU utilization (target: >80%)
- Batch size selection
- Model parallelization strategies

### Decode Phase

**Definition:** Generate output tokens autoregressively one at a time

**Characteristics:**
- Memory-bound matrix-vector calculations
- Underutilizes GPU compute relative to prefill
- Cannot parallelize across tokens (inherent sequentiality)
- Bandwidth-limited operations

**Duration:** Typically 80-90% of total inference time

**Optimization Focus:**
- Memory bandwidth efficiency
- KV cache management
- Batching strategy

## Memory Management & KV Caching

### Memory Consumption Profile

**Model Weights:**
- Example: Llama 2 7B in FP16
- 14B parameters × 2 bytes = ~28GB
- Shared across all requests
- Loading overhead: 1-2 seconds

**KV Cache:**
- Key-value tensors for each layer
- Example: Llama 2 7B with batch 1, seq length 2048
- KV cache: ~2GB
- Scales: Linear with sequence length and batch size
- **Critical bottleneck** in decode phase

**Activation Memory:**
- Temporary intermediate states during computation
- Freed after inference step
- Varies by batch size and model architecture

### Key-Value Cache Challenge

**Problem:** Static over-provisioning for maximum sequence lengths causes fragmentation

**Example:**
```
Batch of 4 sequences:
- Seq 1: Generated 5 tokens (of max 2048)
- Seq 2: Generated 100 tokens (of max 2048)
- Seq 3: Generated 50 tokens (of max 2048)
- Seq 4: Generated 10 tokens (of max 2048)

Memory waste: > 95% of KV cache unused
```

### PagedAttention Solution

**Concept:** Apply memory paging (like OS virtual memory) to KV cache

**Mechanism:**
- Divide KV cache into fixed-size blocks
- Store non-contiguously
- Allocate blocks on-demand
- Release completed blocks immediately

**Benefits:**
- Reduces memory waste from ~95% to ~5%
- Enables larger batch sizes
- Improves throughput 2-4x on dynamic workloads
- No mathematical accuracy loss

## Optimization Strategies

### Batching Approaches

**Static Batching:**
- Fixed batch size
- All sequences generate same token count
- Inefficient for variable workloads
- Simple implementation

**In-Flight Batching (Continuous Batching):**
- Immediately evict completed sequences
- Fill slots with new requests
- Process in lock-step iteration
- Dramatically improves GPU utilization

**Benefit:** 5-10x throughput improvement on real workloads

### Attention Optimization

**Standard Multi-Head Attention (MHA):**
- Each query head has separate K, V
- High memory requirements
- Baseline approach
- Required for some architectures

**Multi-Query Attention (MQA):**
- All queries share single K, V pair
- Reduces memory to 1/H (H = num heads)
- Requires model retraining
- Used in Falcon, newer models

**Grouped-Query Attention (GQA):**
- Group queries (G heads per group)
- Share K, V within group
- Balances MHA and MQA
- Can adapt existing models
- Used in Llama 2 70B, Code Llama

**FlashAttention:**
- I/O-aware algorithm using tiling
- Minimizes GPU memory transfers
- No mathematical change
- 2-4x memory bandwidth reduction
- Implemented in vLLM

## Model Parallelization

### Tensor Parallelism (TP)

**Concept:** Shard transformer layers horizontally

**Implementation:**
```
Layer: [Q W_q] [K W_k] [V W_v] [O W_o]
Split: [Q] [K] [V] [O]  (across 4 GPUs)
```

**Characteristics:**
- All-Reduce communication after attention
- High bandwidth requirement (needs NVLink)
- Suitable for large hidden dimensions
- Low communication overhead with proper sizing

**Recommendation:** Confine TP to NVLink domains (intra-node)

### Pipeline Parallelism (PP)

**Concept:** Partition model layers sequentially across devices

**Architecture:**
```
Device 1: Layers 1-10
Device 2: Layers 11-20
Device 3: Layers 21-30
```

**Challenge:** Pipeline bubbles (idle devices)

**Solution:** Virtual Pipeline Parallelism (VPP)
- Split each layer into chunks
- Interleave chunks across devices
- Reduces idle time
- Trade-off: Increased communication

### Sequence Parallelism (SP)

**Concept:** Partition operations along sequence dimension

**Use Cases:**
- LayerNorm (sequence-wise)
- Dropout (can be separated per token)
- Long sequences (> 4K tokens)

**Requirement:** Sequence length > hidden size for efficiency

**Benefit:** Memory reduction without high communication

## Model Optimization Techniques

### Quantization

**FP8 Quantization:**
- Reduce 16-bit to 8-bit
- 50% memory reduction
- 2-3x faster computation
- Minimal accuracy loss (<2%)

**INT4/INT3 Quantization:**
- Extreme compression
- 75% memory reduction
- More accuracy loss
- Requires careful calibration

**Challenge:** Activation outliers require special handling

### Sparsity Exploitation

**Structured Sparsity (2-out-of-4):**
- 50% parameters are zero
- Hardware acceleration available
- No accuracy loss (with proper training)
- Easy to implement

**Unstructured Sparsity:**
- Arbitrary zero patterns
- Requires specialized sparse kernels
- Limited hardware support
- More aggressive compression

### Distillation

**Process:**
1. Train teacher (large model)
2. Train student (small model) on teacher outputs
3. Fine-tune student on task

**Benefits:**
- 50-70% model size reduction
- Maintains 95-98% of teacher quality
- Faster inference
- Practical for production

## Advanced Serving Techniques

### Speculative Inference

**Concept:** Use cheap draft model to predict multiple tokens, verify with full model

**Process:**
```
1. Draft model generates 4 tokens (cheap)
2. Full model processes 4 tokens in parallel (verify)
3. Accept/reject each token
4. Continue from last accepted token
```

**Performance:**
- 2-4x latency reduction
- Maintain full model quality
- Trade compute for latency

**Use Cases:**
- Low-latency requirements
- Long sequence generation

## Implementation Frameworks

### TensorRT-LLM

**Features:**
- Bundles optimization techniques
- Compiled kernels for efficiency
- Multi-GPU communication primitives
- Multi-platform support (NVIDIA GPUs, CPUs)

**Techniques Included:**
- KV cache with PagedAttention
- In-flight batching
- Quantization (FP8, INT4)
- Speculative decoding
- Tensor/pipeline/sequence parallelism

### NVIDIA NIM

**Approach:** Containerized inference microservice

**Integration:**
- TensorRT-LLM (optimized models)
- vLLM (broader model support)
- SGLang (advanced scheduling)

**Benefits:**
- Single command deployment
- OpenAI API compatible
- Automatic optimization selection
- Production-ready support

## Optimization Decision Tree

```
START: Choose optimization strategy
│
├─ Is model memory < GPU memory?
│  ├─ YES → Use TP=1, focus on batching & in-flight batching
│  └─ NO → Use TP (confine to NVLink domain)
│
├─ Are sequences very long (>4K)?
│  ├─ YES → Add sequence parallelism (SP)
│  └─ NO → Not necessary
│
├─ Is accuracy critical?
│  ├─ YES → Start with FP16, test FP8
│  └─ NO → Aggressive quantization acceptable
│
├─ Is latency critical?
│  ├─ YES → Implement speculative decoding
│  └─ NO → Optimize throughput instead
│
└─ Deploy with selected optimizations
```

## Performance Benchmarking

### Key Metrics

**Throughput:**
- Tokens per second (T/s)
- Requests per second
- Combined: Tokens/second across all requests

**Latency:**
- Time to First Token (TTFT)
- Time per Token (TpT)
- End-to-end latency

**Efficiency:**
- Tokens/second per GPU
- Cost per 1M tokens
- Power consumption

### Measurement Methodology

1. **Baseline:** Unoptimized model
2. **Profile:** Identify bottlenecks (compute vs. memory)
3. **Select Techniques:** Based on bottleneck analysis
4. **Implement & Measure:** Track improvement
5. **Iterate:** Refine parameters

## Practical Recommendations

### Development Environment

- Start with single GPU, no parallelism
- Use smaller batch sizes for testing
- Enable profiling from day one
- Measure impact of each optimization

### Production Deployment

- Use TensorRT-LLM or NVIDIA NIM
- Implement in-flight batching
- Enable KV cache optimization
- Baseline each environment

### Model-Specific Tuning

- Review model architecture (attention type, size)
- Check for sparsity opportunities
- Assess quantization sensitivity
- Test distillation if size critical

## Conclusion

LLM inference optimization requires understanding the fundamental challenge: memory-bandwidth limitation despite computational capacity. No single technique provides complete solution; effective optimization combines:

1. **Memory Management:** PagedAttention, KV cache optimization
2. **Computation:** Batching, parallelization, attention optimization
3. **Model Compression:** Quantization, sparsity, distillation
4. **Serving Strategy:** In-flight batching, speculative decoding

By applying these techniques systematically—starting with profiling, selecting based on bottlenecks, and measuring impact—organizations achieve 5-20x inference performance improvements while reducing costs and latency.
