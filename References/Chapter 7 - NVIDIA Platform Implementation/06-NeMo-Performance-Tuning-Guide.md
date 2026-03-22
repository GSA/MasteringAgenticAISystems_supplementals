# NeMo Framework Performance Tuning Guide

**Source:** 
'/Users/tamnguyen/Documents/GitHub/book1/references/Chapter 1 - Agent Architecture and Design/Nemo_Agent_Toolkit'
https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html

**Framework:** NVIDIA NeMo
**Focus:** Training performance optimization and parallelism strategies
**Audience:** Researchers, engineers training large language models

## Overview

The NeMo Performance Tuning Guide provides systematic approaches to optimizing model training performance. The core principle: **start with data parallelism, then add specialized parallelism strategies only when necessary.**

## Parallelism Strategy Selection

### Recommended Decision Framework

**Step 1: Can model + activations fit on single GPU?**
- **YES** → Use Data Parallelism (DP) only
- **NO** → Add Tensor Parallelism (TP)

**Step 2: Can model fit on TP cluster (sharded across TP GPUs)?**
- **YES** → Use DP + TP
- **NO** → Add Pipeline Parallelism (PP) or Virtual PP (VPP)

**Step 3: Are sequences very long?**
- **YES** → Add Context Parallelism (CP)
- **NO** → Not necessary

## Data Parallelism (DP)

### When to Use

- Model and activation memory fit within GPU memory
- Generally offers optimal performance
- Minimizes communication overhead
- Maximizes per-GPU tensor sizes

### Performance Characteristics

**Communication Overhead:**
- All-Reduce of gradients (after backward pass)
- Bandwidth-efficient operation
- Overlaps with computation when possible

**Memory Optimization:**
- Distributed Optimizer: Shards master parameters and optimizer states across DP ranks
- **No additional communication cost**
- Reduces per-GPU memory footprint
- Allows larger batch sizes

**Throughput:**
- Linear scaling up to ~8-16 GPUs
- Scales to 100+ GPUs with careful tuning
- Each GPU processes different data samples

### Configuration

```yaml
trainer:
  devices: 8
  strategy: ddp  # Distributed Data Parallel

model:
  data_parallel_size: 8
  distributed_backend: nccl
```

## Tensor Parallelism (TP)

### When to Use

- Model weights exceed single GPU memory
- Hidden dimension is large enough (>1024 recommended)
- Intra-node deployment (NVLink available)

### Critical Guidance

**Keep TP confined to NVLink domains** (within single node) to avoid slow inter-node communication

**Recommendation:** TP size ≤ 8 (single node limit on typical servers)

### Memory Impact

**Per-GPU Memory Reduction:**
- TP=4 reduces per-GPU memory by ~4x
- Enables larger hidden dimensions
- Allows longer sequences
- Increases effective training throughput

**Example:**
- Model: 70B parameters
- FP16: ~140 GB weights
- TP=8: ~17.5 GB per GPU (fits on H100)

### Configuration

```yaml
model:
  tensor_model_parallel_size: 8  # 8-way tensor parallelism
  sequence_parallel: false        # Consider enabling for long sequences
  distributed_backend: nccl
```

### Communication Pattern

**All-Reduce after Attention:**
- High bandwidth requirement (several GB/s)
- Benefits from fast intra-node interconnect
- NVLink: 450 GB/s per direction (ideal)
- 200Gbps Ethernet: ~20 GB/s (too slow)

**Recommendation:** Use only for intra-node parallelism

## Pipeline Parallelism (PP) / Virtual PP (VPP)

### Standard Pipeline Parallelism

**Concept:** Sequentially partition model layers

```
Device 1: [Layer 1-10]
Device 2: [Layer 11-20]
Device 3: [Layer 21-30]
```

**Challenge:** Pipeline bubbles
- Device 1 finishes forward pass, must wait for Device 2
- Compute stalls, GPUs idle

### Virtual Pipeline Parallelism (VPP)

**Solution:** Subdivide each layer into chunks

```
Device 1: [Chunk 1-A, Chunk 1-B, ..., Chunk 2-A, Chunk 2-B]
Device 2: [Chunk 1-C, Chunk 1-D, ..., Chunk 2-C, Chunk 2-D]
Device 3: [Chunk 1-E, Chunk 1-F, ..., Chunk 2-E, Chunk 2-F]
```

**Benefits:**
- Reduced pipeline bubbles
- Better device utilization
- Higher throughput

**Trade-off:** Increased inter-stage communication overhead

### Configuration

```yaml
model:
  pipeline_model_parallel_size: 4
  virtual_pipeline_model_parallel_size: 2
  distributed_backend: nccl
```

## Context Parallelism (CP)

### When to Use

- Training with very long sequences (>8K tokens)
- Sequence length > hidden dimension
- Memory pressure from KV cache

### Mechanism

Partitions operations along sequence dimension rather than batch/hidden dimensions

**Operations Suitable for CP:**
- LayerNorm (sequence-wise)
- Dropout (per-token)
- Attention (can partition queries)

**Operations Unsuitable:**
- Embedding (batch operation)
- Linear layers (compute-heavy)

### Performance Characteristics

- Moderate communication overhead
- Significant memory reduction for long sequences
- Best combined with TP for ultra-long sequences

### Configuration

```yaml
model:
  sequence_parallel: true         # Enable sequence parallelism
  tensor_model_parallel_size: 4
  sequence_parallel_degree: 4     # Number of devices
```

## Fully Sharded Data Parallelism (FSDP)

### Two Implementations

**PyTorch-Native FSDP:**
- Pure PyTorch implementation
- Easier debugging
- Good for research/development

**Megatron FSDP (Custom):**
- Optimized for performance
- Minimizes data movement
- Reuses communication buffers
- Recommended for production

### Key Optimizations

**Distributed Optimizer:**
- Shards master parameters across FSDP ranks
- Reduces per-GPU memory
- No communication overhead

**Communication Overlap:**
- Overlaps AllGather with forward pass
- Overlaps ReduceScatter with backward pass
- Hides communication latency

### Configuration

```yaml
trainer:
  strategy: fsdp_native  # or fsdp

model:
  fsdp_sharding_strategy: "full"  # Full sharding
  fsdp_activation_offloading: true
```

## Activation Offloading

### Purpose

Transfer activation memory to host (CPU) memory during training

### Use Cases

- Fine-tuning with LoRA
- Training with very large batch sizes
- Extreme sequence lengths
- FSDP with limited GPU memory

### Trade-off

**Benefit:** Reduced GPU memory pressure
**Cost:** Slower computation (recompute on backward pass)

### Configuration

```yaml
model:
  activation_checkpointing: true
  activation_offloading: true  # Offload to host memory
```

### Performance Expectation

- 20-30% slower training
- 30-50% GPU memory reduction
- Enables otherwise impossible training scenarios

## Communication Overlaps

### Overlapping AllGather with Forward

```
GPU Progress: [AllGather overlaps with forward computation]
         |========|========|
         └─ AllGather communication
         └─ Forward computation
         Result: Faster overall progress
```

### Overlapping ReduceScatter with Backward

Similar pattern for backward pass

### Implementation

NeMo automatically enables overlaps when:
- TP > 1
- Sequence parallelism enabled
- Compatible with configured parallelism

## Profiling & Analysis Tools

### NVIDIA Nsight Systems

**Visualization:** GPU kernel timeline

**Benefits:**
- Identify stalls and bottlenecks
- Visualize communication patterns
- Analyze kernel efficiency

**Usage:**
```bash
nsys profile -o trace.nsys-rep \
  python train.py ...

# View in NVIDIA Nsight Systems GUI
```

### NeMo Memory Profile Plugin

**Analysis:** GPU memory allocation patterns

**Captures:**
- Peak memory usage
- Memory fragmentation
- Activation memory trends

**Configuration:**
```yaml
plugins:
  - name: memory_profiler
    interval: 100  # Log every 100 steps
```

## Practical Tuning Workflow

### Phase 1: Baseline (Single GPU)

1. Train on single GPU with minimal settings
2. Record baseline throughput (tokens/sec)
3. Measure GPU memory usage
4. Identify performance bottleneck

### Phase 2: Add Parallelism

1. If memory insufficient → Add TP
2. If still insufficient → Add PP/VPP
3. If sequence length critical → Add CP
4. Measure throughput at each step

### Phase 3: Optimize Communication

1. Enable communication overlaps
2. Profile with Nsight Systems
3. Adjust chunk sizes if needed
4. Measure final throughput

### Phase 4: Fine-tune Hyperparameters

1. Adjust TP/PP/CP balance
2. Optimize batch size per GPU
3. Test gradient accumulation
4. Profile and iterate

## Recommended Configurations

### 70B Model on H100 Cluster (8 GPUs)

```yaml
# 70B parameters on 8 H100s with NVLink
model:
  hidden_size: 4096
  num_layers: 80
  tensor_model_parallel_size: 8  # TP across 8 GPUs
  pipeline_model_parallel_size: 1
  sequence_parallel: false

trainer:
  devices: 8
  strategy: ddp
  batch_size_per_gpu: 1
  gradient_accumulation_steps: 16
```

### 175B Model on Multi-Node (64 GPUs)

```yaml
# 175B parameters on 64 GPUs across 8 nodes
model:
  hidden_size: 12288
  num_layers: 96
  tensor_model_parallel_size: 8   # TP within each node
  pipeline_model_parallel_size: 8  # PP across nodes
  sequence_parallel: true
  virtual_pipeline_model_parallel_size: 2

trainer:
  devices: 64
  strategy: fsdp_native
  batch_size_per_gpu: 1
  gradient_accumulation_steps: 8
```

## Performance Monitoring

### Metrics to Track

**Training Throughput:**
- Tokens processed per second
- Samples per second
- Should improve with each optimization

**GPU Utilization:**
- Target: >80% for optimal efficiency
- Lower suggests communication bottleneck

**Memory Efficiency:**
- Peak GPU memory used
- Should decrease with proper optimization

**Scaling Efficiency:**
- Throughput improvement / GPU count
- Target: >90% scaling efficiency

## Best Practices

### Design Phase

1. Profile single-GPU training first
2. Identify memory or compute bottleneck
3. Choose appropriate parallelism strategy
4. Plan for scaling

### Implementation

1. Start with conservative settings
2. Gradually increase complexity
3. Measure improvement at each step
4. Profile with actual tools (not guesses)

### Production Tuning

1. Establish baseline metrics
2. Optimize one variable at a time
3. Use reproducible randomization
4. Document configuration and results

## Conclusion

NeMo's performance tuning approach emphasizes:
- **Systematic selection** of parallelism strategies
- **Measurement-driven decisions** using profiling tools
- **Careful communication management** for multi-GPU training
- **Iterative optimization** starting simple, adding complexity as needed

By following this structured approach—understanding when and why to use each parallelism strategy, profiling to identify bottlenecks, and optimizing communication patterns—organizations achieve significant throughput improvements when training large language models at scale.
