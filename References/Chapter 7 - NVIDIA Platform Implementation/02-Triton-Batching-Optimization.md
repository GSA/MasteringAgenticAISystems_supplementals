# Triton Batching Strategies and Optimization

**Source:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html

**Framework:** NVIDIA Triton Inference Server
**Focus:** Dynamic batching, sequence batching, and continuous batching for LLMs
**Audience:** Inference platform engineers and deployment specialists

## Overview

NVIDIA Triton provides multiple batching strategies to optimize inference performance by combining multiple requests. The choice of batching strategy depends on model characteristics and deployment requirements.

## Dynamic Batcher

### Purpose & Use Cases

The dynamic batcher combines inference requests to improve throughput by creating batches automatically. It's designed for **stateless models** and distributes batches across all configured model instances.

**Ideal For:**
- Computer vision models (classification, detection)
- NLP inference (sentence embeddings)
- Stateless transformers
- Independent request processing

### Configuration

**Basic Setup:**
```yaml
# model.pbtxt configuration
dynamic_batching { }
```

**With Parameters:**
```yaml
dynamic_batching {
  max_batch_size: 32
  max_queue_delay_microseconds: 10000
  preferred_batch_sizes: [8, 16, 32]
  priority_levels: 2
  default_priority_level: 0
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 30000
  }
}
```

### Tuning Strategy

**Step 1: Set Maximum Batch Size**
- Start conservatively (32 or 64)
- Increase based on GPU memory and latency requirements
- Consider model input/output sizes

**Step 2: Enable Basic Batching**
```bash
# Test with basic configuration
tritonserver --model-repository=models
```

**Step 3: Use Performance Analyzer**
```bash
perf_analyzer -m model_name \
  -c concurrency \
  --infer-interval=10000
```

**Step 4: Optimize Parameters**
- Adjust max_queue_delay for latency/throughput trade-off
- Set preferred batch sizes if certain sizes perform better
- Configure priority levels for mixed workloads

### Batch Size Selection

**Preferred Batch Sizes:**

A "preferred batch size" should only be configured if that batch size results in significantly higher performance than other batch sizes.

**Example:**
```yaml
# TensorRT models often have specific optimized batch sizes
preferred_batch_sizes: [1, 8, 32]  # Multiples of 8
```

**Performance Considerations:**
- Power-of-2 sizes often optimal for GPU scheduling
- Multiples of 32 optimize Tensor Core utilization (FP16/INT8)
- Test actual deployment batch sizes

### Delayed Batching

**Parameter:** `max_queue_delay_microseconds`

**Purpose:** Allow requests to wait in scheduler to form larger batches

**Trade-off:** Latency vs. Throughput
- Lower values: Faster response, lower batch efficiency
- Higher values: Slower response, higher batch efficiency

**Example:**
```yaml
# Balance: 10ms max wait
max_queue_delay_microseconds: 10000

# Aggressive batching: 100ms max wait
max_queue_delay_microseconds: 100000
```

**Formula:**
```
P(batch_size) = 1 - exp(-(arrival_rate * max_delay))
```

### Priority Queuing

Enable different handling for high-priority requests:

```yaml
dynamic_batching {
  max_batch_size: 32
  priority_levels: 2  # 0 (default), 1 (high priority)
  default_priority_level: 0
}
```

**Benefits:**
- High-priority requests bypass lower-priority ones
- SLA enforcement (urgent vs. batch)
- Mixed workload optimization

### Queue Management

**Queue Policies:**
```yaml
default_queue_policy {
  # When queue is full
  timeout_action: REJECT  # Reject new requests

  # Maximum wait time
  default_timeout_microseconds: 30000
}

# Per-priority policies
priority_queue_policy {
  timeout_action: DEFER
  default_timeout_microseconds: 1000
}
```

**Policy Options:**
- `REJECT` - Return error immediately
- `DEFER` - Wait and retry
- `ALLOW_DELAYED_SCHEDULE` - Queue indefinitely

## Sequence Batcher

### Purpose & Design

The sequence batcher is designed for **stateful models** where a sequence of inference requests must be routed to the same model instance while creating dynamic batches.

**Ideal For:**
- Language models with hidden state
- Recurrent neural networks (RNNs)
- Streaming speech recognition
- Conversational AI with context

### Configuration

```yaml
sequence_batching {
  max_sequence_slots: 4
  max_queue_delay_microseconds: 10000
  control_input_map {
    key: "START"
    value {
      data_type: TYPE_UINT32
      dims: [1]
    }
  }
  control_input_map {
    key: "END"
    value {
      data_type: TYPE_UINT32
      dims: [1]
    }
  }
}
```

### State Management

**Sequence Slots:** Fixed number of concurrent sequences the model can maintain

**Key Concepts:**
- **Sequence Start:** Signal beginning of new sequence
- **Sequence End:** Signal completion, free slot for new sequence
- **Correlation ID:** Tie requests to same sequence

### Correlation ID Handling

```python
# Client specifies correlation ID
request = grpcclient.InferInput("input", [1, 100], "FP32")
request.set_request_property("sequence_id", 1)
```

## Iterative Sequences (Continuous Batching for LLMs)

### Purpose

Enable stateful models to process single requests over multiple scheduling iterations, enabling **continuous batching** for LLM inference optimization.

**Key Feature:** Backends can complete requests at any iteration stage

### Use Cases

**Primary:** Large Language Model inference where:
- Token generation happens iteratively
- Batch slots freed as tokens complete
- Others move up to fill freed slots
- High efficiency without synchronization

### Architecture

```
Iteration 1:  Req1, Req2, Req3, Req4 → process
Iteration 2:  Req2, Req3, Req4, Req5 → process (Req1 done)
Iteration 3:  Req3, Req4, Req5, Req6 → process (Req2 done)
```

**Benefit:** Maintains high batch utilization throughout processing

## Ragged Batching

### Concept

Avoid explicit padding by allowing specification of inputs that don't require shape checking.

**Use Case:** Variable-length sequences (NLP)

### Configuration

```yaml
dynamic_batching {
  max_batch_size: 32
  # Specify inputs without padding requirement
}
```

**Example:**
```
Request 1: sequence length 10
Request 2: sequence length 25
Request 3: sequence length 15

Without ragging: pad all to 25 (memory waste)
With ragging: process as-is (efficient)
```

## Performance Optimization

### Batching Strategy Selection

| Strategy | Use Case | Max Throughput | Latency | Complexity |
|---|---|---|---|---|
| **Dynamic** | Stateless models | Highest | Medium | Low |
| **Sequence** | Stateful models | High | Medium | High |
| **Continuous** | LLM inference | Highest | Low-Medium | Medium |
| **None** | Real-time (<10ms) | Lower | Lowest | Very Low |

### Configuration Recipes

**High-Throughput Batch Processing:**
```yaml
dynamic_batching {
  max_batch_size: 128
  max_queue_delay_microseconds: 100000
  preferred_batch_sizes: [32, 64, 128]
}
```

**Low-Latency Interactive:**
```yaml
dynamic_batching {
  max_batch_size: 8
  max_queue_delay_microseconds: 1000
}
```

**Mixed Workload:**
```yaml
dynamic_batching {
  max_batch_size: 32
  priority_levels: 2
  default_queue_policy {
    timeout_action: REJECT
  }
}
```

## Performance Measurement

### Using perf_analyzer

```bash
# Measure throughput with dynamic batching
perf_analyzer -m model_name \
  -c 100 \
  --infer-interval=0 \
  -r 60

# Measure latency with specific concurrency
perf_analyzer -m model_name \
  -c 10 \
  -d
```

### Key Metrics

- **Inferences Per Second (IPS):** Throughput
- **Average Latency:** Mean time per request
- **P95/P99 Latency:** Tail latencies
- **GPU Utilization:** % of GPU capacity used

## Deployment Best Practices

### Batching Setup Checklist

- [ ] Understand model statefulness (stateless vs. stateful)
- [ ] Choose appropriate batching strategy
- [ ] Configure max_batch_size for GPU memory
- [ ] Set queue delay for latency/throughput trade-off
- [ ] Test with performance analyzer
- [ ] Set preferred batch sizes if needed
- [ ] Configure timeouts and queue policies
- [ ] Monitor in production

### Monitoring

**Metrics to Track:**
- Average batch size
- Queue depth
- Request latency distribution
- GPU utilization
- Model latency vs. queue latency

## Troubleshooting

### Low Batch Sizes in Production

**Cause:** Requests arriving too slowly

**Solutions:**
- Increase `max_queue_delay` to allow waiting
- Adjust application request rate
- Use client batching to pre-batch requests

### High Latency with Batching

**Cause:** Queue delay exceeds computation benefit

**Solutions:**
- Reduce `max_queue_delay`
- Decrease `max_batch_size`
- Use priority levels for critical requests

### Memory Issues

**Cause:** Large batches exceed GPU memory

**Solutions:**
- Reduce `max_batch_size`
- Enable model quantization
- Use sequence/continuous batching

## Conclusion

Triton batching strategies enable optimal performance by automatically combining requests. The choice between dynamic, sequence, and continuous batching depends on model characteristics and deployment requirements.

By carefully configuring batching parameters and measuring performance, organizations can achieve 5-20x throughput improvements while maintaining acceptable latency, enabling efficient inference at scale.
