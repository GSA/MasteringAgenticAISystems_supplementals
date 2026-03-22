# NeMo RL: Reinforcement Learning for Agentic AI

**Source:** https://docs.nvidia.com/nemo/rl/ and https://github.com/NVIDIA-NeMo/RL

**Version:** Latest stable release
**Framework:** NVIDIA NeMo RL
**Focus:** Scalable reinforcement learning for multimodal models and agents

## Overview

NeMo RL is an open-source post-training framework from NVIDIA designed for reinforcement learning on multimodal models (LLMs, VLMs, etc.). It emphasizes "flexibility, reproducibility, and scale" with support for both single-GPU experiments and distributed multi-node deployments, enabling efficient training of agents from 1B to 70B+ parameters.

## Architecture & Design

### Core Design Principles

**Scalability** - Distributed infrastructure supporting heterogeneous hardware from laptops to massive clusters

**Flexibility** - Support for diverse learning algorithms and model architectures

**Reproducibility** - Deterministic training with controlled randomization

**Production-Ready** - Enterprise-grade error handling and monitoring

## Training Backends

### DTensor (PyTorch Native)

**Framework:** PyTorch's next-generation distributed training system

**Parallelism Strategies:**
- Tensor Parallelism (TP)
- Sequence Parallelism (SP)
- Pipeline Parallelism (PP)
- Context Parallelism (CP)
- Fully Sharded Data Parallelism (FSDP2)

**Advantages:**
- Native PyTorch support
- Better integration with new models
- Improved debugging

### Megatron Core (High-Performance)

**Framework:** NVIDIA's optimized training engine

**Capabilities:**
- 6D parallelism support
- Maximum training throughput
- Extreme-scale optimization

**Use Cases:**
- Very large model training (100B+)
- Production deployments requiring peak performance
- Data center scale training

## Generation Backends

### vLLM Integration

**Inference Engine:** High-throughput, memory-efficient inference

**Features:**
- Fast token generation for rollouts
- Batching support
- Memory-optimized operation

### Megatron Native Inference

**Direct Integration:** Native Megatron inference without weight conversion

**Benefits:**
- Zero-copy inference
- Maximum performance
- Seamless training-inference pipeline

## Supported Algorithms

### 1. GRPO (Group Relative Policy Optimization)

**Purpose:** Policy optimization with relative advantage estimation

**Features:**
- Environment interaction support
- Group-based reward estimation
- Efficient policy updates

**Use Cases:**
- General RL problems
- Mathematical reasoning
- Complex task learning

### 2. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

**Innovation:** New algorithm decoupling clipping and sampling strategies

**Advantages:**
- More stable training
- Better sample efficiency
- Improved convergence properties

### 3. On-Policy Distillation

**Approach:** Student-teacher alignment via KL divergence

**Process:**
1. Teacher generates high-quality rollouts
2. Student learns from teacher distribution
3. Policy refinement through guided learning

**Benefits:**
- Knowledge transfer
- Computational efficiency
- Quality improvement

### 4. SFT (Supervised Fine-Tuning)

**Purpose:** Foundation for RL pre-training

**Process:**
- Teach model correct behavior patterns
- Establish initial policy
- Reduce action space uncertainty

### 5. DPO (Direct Preference Optimization)

**Advantage:** Eliminates reward model training

**Process:**
1. Collect preference pairs (good vs. bad responses)
2. Train on preference distribution directly
3. Implicit reward learning

### 6. RM (Reward Model Training)

**Purpose:** Train reward evaluators for RL

**Function:**
- Score completions
- Guide policy optimization
- Evaluate solution quality

## Multi-Turn Generation & Tool Use

### Multi-Turn RL Support

**Conversational Agents:**
- Extended dialogue optimization
- Context management
- Response quality improvement

### Tool-Use Integration

**Agent Capabilities:**
- Function calling
- API integration
- External tool coordination

**Training:** RL-optimize tool-selection policies

### Game Integration

**Reinforcement Learning Domains:**
- Game-based learning environments
- Competitive policy training
- Complex interaction modeling

## Distributed Training Infrastructure

### Ray-Based Orchestration

**Resource Management:** Efficient Ray-based distributed infrastructure

**Features:**
- Heterogeneous hardware support
- Dynamic resource allocation
- Fault tolerance

### Process Isolation

**Actor Architecture:** Process isolation between RL actors eliminates global state conflicts

**Benefits:**
- Thread-safe operations
- Independent environment management
- Reliable distributed execution

### Multi-Environment Support

**Dependency Isolation:** Multiple training environments with independent dependencies

**Flexibility:**
- Environment-specific configurations
- Concurrent environment training
- Parallel experience collection

### Sequence Packing

**Optimization:** Efficient batch composition through sequence packing

**Impact:**
- GPU memory efficiency
- Improved throughput
- Reduced training time

## Model Support & Scale

### Supported Model Families

| Model | Size | Status |
|---|---|---|
| Qwen | 1B-14B | ✓ Full support |
| Llama | 1B-70B | ✓ Full support |
| DeepSeek-V3 | 1B+ | ✓ Full support |
| Qwen-3 MoE | 1B+ | ✓ Full support |
| Custom Models | Any | ✓ Via API |

### Scale Range

- **Minimum:** 1B parameters (edge devices)
- **Maximum:** 70B+ parameters (distributed clusters)
- **Scaling:** Linear efficiency to 1000+ GPUs

### Sequence Length Support

**Extended Context:** Via parallelism strategies (SP, CP)

**Applications:**
- Long-document reasoning
- Extended conversation
- Complex problem-solving

### Specialization

**VLM Support:** Vision-language model training (in development)

**Custom Architectures:** Framework supports new model families

## Installation & Quick Start

### Environment Setup

```bash
# Clone repository
git clone --recursive github.com/NVIDIA-NeMo/RL.git
cd nemo-rl

# Create virtual environment with uv
uv venv

# Run example training
uv run python examples/run_grpo_math.py
```

### Dependency Management

**UV Package Manager:**
- Deterministic environment creation
- Reproducible builds
- Faster resolution than pip

**Recommendation:** Use `uv run` instead of virtual environment activation for consistency

## Training Pipeline

### 1. Data Preparation

- Collect training examples
- Format for algorithm requirements
- Prepare preference pairs if needed

### 2. Model Configuration

- Select base model
- Configure parallelism strategy
- Set training hyperparameters

### 3. Training Execution

```bash
python train.py \
  --config-path nemo_rl/configs \
  --config-name grpo_math \
  model.hf_model_name=llama-2-7b
```

### 4. Evaluation & Validation

- Monitor training metrics
- Evaluate on test set
- Compare against baseline

## Integration with Agent Systems

### Agent Policy Training

NeMo RL trains the policy backbone for:
- Decision-making agents
- Tool-using systems
- Multi-step reasoning agents

### Feedback Loops

**Reinforcement Learning Cycle:**
1. Agent takes actions
2. Receives environment feedback
3. Policy updates via RL algorithms
4. Improved decision-making

## Performance & Benchmarks

### Training Efficiency

**Speed:** Typical speedups from distributed training:
- 10x with 10 GPUs
- 100x with 100 GPUs
- Near-linear scaling to 1000+ GPUs

**Convergence:** Stable training curves with various algorithms

### Quality Improvements

**Typical Gains:**
- 10-20% accuracy improvements over SFT
- Better generalization on unseen tasks
- Improved reasoning quality

## Best Practices

### Training Strategy

1. **Start with SFT** - Establish baseline behavior
2. **Validate Data** - Ensure quality examples
3. **Small-Scale Trials** - Single GPU validation
4. **Gradual Scaling** - Add GPUs as confidence grows
5. **Hyperparameter Tuning** - Optimize learning rates, batch sizes
6. **Monitor Convergence** - Watch training metrics
7. **Evaluate Regularly** - Test on held-out set

### Production Deployment

- Use NGC containers for consistency
- Version control training configs
- Log experiments systematically
- Monitor policy performance
- Plan rollback strategies

## Conclusion

NeMo RL provides production-ready reinforcement learning for training advanced agentic AI systems. Its flexible algorithm support, efficient distributed training, and seamless integration with NeMo models make it ideal for organizations building goal-oriented AI agents that improve through interaction and feedback.

From mathematical reasoning agents to tool-using assistants, NeMo RL enables continuous policy improvement at scale.
