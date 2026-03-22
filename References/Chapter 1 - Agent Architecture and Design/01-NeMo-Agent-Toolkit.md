# NeMo Agent Toolkit Overview

**Source:** 
../Nemo_Agent_Toolkit/
https://developer.nvidia.com/nemo-agent-toolkit

## What It Is

NVIDIA NeMo Agent Toolkit is an open-source framework for constructing, evaluating, and enhancing AI agents across multiple frameworks. The toolkit enables "unified, cross-framework integration across connected AI agent systems" while helping organizations identify performance bottlenecks and optimize workflows efficiently.

## Core Capabilities

**Development Support**: The toolkit offers YAML configuration builders for rapid prototyping and includes reusable tool collections and agentic workflows to streamline system creation.

**Performance Optimization**: It features an Agent Hyperparameter Optimizer that automatically selects optimal settings like LLM type and temperature based on metrics including accuracy, groundedness, and latency.

**Monitoring & Profiling**: Fine-grained telemetry captures detailed metrics on agent coordination, tool efficiency, and computational expenses across multi-agent systems.

**Framework Compatibility**: Works with LangChain, CrewAI, custom frameworks, and supports the Model Context Protocol (MCP), allowing agents to access tools from MCP registries.

## Getting Started

Quick installation via pip:
```
pip install nvidia-nat
nat --help
```

Comprehensive resources include documentation, GitHub repository access, community forums, video tutorials, and starter notebooks demonstrating practical implementation of agentic workflows.

## Key Benefits

Organizations can parallelize workflows, cache expensive operations, evaluate accuracy quickly, and scale from individual agents to enterprise-grade digital workforces while reducing cloud expenditures.
