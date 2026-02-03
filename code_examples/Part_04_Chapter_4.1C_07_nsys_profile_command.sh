#!/bin/bash
# Nsight Systems profiling command with comprehensive tracing

nsys profile \
  --output=/workspace/output/agent_profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sample=cpu \
  --cpuctxsw=none \
  python agent_inference.py
