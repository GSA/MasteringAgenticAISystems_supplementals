#!/bin/bash
# Run profiling on ReAct agent

# Run profiling (captures ~30 seconds by default)
nsys profile \
  --output=/workspace/output/react_agent \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  python agent_inference.py

# Output files generated:
# - react_agent.nsys-rep (binary report)
# - react_agent.sqlite (timeline database for GUI analysis)
