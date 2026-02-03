#!/bin/bash
# Launch Nsight Systems container with GPU access

docker run -it --rm \
  --gpus all \
  -v ~/nsight-profiling:/workspace \
  nvcr.io/nvidia/nsight-systems:2025.5.1 \
  bash
