#!/bin/bash
# Deploy INT8 quantized model
docker run --gpus all \
  -p 8000:8000 \
  nvcr.io/nvidia/nim:llama2-7b-instruct-int8
