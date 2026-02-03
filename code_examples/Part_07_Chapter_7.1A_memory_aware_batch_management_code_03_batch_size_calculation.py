# Monitor memory and adjust batch size dynamically
import psutil
import torch

def get_max_batch_size(draft_model, target_model, gpu_memory_gb):
    draft_memory = draft_model.memory_footprint_gb
    target_memory = target_model.memory_footprint_gb
    available = gpu_memory_gb - draft_memory - target_memory - 2  # 2GB buffer

    # Each request consumes ~2GB KV cache
    max_batch = int(available / 2)
    return max(1, max_batch)
