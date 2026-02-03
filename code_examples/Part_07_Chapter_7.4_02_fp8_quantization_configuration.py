# FP8 calibration config
calibration_config = {
    "num_calib_batches": 512,
    "calib_batch_size": 1,
    "calib_max_seq_length": 512,
    "quant_algo": "FP8",  # FP8 E4M3 format
    "kv_cache_dtype": "FP8"  # Also quantize KV cache
}
