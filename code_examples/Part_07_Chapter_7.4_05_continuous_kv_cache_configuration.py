# Disable paged KV cache for lowest latency
build_config.plugin_config.set_paged_kv_cache(False)
build_config.plugin_config.set_remove_input_padding(True)  # Still optimize padding
