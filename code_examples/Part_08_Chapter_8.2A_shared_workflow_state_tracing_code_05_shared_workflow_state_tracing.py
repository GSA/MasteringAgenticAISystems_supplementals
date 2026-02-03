class SharedWorkflowState:
    def __init__(self, trace_context: dict):
        self.state = {}
        self.version = 0
        self.lock = asyncio.Lock()

        # Extract trace context for state operations
        self.trace_ctx = propagator.extract(trace_context)

    async def read(self, agent: str, key: str):
        with tracer.start_as_current_span("state_read", context=self.trace_ctx) as span:
            span.set_attribute("state.operation", "read")
            span.set_attribute("state.agent", agent)
            span.set_attribute("state.key", key)
            span.set_attribute("state.version", self.version)

            async with self.lock:
                read_start = time.time()
                value = self.state.get(key)
                read_time = time.time() - read_start

                span.set_attribute("state.read.duration_ms", read_time * 1000)
                span.set_attribute("state.read.value_size_bytes",
                                 len(str(value)) if value else 0)

                if read_time > 0.1:  # 100ms threshold
                    span.add_event("slow_state_read", {
                        "duration_ms": read_time * 1000,
                        "concurrent_operations": self.get_pending_operations()
                    })

                return value, self.version

    async def write(self, agent: str, key: str, value: any, expected_version: int):
        with tracer.start_as_current_span("state_write", context=self.trace_ctx) as span:
            span.set_attribute("state.operation", "write")
            span.set_attribute("state.agent", agent)
            span.set_attribute("state.key", key)
            span.set_attribute("state.expected_version", expected_version)
            span.set_attribute("state.current_version", self.version)

            async with self.lock:
                # Detect version conflicts (concurrent modification)
                if self.version != expected_version:
                    span.add_event("version_conflict", {
                        "expected": expected_version,
                        "actual": self.version,
                        "conflict_detected": True
                    })
                    span.set_status(Status(StatusCode.ERROR, "Version conflict"))
                    raise ValueError(
                        f"Concurrent modification detected: expected version "
                        f"{expected_version}, current version {self.version}"
                    )

                write_start = time.time()
                self.state[key] = value
                self.version += 1
                write_time = time.time() - write_start

                span.set_attribute("state.write.duration_ms", write_time * 1000)
                span.set_attribute("state.new_version", self.version)
                span.add_event("state_updated", {
                    "new_version": self.version,
                    "value_size_bytes": len(str(value))
                })
