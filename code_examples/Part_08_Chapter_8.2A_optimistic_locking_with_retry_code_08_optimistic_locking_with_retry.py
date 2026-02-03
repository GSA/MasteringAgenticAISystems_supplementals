class SharedWorkflowState:
    def __init__(self, trace_context: dict):
        self.state = {}
        self.version = 0
        self.lock = asyncio.Lock()
        self.trace_ctx = propagator.extract(trace_context)

    async def read(self, agent: str, key: str):
        """Read with version tracking for optimistic locking."""

        with tracer.start_as_current_span("state_read", context=self.trace_ctx) as span:
            span.set_attribute("state.operation", "read")
            span.set_attribute("state.agent", agent)
            span.set_attribute("state.key", key)

            async with self.lock:
                value = self.state.get(key)
                current_version = self.version

                span.set_attribute("state.version", current_version)
                span.set_attribute("state.value_exists", value is not None)

                return value, current_version

    async def write_with_retry(
        self, agent: str, key: str, value: any,
        expected_version: int, max_retries: int = 3
    ):
        """Write with automatic retry on version conflicts."""

        retry_count = 0

        while retry_count < max_retries:
            try:
                await self.write(agent, key, value, expected_version)
                return  # Success

            except VersionConflictError as e:
                retry_count += 1

                with tracer.start_as_current_span(
                    "state_write_retry", context=self.trace_ctx
                ) as span:
                    span.set_attribute("retry.attempt", retry_count)
                    span.set_attribute("retry.max_attempts", max_retries)
                    span.add_event("version_conflict_retrying", {
                        "expected_version": expected_version,
                        "actual_version": e.actual_version,
                        "conflicts": retry_count
                    })

                if retry_count >= max_retries:
                    raise MaxRetriesExceeded(
                        f"Failed to write after {max_retries} attempts "
                        f"due to persistent version conflicts"
                    )

                # Re-read to get current version and apply changes
                current_value, expected_version = await self.read(agent, key)
                value = self.merge_changes(current_value, value, agent)

                # Exponential backoff
                await asyncio.sleep(0.1 * (2 ** retry_count))

    async def write(
        self, agent: str, key: str, value: any, expected_version: int
    ):
        """Write with optimistic locking."""

        with tracer.start_as_current_span("state_write", context=self.trace_ctx) as span:
            span.set_attribute("state.operation", "write")
            span.set_attribute("state.agent", agent)
            span.set_attribute("state.key", key)
            span.set_attribute("state.expected_version", expected_version)

            async with self.lock:
                span.set_attribute("state.current_version", self.version)

                # Optimistic locking check
                if self.version != expected_version:
                    span.add_event("version_conflict", {
                        "expected": expected_version,
                        "actual": self.version,
                        "writer_agent": agent
                    })
                    span.set_status(
                        Status(StatusCode.ERROR, "Version conflict")
                    )
                    raise VersionConflictError(
                        expected=expected_version,
                        actual=self.version,
                        agent=agent
                    )

                # Version matches - safe to write
                self.state[key] = value
                self.version += 1

                span.set_attribute("state.new_version", self.version)
                span.add_event("state_updated", {
                    "new_version": self.version,
                    "value_size": len(str(value))
                })

    def merge_changes(self, current_value: any, new_value: any, agent: str) -> any:
        """Merge conflicting changes when possible."""

        # Application-specific merge logic
        # For research reports, different agents update different sections

        if isinstance(current_value, dict) and isinstance(new_value, dict):
            # Merge dict updates (non-overlapping keys)
            merged = current_value.copy()
            merged.update(new_value)
            return merged

        # For non-mergeable types, new value overwrites
        return new_value
