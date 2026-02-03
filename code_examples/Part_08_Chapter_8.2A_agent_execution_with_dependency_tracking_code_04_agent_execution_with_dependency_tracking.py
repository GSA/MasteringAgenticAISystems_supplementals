async def spawn_agent(self, name: str, topic: str, trace_context: dict):
    # Extract parent trace context to maintain causality
    ctx = propagator.extract(trace_context)

    # Create agent span as child of orchestrator span
    with tracer.start_as_current_span(f"{name}_execution", context=ctx) as span:
        span.set_attribute("agent.name", name)
        span.set_attribute("agent.input.topic", topic)
        span.add_event("agent_started")

        try:
            # Track dependency resolution
            dependencies = self.get_agent_dependencies(name)
            span.set_attribute("agent.dependencies.count", len(dependencies))

            # Record dependency wait times
            for dep_name in dependencies:
                dep_start = time.time()
                span.add_event(f"waiting_for_{dep_name}")

                # Wait for dependency agent to complete
                dep_result = await self.wait_for_agent(dep_name)

                dep_wait_time = time.time() - dep_start
                span.set_attribute(f"agent.dependency.{dep_name}.wait_seconds",
                                 dep_wait_time)

                if dep_wait_time > 10:  # Threshold for suspicious delays
                    span.add_event(f"slow_dependency_{dep_name}", {
                        "wait_seconds": dep_wait_time,
                        "threshold_exceeded": True
                    })

            # Execute agent logic with timing
            exec_start = time.time()
            result = await self.execute_agent(name, topic)
            exec_time = time.time() - exec_start

            span.set_attribute("agent.execution.duration_seconds", exec_time)
            span.set_attribute("agent.output.size_bytes", len(str(result)))
            span.add_event("agent_completed")

            return result

        except Exception as e:
            # Record detailed failure information
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.add_event("agent_failed", {
                "exception": str(e),
                "traceback": traceback.format_exc()
            })
            raise
