class ResearchOrchestrator:
    def coordinate_research(self, topic: str):
        # Create root span for entire workflow
        with tracer.start_as_current_span("research_workflow") as span:
            span.set_attribute("research.topic", topic)
            span.set_attribute("workflow.agent_count", 5)

            # Record workflow start state
            span.add_event("workflow_started", {
                "timestamp": time.time(),
                "concurrent_workflows": get_active_workflow_count()
            })

            # Spawn specialist agents with trace context propagation
            tasks = []
            for agent_name in ["planner", "researcher", "analyst", "writer", "reviewer"]:
                # Extract current trace context into carrier dict
                carrier = {}
                propagator.inject(carrier)

                # Create child span for agent execution
                with tracer.start_as_current_span(f"agent_{agent_name}") as agent_span:
                    agent_span.set_attribute("agent.name", agent_name)
                    agent_span.set_attribute("agent.role", self.get_agent_role(agent_name))

                    # Spawn agent with propagated trace context
                    task = self.spawn_agent(agent_name, topic, carrier)
                    tasks.append(task)

            # Wait for all agents to complete (or fail)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results and record failures
            failures = [r for r in results if isinstance(r, Exception)]
            span.set_attribute("workflow.failures.count", len(failures))
            span.set_attribute("workflow.success_rate",
                             (len(results) - len(failures)) / len(results))

            if failures:
                # Record detailed failure information
                for idx, failure in enumerate(failures):
                    span.add_event(f"agent_failure_{idx}", {
                        "error_type": type(failure).__name__,
                        "error_message": str(failure),
                        "agent": results.index(failure)
                    })
                span.set_status(Status(StatusCode.ERROR, "Coordination failures detected"))

            return self.synthesize_results(results)
