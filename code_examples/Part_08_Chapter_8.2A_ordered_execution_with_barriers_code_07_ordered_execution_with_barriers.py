class OrderedExecution:
    def __init__(self, dependency_graph: DependencyGraph):
        self.graph = dependency_graph
        self.completed_agents = {}
        self.completion_events = {}

    async def execute_workflow(self, topic: str):
        with tracer.start_as_current_span("ordered_workflow") as span:
            span.set_attribute("research.topic", topic)

            # Get phase-based execution order
            execution_phases = self.graph.get_execution_order()
            span.set_attribute("workflow.phases", len(execution_phases))

            all_results = {}

            # Execute each phase sequentially
            for phase_idx, agents_in_phase in enumerate(execution_phases):
                span.add_event(f"phase_{phase_idx}_started", {
                    "agents": agents_in_phase,
                    "agent_count": len(agents_in_phase)
                })

                # Within phase, execute agents in parallel
                phase_tasks = []
                for agent_name in agents_in_phase:
                    task = self.execute_agent_with_barriers(
                        agent_name, topic, all_results
                    )
                    phase_tasks.append((agent_name, task))

                # Wait for all agents in this phase to complete
                phase_results = await asyncio.gather(
                    *[task for _, task in phase_tasks],
                    return_exceptions=True
                )

                # Record results and check for phase failures
                phase_failures = []
                for idx, (agent_name, result) in enumerate(
                    zip([name for name, _ in phase_tasks], phase_results)
                ):
                    if isinstance(result, Exception):
                        phase_failures.append((agent_name, result))
                        span.add_event(f"phase_{phase_idx}_agent_failed", {
                            "agent": agent_name,
                            "error": str(result)
                        })
                    else:
                        all_results[agent_name] = result
                        self.completed_agents[agent_name] = result

                # Phase barrier: Stop workflow if phase has failures
                if phase_failures:
                    span.set_status(
                        Status(StatusCode.ERROR,
                              f"Phase {phase_idx} failures: {len(phase_failures)}")
                    )
                    raise WorkflowFailureException(
                        f"Phase {phase_idx} failed: {phase_failures}"
                    )

                span.add_event(f"phase_{phase_idx}_completed", {
                    "successful_agents": len(agents_in_phase) - len(phase_failures),
                    "phase_duration": time.time() - phase_start
                })

            return all_results

    async def execute_agent_with_barriers(
        self, agent_name: str, topic: str, completed_results: dict
    ):
        """Execute agent only after dependencies complete."""

        with tracer.start_as_current_span(f"{agent_name}_execution") as span:
            # Wait for all dependencies to complete
            dependencies = self.graph.dependencies.get(agent_name, [])
            span.set_attribute("agent.dependencies", dependencies)

            for dep_name in dependencies:
                span.add_event(f"waiting_for_{dep_name}")

                # Block until dependency completes
                while dep_name not in self.completed_agents:
                    await asyncio.sleep(0.1)  # Check every 100ms

                span.add_event(f"dependency_{dep_name}_satisfied")

            # All dependencies satisfied - execute agent
            span.add_event("dependencies_satisfied_starting_execution")

            # Pass dependency outputs to agent
            dep_outputs = {
                dep: completed_results[dep]
                for dep in dependencies
            }

            result = await self.execute_agent(agent_name, topic, dep_outputs)

            span.set_attribute("agent.result.size", len(str(result)))
            span.add_event("execution_completed")

            return result
