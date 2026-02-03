MAX_ITERATIONS = 15
seen_states = set()

for iteration in range(MAX_ITERATIONS):
    # Hash current state to detect cycles
    state_hash = hash_state(state)
    if state_hash in seen_states:
        raise LoopDetectedError(
            f"Detected repeated state at iteration {iteration}. "
            f"Agent has entered a cycle and cannot make progress."
        )
    seen_states.add(state_hash)

    # Check for task completion
    if task_complete(state):
        logger.info(f"Task completed successfully in {iteration} iterations")
        break

    # Execute next action
    action = agent.decide_action(state)
    new_state = execute_action(action)

    # Validate progress
    if not made_progress(state, new_state):
        logger.warning(f"No progress at iteration {iteration}, adjusting strategy")
        new_state = fallback_strategy(state)

    state = new_state
else:
    # Loop exhausted without completion
    raise MaxIterationsError(
        f"Failed to complete task within {MAX_ITERATIONS} iterations. "
        f"Task may be too complex or approach may be inappropriate."
    )
