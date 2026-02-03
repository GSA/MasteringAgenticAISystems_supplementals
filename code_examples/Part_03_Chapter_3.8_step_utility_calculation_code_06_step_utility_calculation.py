def calculate_step_utility(action: dict,
                          state_before: dict,
                          state_after: dict,
                          goal_state: dict) -> float:
    """
    Score action utility: how much did this action advance toward goal?
    Returns: 1.0 (definite progress), 0.5 (neutral/exploratory), 0.0 (regression)
    """
    # Measure distance to goal before and after action
    distance_before = calculate_goal_distance(state_before, goal_state)
    distance_after = calculate_goal_distance(state_after, goal_state)

    if distance_after < distance_before * 0.9:
        # Significant progress toward goal (≥10% distance reduction)
        return 1.0
    elif distance_after < distance_before * 0.99:
        # Marginal progress (distance reduced but <10%)
        return 0.7
    elif distance_after <= distance_before * 1.01:
        # Neutral (distance essentially unchanged, exploratory action)
        return 0.5
    elif distance_after <= distance_before * 1.1:
        # Minor regression (distance increased slightly, correctable)
        return 0.3
    else:
        # Significant regression (moved away from goal)
        return 0.0

# Aggregate step utilities across trajectory
trajectory_utility = np.mean([
    calculate_step_utility(action, states[i], states[i+1], goal)
    for i, action in enumerate(trajectory)
])