def calculate_tool_selection_accuracy(actual_trajectory: list,
                                     reference_trajectory: list) -> float:
    """Measure tool selection accuracy at each step"""
    # Align trajectories (handle length differences)
    aligned_pairs = align_trajectories(actual_trajectory, reference_trajectory)

    correct_selections = sum(
        1 for actual_step, ref_step in aligned_pairs
        if actual_step['tool'] == ref_step['tool']
    )

    return correct_selections / len(aligned_pairs)