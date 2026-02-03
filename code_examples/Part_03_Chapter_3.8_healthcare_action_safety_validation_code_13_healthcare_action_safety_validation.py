def validate_healthcare_action_safety(proposed_action: dict,
                                     patient_context: dict,
                                     action_history: list) -> tuple[bool, str]:
    """
    Validate proposed action against safety constraints before execution

    Returns: (safe_to_proceed, explanation)
    """
    action_name = proposed_action["tool"]

    # Check drug allergy constraints
    if action_name == "prescribe_medication":
        medication = proposed_action["parameters"]["medication"]
        allergies = patient_context.get("drug_allergies", [])

        if medication in allergies or any(cross_reacts(medication, allergy)
                                          for allergy in allergies):
            return False, f"Drug allergy violation: {medication} conflicts with {allergies}"

    # Check prerequisite completion
    prerequisites = get_required_prerequisites(action_name)
    for prereq in prerequisites:
        if not action_completed_in_history(prereq, action_history):
            return False, f"Missing prerequisite: {prereq} required before {action_name}"

    # Check temporal constraints
    if not within_valid_time_window(proposed_action, patient_context):
        return False, f"Temporal constraint violated: action outside valid time window"

    return True, "Safety validation passed"