# Load state from iteration 2 (before the failure)
checkpoint = app.get_state(config=config, checkpoint_id="iteration_2")

# Modify state to test alternative approach
modified_state = checkpoint.values.copy()
modified_state["max_iterations"] = 5  # Allow more attempts
modified_state["temperature"] = 0.1   # Make generation more conservative

# Resume from modified state
new_final_state = app.invoke(modified_state, config=config)
