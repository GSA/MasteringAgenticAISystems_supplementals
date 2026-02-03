from semantic_kernel.functions import FunctionChoiceBehavior

# Require specific function invocation (deterministic)
result = await kernel.invoke_prompt(
    prompt="Calculate customer C12345's lifetime value",
    function_choice_behavior=FunctionChoiceBehavior.Required(
        filters={"included_functions": ["customer_analytics.calculate_lifetime_value"]}
    )
)

# Auto-select from specific plugin only (constrained routing)
result = await kernel.invoke_prompt(
    prompt=user_query,
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={"included_plugins": ["customer_service", "order_management"]}
    )
)
