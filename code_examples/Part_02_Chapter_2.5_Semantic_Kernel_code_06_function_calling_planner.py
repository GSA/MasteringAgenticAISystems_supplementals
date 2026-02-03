from semantic_kernel.planners import FunctionCallingStepwisePlanner

# Configure automatic function calling orchestration
planner = FunctionCallingStepwisePlanner(
    kernel=kernel,
    max_iterations=10,  # Prevent infinite loops
    max_tokens=4000
)

# User query triggers dynamic routing
user_query = "What's the status of customer C12345's recent order, and can you send them a personalized thank you message?"

# Kernel orchestrates plugin invocations
result = await planner.invoke(user_query)

# Behind the scenes, the LLM:
# 1. Analyzes query: needs customer data + order status + message generation
# 2. Routes to customer_service.get_customer_profile(customer_id="C12345")
# 3. Routes to order_service.get_order_status(customer_id="C12345")
# 4. Routes to messaging_service.send_personalized_message(
#       customer_id="C12345",
#       message_type="thank_you",
#       context=<profile + order data>
#    )
# 5. Synthesizes results into coherent response
