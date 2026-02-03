# Initialize kernel and register services
kernel = Kernel()

# Register AI service for semantic functions
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )
)

# Register CRM plugin
crm_plugin = EnterpriseCRMPlugin(
    crm_api_base_url=os.getenv("CRM_API_BASE_URL"),
    api_key=os.getenv("CRM_API_KEY"),
    kernel=kernel
)
kernel.add_plugin(crm_plugin, plugin_name="crm")

# Set up automatic orchestration
from semantic_kernel.planners import FunctionCallingStepwisePlanner
planner = FunctionCallingStepwisePlanner(kernel=kernel, max_iterations=8)

# Complex user query requiring multi-step orchestration
user_query = """
Customer C12345 just sent an urgent email complaining about billing issues.
Can you:
1. Pull their profile to understand their account standing
2. Analyze the sentiment of their message: "I've been charged twice this month
   and no one is responding to my emails. This is completely unacceptable."
3. Draft a personalized response acknowledging the issue and explaining next steps
"""

# Execute with automatic orchestration
result = await planner.invoke(user_query)
print(result)
