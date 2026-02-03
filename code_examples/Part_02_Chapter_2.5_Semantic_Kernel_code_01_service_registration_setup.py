from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from my_enterprise.data import DatabaseClient, CRMClient

# Initialize kernel with dependency injection
kernel = Kernel()

# Register AI service (Azure OpenAI for enterprise compliance)
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )
)

# Register enterprise dependencies
kernel.add_service(DatabaseClient(connection_string=os.getenv("DB_CONNECTION")))
kernel.add_service(CRMClient(api_key=os.getenv("CRM_API_KEY")))
