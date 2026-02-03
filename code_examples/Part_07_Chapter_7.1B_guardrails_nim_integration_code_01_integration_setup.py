from nemoguardrails import RailsConfig, LLMRails
from openai import OpenAI

# Initialize NIM client
nim_client = OpenAI(
    api_key="your-nim-api-key",
    base_url="http://nim.example.com/v1"
)

# Load guardrails configuration
config = RailsConfig.from_path("./config")

# Wrap NIM with guardrails
rails = LLMRails(config, llm=nim_client)

# Make protected inference request
response = rails.generate(
    messages=[
        {"role": "user", "content": "What is my social security number?"}
    ]
)

# Guardrails automatically enforce policies
print(response["content"])
# Output: "I cannot provide personally identifiable information..."
