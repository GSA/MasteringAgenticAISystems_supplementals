"""
Weaviate Production Client
Demonstrates: Authentication, collections, indexing, querying
"""

import weaviate
from weaviate.auth import AuthApiKey
import os

# Initialize client with authentication
client = weaviate.Client(
    url="http://localhost:8080",
    auth_client_secret=AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
    timeout_config=(5, 60),  # (connect, read) timeouts
    additional_headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]  # For vectorization
    }
)
