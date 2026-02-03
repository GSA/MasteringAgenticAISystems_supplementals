"""
NVIDIA NeVA 22B for Visual Question Answering
Demonstrates detailed image understanding for RAG preprocessing
"""

import requests
import base64
from typing import Dict, Any

class NeVAClient:
    """
    Client for NVIDIA NeVA 22B via NVIDIA NIM API
    """

    def __init__(self, api_key: str, endpoint: str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/neva-22b"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API submission"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate_caption(self, image_path: str, detail_level: str = "high") -> str:
        """
        Generate detailed caption for image

        Args:
            image_path: Path to image file
            detail_level: "low" (brief), "high" (detailed), "extreme" (comprehensive)

        Returns:
            Text caption describing image content
        """
        image_b64 = self.encode_image(image_path)

        prompt_map = {
            "low": "Briefly describe this image in one sentence.",
            "high": "Describe this image in detail, including objects, layout, colors, and context.",
            "extreme": "Provide a comprehensive description of this image, including all visible elements, their relationships, spatial arrangement, colors, text content (if any), and overall context."
        }

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt_map[detail_level]
                }
            ],
            "image": image_b64,
            "max_tokens": 500,
            "temperature": 0.2  # Low temperature for factual descriptions
        }

        response = requests.post(self.endpoint, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def visual_question_answering(self, image_path: str, question: str) -> str:
        """
        Answer a specific question about image content

        Args:
            image_path: Path to image file
            question: Natural language question about the image

        Returns:
            Answer based on visual analysis
        """
        image_b64 = self.encode_image(image_path)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Answer this question about the image: {question}"
                }
            ],
            "image": image_b64,
            "max_tokens": 200,
            "temperature": 0.1  # Very low for factual answers
        }

        response = requests.post(self.endpoint, headers=self.headers, json=payload)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

# Example usage for RAG preprocessing
neva = NeVAClient(api_key="nvapi-xxx")

# Generate caption for indexing
caption = neva.generate_caption(
    "reports/architecture_diagram.png",
    detail_level="high"
)

print(f"Caption for indexing: {caption}")
# Output: "A distributed system architecture diagram showing three microservices
# (API Gateway, Processing Service, Database) connected via message queues.
# The API Gateway receives HTTPS requests, routes to Processing Service via
# RabbitMQ, which persists data to PostgreSQL database. Each service is
# containerized in Docker with health check endpoints."

# Later, during inference with retrieved image
question = "What message queue technology is used between services?"
answer = neva.visual_question_answering(
    "reports/architecture_diagram.png",
    question
)

print(f"VQA Answer: {answer}")
# Output: "RabbitMQ is used as the message queue between the API Gateway
# and Processing Service."
