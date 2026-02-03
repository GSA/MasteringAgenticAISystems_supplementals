from typing import Dict, List
from pydantic import BaseModel
from openai import OpenAI

class PerceptionOutput(BaseModel):
    """Structured information extracted from raw input"""
    user_id: str
    issue_category: str  # hardware, software, account
    entities: List[str]  # products, error codes mentioned
    sentiment: str  # frustrated, neutral, satisfied
    implicit_references: List[str]  # "same issue", "like before"
    urgency: int  # 1-5 scale

class PerceptionModule:
    def __init__(self, llm_client: OpenAI):
        self.client = llm_client

    def process_input(self, raw_message: str, user_id: str) -> PerceptionOutput:
        """Transform raw support ticket into structured information"""

        # Use structured output to extract entities and context
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": """Extract structured information from support tickets.
                Identify issue categories, mentioned entities, user sentiment, and implicit references
                to previous interactions."""},
                {"role": "user", "content": raw_message}
            ],
            response_format=PerceptionOutput
        )

        perception = completion.choices[0].message.parsed
        perception.user_id = user_id

        return perception
