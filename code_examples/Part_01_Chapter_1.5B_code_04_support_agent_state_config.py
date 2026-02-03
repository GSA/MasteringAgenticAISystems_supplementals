"""
NVIDIA Platform Example: Logic Tree Implementation with LangGraph

Demonstrates:
- Integration with NVIDIA NIM for fast inference
- Graph-based control flow with conditional edges
- State management for multi-turn conversations
- Best practices for production deployment
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# ====================================================================
# STATE SCHEMA
# ====================================================================

class SupportAgentState(TypedDict):
    """
    State schema for customer support routing agent.

    This defines all information that persists across the agent's
    decision tree traversal.
    """
    user_query: str
    query_category: Literal["billing", "technical", "general", "escalation"]
    conversation_history: list[dict]
    resolution_status: Literal["pending", "resolved", "escalated"]
    agent_responses: list[str]
    metadata: dict  # Tracking info: timestamps, confidence scores, etc.


# ====================================================================
# NVIDIA NIM CONFIGURATION
# ====================================================================

# Configure NVIDIA NIM endpoint for fast inference
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",  # High-quality routing model
    nvidia_api_key="nvapi-...",  # Your NIM API key
    temperature=0.1,  # Low temperature for consistent routing
    max_tokens=200,  # Short responses for routing decisions
)

# Performance configuration optimized for NVIDIA platform
nim_config = {
    "use_tensorrt": True,      # Enable TensorRT optimization
    "batch_size": 1,            # Real-time routing (no batching delay)
    "precision": "fp16",        # Half-precision for 2x throughput
    "streaming": False,         # Routing doesn't need streaming
}
