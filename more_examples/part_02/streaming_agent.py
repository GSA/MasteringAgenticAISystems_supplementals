"""
Code Example 2.5.1: Streaming Agent with Real-time Response

Purpose: Implement token-by-token streaming for responsive user experience

Concepts Demonstrated:
- Async streaming with Python generators
- Server-Sent Events (SSE) for real-time delivery
- LangChain streaming integration
- Time to First Token (TTFT) optimization
- Error handling for streaming failures

Author: NVIDIA Agentic AI Certification
Chapter: 2, Section: 2.5
Exam Skill: 2.5 - Develop dynamic conversation flows with real-time streaming
"""

import asyncio
import time
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn


# ============================================================================
# SIMULATED LLM (for demonstration)
# ============================================================================

class SimulatedLLM:
    """
    Simulated LLM for demonstration purposes.

    In production: Use OpenAI, NVIDIA NIM, or other LLM APIs with streaming.
    """

    async def astream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream response tokens asynchronously.

        Simulates realistic LLM streaming behavior:
        - Initial latency (TTFT)
        - Token-by-token generation
        - Variable token timing
        """
        # Simulate initial processing latency (TTFT)
        await asyncio.sleep(0.5)  # 500ms TTFT - target is <1s

        # Generate simulated response
        response = self._generate_response(prompt)
        tokens = response.split()

        # Stream tokens one at a time
        for token in tokens:
            yield token + " "
            # Simulate generation time (varied for realism)
            await asyncio.sleep(0.05 + (0.02 * (len(token) / 10)))

    def _generate_response(self, prompt: str) -> str:
        """Generate simulated response based on prompt"""
        if "order" in prompt.lower():
            return (
                "I can help you track your order. Your order #12345 was "
                "shipped on June 10th and should arrive within 3-5 business "
                "days. You can track it using tracking number TRK789XYZ at "
                "our shipping partner's website. Is there anything else you'd "
                "like to know about your order?"
            )
        elif "return" in prompt.lower():
            return (
                "Our return policy allows returns within 30 days of purchase. "
                "To initiate a return, you'll need your order number and a "
                "brief reason for the return. I can help you start the return "
                "process right now if you'd like. Do you have your order number?"
            )
        elif "product" in prompt.lower():
            return (
                "Our products are available in multiple sizes and colors. "
                "The most popular options are Medium in Navy Blue and Large "
                "in Charcoal Gray. All products come with a 1-year warranty "
                "and free shipping on orders over $50. Would you like more "
                "details about a specific product?"
            )
        else:
            return (
                "Thank you for reaching out to our customer support. I'm here "
                "to help with any questions about orders, returns, products, "
                "or account issues. How can I assist you today?"
            )


# ============================================================================
# STREAMING AGENT
# ============================================================================

class StreamingAgent:
    """
    Customer support agent with real-time streaming responses.

    Features:
    - Token-by-token streaming
    - Sub-second Time to First Token (TTFT)
    - Error recovery
    - Performance metrics tracking
    """

    def __init__(self):
        self.llm = SimulatedLLM()
        self.metrics = {
            "total_requests": 0,
            "ttft_sum": 0.0,
            "avg_ttft": 0.0
        }

    async def stream_response(self, query: str) -> AsyncGenerator[str, None]:
        """
        Stream agent response token-by-token.

        Yields:
            str: Individual tokens as they're generated

        Metrics tracked:
            - Time to First Token (TTFT)
            - Total generation time
            - Token count
        """
        start_time = time.time()
        first_token = True
        token_count = 0

        try:
            # Stream from LLM
            async for token in self.llm.astream(query):
                # Track TTFT (time to first token)
                if first_token:
                    ttft = (time.time() - start_time) * 1000  # Convert to ms
                    self._update_ttft_metrics(ttft)
                    first_token = False

                token_count += 1
                yield token

            # Log completion metrics
            total_time = (time.time() - start_time) * 1000
            self._log_completion(query, token_count, total_time)

        except Exception as e:
            # Handle streaming errors gracefully
            yield f"\n\n[Error: {str(e)}. Please try again.]"

    def _update_ttft_metrics(self, ttft: float):
        """Update Time to First Token metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["ttft_sum"] += ttft
        self.metrics["avg_ttft"] = self.metrics["ttft_sum"] / self.metrics["total_requests"]

    def _log_completion(self, query: str, tokens: int, time_ms: float):
        """Log completion metrics"""
        print(f"\n[Metrics] Query: {query[:30]}...")
        print(f"  Tokens: {tokens}")
        print(f"  Time: {time_ms:.1f}ms")
        print(f"  TTFT: {self.metrics['avg_ttft']:.1f}ms (avg)")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Streaming Agent API")
agent = StreamingAgent()


@app.get("/stream")
async def stream_agent_response(query: str):
    """
    Streaming endpoint using Server-Sent Events (SSE).

    Args:
        query: User query string

    Returns:
        StreamingResponse: SSE stream of tokens

    Example:
        GET /stream?query=Where+is+my+order

        Event stream:
        data: I
        data: can
        data: help
        data: you
        data: track
        ...
    """

    async def event_generator():
        """Generate SSE-formatted events"""
        async for token in agent.stream_response(query):
            # SSE format: "data: {content}\n\n"
            yield f"data: {token}\n\n"

        # Send completion marker
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/batch")
async def batch_agent_response(query: str):
    """
    Non-streaming (batch) endpoint for comparison.

    Returns entire response at once after generation completes.
    """
    response_tokens = []

    async for token in agent.stream_response(query):
        response_tokens.append(token)

    full_response = "".join(response_tokens)

    return {"response": full_response}


@app.get("/metrics")
async def get_metrics():
    """Get agent performance metrics"""
    return agent.metrics


# ============================================================================
# CLIENT SIMULATION
# ============================================================================

async def simulate_streaming_client(query: str):
    """
    Simulate client receiving streaming response.

    In browser JavaScript, use EventSource:

    ```javascript
    const eventSource = new EventSource('/stream?query=test');
    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();
        } else {
            document.getElementById('response').innerHTML += event.data;
        }
    };
    ```
    """
    print(f"\n{'='*60}")
    print(f"Streaming Response Demo")
    print(f"{'='*60}")
    print(f"Query: {query}\n")
    print("Response (streaming):")

    response_tokens = []
    start_time = time.time()
    first_token_time = None

    async for token in agent.stream_response(query):
        # Print token as it arrives (simulating UI update)
        print(token, end="", flush=True)
        response_tokens.append(token)

        # Track first token time
        if first_token_time is None:
            first_token_time = time.time()

    total_time = time.time() - start_time
    ttft = first_token_time - start_time if first_token_time else 0

    print(f"\n\n{'='*60}")
    print(f"Performance Metrics:")
    print(f"  Time to First Token (TTFT): {ttft*1000:.1f}ms")
    print(f"  Total Time: {total_time*1000:.1f}ms")
    print(f"  Total Tokens: {len(response_tokens)}")
    print(f"  Tokens/sec: {len(response_tokens)/total_time:.1f}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def demo():
    """Run demo of streaming vs batch"""
    print("\n" + "="*60)
    print("Streaming Agent - Code Example 2.5.1")
    print("="*60)

    # Demo 1: Streaming
    await simulate_streaming_client("Where is my order?")

    # Demo 2: Different query
    await simulate_streaming_client("What sizes does this product come in?")

    # Demo 3: Return policy
    await simulate_streaming_client("What is your return policy?")

    print("\n" + "="*60)
    print("Key Takeaways:")
    print("="*60)
    print("✓ TTFT < 1 second creates perception of instant response")
    print("✓ Tokens stream continuously - no blank screen wait")
    print("✓ User sees progress immediately")
    print("✓ Same total latency, vastly better UX")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())

    # To run FastAPI server:
    # uvicorn streaming_agent:app --reload
    # Then access:
    # - Streaming: http://localhost:8000/stream?query=test
    # - Batch: http://localhost:8000/batch?query=test
    # - Metrics: http://localhost:8000/metrics
