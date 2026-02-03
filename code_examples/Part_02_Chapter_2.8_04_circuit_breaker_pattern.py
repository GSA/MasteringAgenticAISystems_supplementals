from typing import Callable, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
import logging
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.

    Monitors failure rates and latency, automatically opening circuit
    when thresholds are exceeded to prevent cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: int = 60,
        latency_threshold: float = 5.0
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes in half-open before closing
            timeout: Seconds to wait before entering half-open state
            latency_threshold: Maximum acceptable latency in seconds
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.latency_threshold = latency_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None

        self.lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function protected by circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self.lock:
            current_state = self._get_state()

            if current_state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker OPEN. Last failure: {self.last_failure_time}. "
                    f"Retry after {self.timeout}s timeout."
                )

            if current_state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker HALF-OPEN. Testing recovery...")

        # Execute function with latency monitoring
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)

            # Check latency
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.latency_threshold:
                logger.warning(
                    f"Function exceeded latency threshold: {elapsed:.2f}s > {self.latency_threshold}s"
                )
                self._record_failure()
                raise CircuitBreakerError(
                    f"Latency threshold exceeded: {elapsed:.2f}s"
                )

            # Success
            self._record_success()
            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _get_state(self) -> CircuitState:
        """
        Determine current circuit state.

        Automatically transitions from OPEN to HALF-OPEN after timeout.
        """
        if self.state == CircuitState.OPEN:
            if self.last_failure_time is not None:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    logger.info(
                        f"Circuit breaker timeout expired ({self.timeout}s). "
                        "Entering HALF-OPEN state."
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    self.success_count = 0

        return self.state

    def _record_success(self) -> None:
        """Record successful function execution."""
        with self.lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(
                    f"Circuit breaker success in HALF-OPEN state "
                    f"({self.success_count}/{self.success_threshold})"
                )

                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker CLOSING (recovery successful)")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0

    def _record_failure(self) -> None:
        """Record failed function execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker failure in HALF-OPEN state. Reopening circuit.")
                self.state = CircuitState.OPEN
                self.success_count = 0

            elif self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker OPENING. Failure threshold exceeded "
                    f"({self.failure_count}/{self.failure_threshold})"
                )
                self.state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self.lock:
            logger.info("Manually resetting circuit breaker to CLOSED state")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


# Usage example: Multi-agent research system with circuit protection
class ResearchAgent:
    """Agent that searches external APIs (protected by circuit breaker)."""

    def __init__(self, breaker: CircuitBreaker):
        self.breaker = breaker

    def search_papers(self, query: str) -> list[dict]:
        """Search academic papers with circuit breaker protection."""
        def _search():
            # Simulate API call that might fail or be slow
            import requests
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 10},
                timeout=3.0
            )
            response.raise_for_status()
            return response.json()["data"]

        try:
            return self.breaker.call(_search)
        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker prevented API call: {str(e)}")
            # Fallback to cached results or empty list
            return []


class AnalysisAgent:
    """Agent that analyzes papers using LLM (protected by circuit breaker)."""

    def __init__(self, llm_client, breaker: CircuitBreaker):
        self.llm_client = llm_client
        self.breaker = breaker

    def analyze_papers(self, papers: list[dict]) -> dict:
        """Analyze papers with circuit breaker protection."""
        def _analyze():
            summaries = []
            for paper in papers:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": f"Summarize this paper in 2 sentences:\n{paper.get('title', 'No title')}\n{paper.get('abstract', 'No abstract')}"
                    }],
                    max_tokens=100,
                    timeout=5.0
                )
                summaries.append(response.choices[0].message.content)
            return {"summaries": summaries, "count": len(summaries)}

        try:
            return self.breaker.call(_analyze)
        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker prevented LLM calls: {str(e)}")
            # Gracefully degrade to paper titles only
            return {
                "summaries": [p.get("title", "No title") for p in papers],
                "count": len(papers),
                "degraded": True
            }


# Multi-agent orchestration with circuit breaker protection
def research_pipeline_with_circuit_protection(query: str) -> dict:
    """
    Execute multi-agent research pipeline with circuit breaker protection.

    Prevents cascading failures when external APIs or LLMs degrade.
    """
    # Create separate circuit breakers for each component
    search_breaker = CircuitBreaker(
        failure_threshold=3,
        success_threshold=2,
        timeout=30,
        latency_threshold=5.0
    )

    analysis_breaker = CircuitBreaker(
        failure_threshold=5,
        success_threshold=2,
        timeout=60,
        latency_threshold=10.0
    )

    # Initialize agents with circuit protection
    from openai import OpenAI
    llm_client = OpenAI()

    research_agent = ResearchAgent(search_breaker)
    analysis_agent = AnalysisAgent(llm_client, analysis_breaker)

    # Execute pipeline with failure isolation
    logger.info(f"Starting research pipeline for query: {query}")

    try:
        # Step 1: Search papers (protected by circuit breaker)
        papers = research_agent.search_papers(query)
        logger.info(f"Found {len(papers)} papers")

        if not papers:
            return {
                "status": "degraded",
                "message": "Search service unavailable. Circuit breaker active.",
                "results": []
            }

        # Step 2: Analyze papers (protected by circuit breaker)
        analysis = analysis_agent.analyze_papers(papers)

        return {
            "status": "success" if not analysis.get("degraded") else "degraded",
            "papers_found": len(papers),
            "summaries": analysis["summaries"],
            "degraded": analysis.get("degraded", False)
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "results": []
        }
