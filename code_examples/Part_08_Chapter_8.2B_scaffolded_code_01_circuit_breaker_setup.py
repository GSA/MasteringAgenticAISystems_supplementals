# Guided Exercise: Circuit Breaker Pattern Implementation
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
from typing import Callable, Any, Tuple

class CircuitState(Enum):
    """Circuit breaker operating states."""
    CLOSED = "closed"          # Normal operation - requests proceed
    OPEN = "open"              # Fast-fail mode - requests immediately rejected
    HALF_OPEN = "half_open"    # Recovery testing - single probe request allowed

@dataclass
class CircuitBreakerConfig:
    """Configuration parameters for circuit breaker behavior."""
    failure_threshold: float = 0.5           # Open circuit after this fraction of failures
    window_size: timedelta = timedelta(seconds=10)     # Sliding window for failure calculation
    timeout_duration: timedelta = timedelta(seconds=30)  # Duration to stay OPEN before testing recovery
    success_threshold: int = 2               # Consecutive successes in HALF_OPEN to close circuit

class CircuitBreaker:
    """Circuit breaker implementing three-state failure protection."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque()  # Stores (timestamp, success) tuples for sliding window

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker protection.

        TODO: Implement circuit breaker logic following these requirements:

        1. If state is OPEN:
           - Check if timeout_duration has elapsed since last_failure_time
           - If timeout expired: transition to HALF_OPEN state
           - If timeout not expired: raise CircuitBreakerError immediately (fast-fail)

        2. If state is HALF_OPEN:
           - Allow ONE request to execute (this is the test probe)
           - If request succeeds: increment success_count
           - If success_count >= success_threshold: transition to CLOSED
           - If request fails: transition back to OPEN

        3. If state is CLOSED:
           - Execute request normally
           - Record outcome (timestamp, success) in request_history
           - Clean old entries from request_history (outside sliding window)
           - Calculate failure rate over sliding window
           - If failure_rate >= failure_threshold: transition to OPEN

        Args:
            func: Callable to execute with circuit breaker protection
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result of func execution

        Raises:
            CircuitBreakerError: When circuit is OPEN and fast-failing
            Original exception: When func raises during execution
        """
        pass  # Your implementation here

    def _remove_old_requests(self):
        """Remove request records outside the sliding window."""
        pass  # TODO: Implement sliding window cleanup

    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate from request_history."""
        pass  # TODO: Implement failure rate calculation

    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        pass  # TODO: Implement state transition

    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        pass  # TODO: Implement state transition

    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        pass  # TODO: Implement state transition

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is OPEN and request is fast-failed."""
    pass
