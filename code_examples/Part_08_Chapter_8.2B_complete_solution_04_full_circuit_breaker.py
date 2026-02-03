# Complete Circuit Breaker Implementation
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: float = 0.5
    window_size: timedelta = timedelta(seconds=10)
    timeout_duration: timedelta = timedelta(seconds=30)
    success_threshold: int = 2

class CircuitBreakerError(Exception):
    """Raised when circuit breaker fast-fails an OPEN circuit."""
    pass

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_history = deque()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        # Check OPEN state: Has timeout expired?
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > self.config.timeout_duration:
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Retry after "
                    f"{(self.last_failure_time + self.config.timeout_duration - datetime.now()).seconds}s"
                )

        # HALF_OPEN state: Execute test request
        if self.state == CircuitState.HALF_OPEN:
            try:
                result = func(*args, **kwargs)
                self.success_count += 1

                # After success_threshold consecutive successes, close circuit
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()

                return result
            except Exception as e:
                # Test request failed - reopen circuit
                self._transition_to_open()
                raise

        # CLOSED state: Normal execution with failure tracking
        if self.state == CircuitState.CLOSED:
            try:
                result = func(*args, **kwargs)
                # Record success
                self.request_history.append((datetime.now(), True))
                self._remove_old_requests()
                return result
            except Exception as e:
                # Record failure
                self.request_history.append((datetime.now(), False))
                self.failure_count += 1
                self._remove_old_requests()

                # Check if failure rate exceeds threshold
                failure_rate = self._calculate_failure_rate()
                if failure_rate >= self.config.failure_threshold:
                    self._transition_to_open()

                raise  # Re-raise original exception

    def _remove_old_requests(self):
        """Remove requests outside sliding window."""
        cutoff_time = datetime.now() - self.config.window_size

        # Remove from left (oldest) while timestamps are too old
        while self.request_history and self.request_history[0][0] < cutoff_time:
            self.request_history.popleft()

    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate from request_history."""
        if not self.request_history:
            return 0.0

        failures = sum(1 for timestamp, success in self.request_history if not success)
        return failures / len(self.request_history)

    def _transition_to_open(self):
        """Transition to OPEN state - begin fast-failing."""
        self.state = CircuitState.OPEN
        self.last_failure_time = datetime.now()
        print(f"Circuit breaker OPENED at {self.last_failure_time}")

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state - begin recovery testing."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        print("Circuit breaker entering HALF_OPEN (testing recovery)")

    def _transition_to_closed(self):
        """Transition to CLOSED state - resume normal operation."""
        self.state = CircuitState.CLOSED
        self.request_history.clear()
        self.failure_count = 0
        self.success_count = 0
        print("Circuit breaker CLOSED (recovery confirmed)")
