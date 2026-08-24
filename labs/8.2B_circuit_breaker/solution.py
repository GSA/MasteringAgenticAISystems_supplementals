"""
Chapter 8.2B Lab — Circuit Breaker for Agent Tool Calls (solution)

Complete implementation. If you're stuck on a TODO in starter.py, read the
matching numbered section here — but try Hints 1-3 in lab.md first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""


@dataclass
class CircuitBreaker:
    """
    A three-state circuit breaker.

    CLOSED: calls pass through normally. Failures are counted in a rolling
        window of `window_size` calls. If the failure rate within that window
        reaches `failure_threshold`, the circuit trips to OPEN.
    OPEN: calls are rejected immediately (CircuitBreakerOpenError), without
        touching the wrapped function, until `recovery_timeout` seconds have
        elapsed since the trip. This is what protects a struggling downstream
        service from retry storms.
    HALF_OPEN: exactly one trial call is allowed through. If it succeeds, the
        circuit resets to CLOSED. If it fails, the circuit reopens and the
        recovery timer restarts.
    """

    failure_threshold: float = 0.5
    window_size: int = 10
    recovery_timeout: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _results: list[bool] = field(default_factory=list, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_trial_in_flight: bool = field(default=False, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        """Run fn() through the circuit breaker and return its result."""
        if self._state == CircuitState.CLOSED:
            return self._call_closed(fn)

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return self._call_half_open(fn)
            remaining = self.recovery_timeout - elapsed
            raise CircuitBreakerOpenError(
                f"circuit open, retry in {remaining:.1f}s"
            )

        # self._state == CircuitState.HALF_OPEN
        if self._half_open_trial_in_flight:
            raise CircuitBreakerOpenError("half-open trial already in flight")
        return self._call_half_open(fn)

    def _call_closed(self, fn: Callable[[], T]) -> T:
        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _call_half_open(self, fn: Callable[[], T]) -> T:
        self._half_open_trial_in_flight = True
        try:
            result = fn()
        except Exception:
            self._opened_at = time.monotonic()
            self._state = CircuitState.OPEN
            raise
        else:
            self._state = CircuitState.CLOSED
            self._results = []
            self._opened_at = None
            return result
        finally:
            self._half_open_trial_in_flight = False

    def _record_success(self) -> None:
        self._results.append(True)
        self._trim_window()

    def _record_failure(self) -> None:
        self._results.append(False)
        self._trim_window()
        if self._should_trip():
            self._trip()

    def _trim_window(self) -> None:
        if len(self._results) > self.window_size:
            self._results = self._results[-self.window_size :]

    def _should_trip(self) -> bool:
        if len(self._results) < self.window_size:
            return False
        failure_rate = self._results.count(False) / len(self._results)
        return failure_rate >= self.failure_threshold

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
