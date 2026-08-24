"""
Chapter 8.2B Lab — Circuit Breaker for Agent Tool Calls (starter)

You are completing the CircuitBreaker class used by TicketFlow's support agent to
guard its call to a flaky payment-verification tool. Read lab.md before touching
this file — it explains the scenario and walks you through the CLOSED state.

Your job: implement the OPEN and HALF_OPEN transitions marked with TODO below.
Do not change the public method signatures (call, state, _record_success,
_record_failure) — test_lab.py depends on them.
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
            # TODO 1: If enough time has passed since the trip (self._opened_at),
            # transition to HALF_OPEN and allow exactly one trial call through
            # by delegating to self._call_half_open(fn). Otherwise, reject the
            # call immediately by raising CircuitBreakerOpenError with a message
            # that includes how many seconds remain until recovery is attempted.
            raise NotImplementedError("TODO 1: implement OPEN-state handling")

        # self._state == CircuitState.HALF_OPEN
        # TODO 2: A trial call may already be in flight (another caller is
        # mid-call). Reject concurrent callers with CircuitBreakerOpenError
        # while a trial is in flight; otherwise run the trial via
        # self._call_half_open(fn).
        raise NotImplementedError("TODO 2: implement HALF_OPEN-state handling")

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
        # TODO 3: Mark a trial as in flight, run fn(). On success, reset the
        # circuit fully to CLOSED (clear results, clear _opened_at) and return
        # the result. On failure, reopen the circuit (set _opened_at to now)
        # and re-raise. Either way, clear the in-flight flag before returning
        # or raising.
        raise NotImplementedError("TODO 3: implement the half-open trial call")

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
