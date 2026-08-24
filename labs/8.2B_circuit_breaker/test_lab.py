"""
Chapter 8.2B Lab — self-check.

Run this yourself once you've filled in the TODOs in starter.py:

    python test_lab.py

All 7 tests should print "ok". Any FAIL or ERROR line names the TODO you still
need to finish — the test names map directly to TODO 1 / TODO 2 / TODO 3 in
starter.py. This is your success signal: you're done when this file passes
with no edits to itself.

(To check the reference solution instead of your own work, maintainers can run
`LAB_MODULE=solution python test_lab.py` — you don't need this as a learner.)
"""

from __future__ import annotations

import importlib
import os
import time
import unittest

MODULE_NAME = os.environ.get("LAB_MODULE", "starter")
mod = importlib.import_module(MODULE_NAME)

CircuitBreaker = mod.CircuitBreaker
CircuitState = mod.CircuitState
CircuitBreakerOpenError = mod.CircuitBreakerOpenError


def ok_call():
    return "ok"


def failing_call():
    raise RuntimeError("payment-verification tool is down")


class TestClosedState(unittest.TestCase):
    """Baseline behavior — provided for you, should already pass."""

    def test_closed_state_passes_calls_through(self):
        cb = CircuitBreaker(failure_threshold=0.5, window_size=4)
        self.assertEqual(cb.call(ok_call), "ok")
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_closed_state_trips_to_open_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=0.5, window_size=4)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                cb.call(failing_call)
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                cb.call(failing_call)
        self.assertEqual(cb.state, CircuitState.OPEN)


class TestOpenState(unittest.TestCase):
    """Exercises TODO 1."""

    def _tripped_breaker(self, recovery_timeout=0.1):
        cb = CircuitBreaker(
            failure_threshold=0.5, window_size=2, recovery_timeout=recovery_timeout
        )
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                cb.call(failing_call)
        self.assertEqual(cb.state, CircuitState.OPEN)
        return cb

    def test_open_state_rejects_calls_without_running_them(self):
        cb = self._tripped_breaker(recovery_timeout=30.0)
        calls_made = []

        def tracked_call():
            calls_made.append(1)
            return "ok"

        with self.assertRaises(CircuitBreakerOpenError):
            cb.call(tracked_call)
        self.assertEqual(calls_made, [], "OPEN must not invoke the wrapped function")

    def test_open_state_transitions_to_half_open_after_timeout(self):
        cb = self._tripped_breaker(recovery_timeout=0.1)
        time.sleep(0.15)
        result = cb.call(ok_call)
        self.assertEqual(result, "ok")
        self.assertEqual(cb.state, CircuitState.CLOSED)


class TestHalfOpenState(unittest.TestCase):
    """Exercises TODO 2 and TODO 3."""

    def _half_open_breaker(self, recovery_timeout=0.1):
        cb = CircuitBreaker(
            failure_threshold=0.5, window_size=2, recovery_timeout=recovery_timeout
        )
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                cb.call(failing_call)
        time.sleep(recovery_timeout + 0.05)
        return cb

    def test_half_open_success_resets_to_closed(self):
        cb = self._half_open_breaker()
        self.assertEqual(cb.call(ok_call), "ok")
        self.assertEqual(cb.state, CircuitState.CLOSED)
        # a fresh failure window — should take window_size failures to trip again,
        # proving the failure history was actually cleared on reset
        with self.assertRaises(RuntimeError):
            cb.call(failing_call)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_half_open_failure_reopens_circuit(self):
        cb = self._half_open_breaker()
        with self.assertRaises(RuntimeError):
            cb.call(failing_call)
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_half_open_rejects_concurrent_trial(self):
        cb = self._half_open_breaker()
        # Force the breaker into HALF_OPEN with a trial already in flight,
        # simulating a second caller arriving while the first trial call is
        # still running (call() only makes this transition lazily on its own,
        # so we set it directly to isolate the concurrency guard from timing).
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_trial_in_flight = True
        with self.assertRaises(CircuitBreakerOpenError):
            cb.call(ok_call)


if __name__ == "__main__":
    unittest.main(verbosity=2)
