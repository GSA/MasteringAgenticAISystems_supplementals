---
title: "Build a Circuit Breaker for a Flaky Downstream Tool"
chapter: "8.2B"
knowledge_items:
  - "8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions"
  - "8.2B: failure-threshold and sliding-window tuning for trip/reset behavior"
prerequisites:
  - "Comfortable with Python classes, exceptions, and Enum"
  - "Chapter 8.1 (reliability fundamentals) read first"
  - "Python 3.10+"
learning_objectives:
  - "Explain why a naive retry loop around a failing tool call makes an agent's reliability problem worse, not better"
  - "Implement the three-state circuit breaker pattern (CLOSED to OPEN to HALF_OPEN) from scratch"
  - "Reason about the tradeoff between failure_threshold/window_size (breaker sensitivity) and recovery_timeout (recovery caution)"
  - "Verify state-transition logic with a small, deterministic test suite instead of eyeballing behavior"
estimated_lab_time: "60-90 minutes"
frameworks:
  - "none — Python 3.10+ standard library only"
maturity: "draft"
related_reference_lab: "labs/8.2B_circuit_breaker/lab.md"
---

# Lab 8.2B — Circuit Breaker for Agent Tool Calls

## Objectives

By the end of this lab you will be able to:

1. Explain why a naive retry loop around a failing tool call makes an agent's
   reliability problem worse instead of better.
2. Implement the three-state circuit breaker pattern (CLOSED → OPEN → HALF_OPEN)
   from scratch.
3. Reason about the tradeoff between `failure_threshold`/`window_size` (how
   sensitive the breaker is) and `recovery_timeout` (how long it stays cautious).
4. Verify state-transition logic with a small, deterministic test suite instead of
   eyeballing behavior.

## Scenario

TicketFlow is a support agent that calls a `verify_payment` tool as part of
refund requests. During a routine deploy, the payment-verification service starts
failing intermittently — not down, just slow and flaky. TicketFlow's existing code
retries every failed call up to 3 times.

Under load, that retry logic turns a partial outage into a full one: each of
TicketFlow's 1,000 concurrent sessions can generate up to 3 extra requests against
an already-struggling service, and response times climb from 200ms to several
seconds for every user — including the ones whose calls would have succeeded on
the first try. This is the scenario the existing `Part_08_Chapter_8.2B_Labs.md`
example file describes; this lab has you build the fix from scratch rather than
read about it.

A **circuit breaker** wraps the risky call and stops making it once failures cross
a threshold, instead of hammering it — the mechanism you'll implement here.

## Setup

No installation needed beyond Python itself — this lab has zero third-party
dependencies (see `requirements.txt`).

```bash
cd labs/8.2B_circuit_breaker
python3 --version   # confirm 3.10 or newer
python3 test_lab.py # run the self-check now, before changing anything
```

## We Do — the CLOSED state

Open `starter.py`. The `CircuitBreaker` class is a `dataclass` with three states
(`CircuitState.CLOSED`, `OPEN`, `HALF_OPEN`) and a rolling window of the last
`window_size` call outcomes.

The CLOSED-state path is already implemented for you — read `_call_closed`,
`_record_success`, `_record_failure`, `_should_trip`, and `_trip` before writing
any code. Walk through it:

- `_call_closed` runs `fn()`. On success it records a success; on exception it
  records a failure and re-raises (the caller still sees the real error).
- `_record_failure` appends to `_results`, trims the window to the last
  `window_size` entries, and checks `_should_trip`.
- `_should_trip` only evaluates once the window is full (`window_size` calls have
  happened), then trips if the failure rate within that window is at or above
  `failure_threshold`.
- `_trip` flips `_state` to `OPEN` and records `_opened_at` — this timestamp is
  what OPEN-state recovery will check against, which you're about to implement.

You already ran the self-check once in Setup. In that run, `TestClosedState`'s two
tests passed and the rest failed with `NotImplementedError` — that's expected.
Those failures are your worklist for the rest of this lab.

## You Do — OPEN and HALF_OPEN

Implement the three `TODO`s in `starter.py`'s `call()` and `_call_half_open()`
methods. Don't change any method signature — `test_lab.py` calls them directly.

**TODO 1 (OPEN state):** When the circuit is OPEN, calls must be rejected
*without* invoking the wrapped function — that's the entire point of the
breaker, so a struggling downstream service isn't hit again. But if enough time
has passed (`self.recovery_timeout` seconds since `self._opened_at`), the
breaker should give the downstream service one trial call instead of staying
shut forever.

**TODO 2 (HALF_OPEN dispatch):** In `HALF_OPEN`, only one trial call is allowed
through at a time. If a trial is already running (`_half_open_trial_in_flight`
is `True`), a second concurrent caller must be rejected, not queued or run.

**TODO 3 (the trial call itself, `_call_half_open`):** Run the trial. On
success, the breaker should reset completely — back to `CLOSED`, with the
failure history cleared, so a chapter's worth of past failures doesn't linger
and immediately re-trip it. On failure, the breaker should reopen (and restart
the recovery clock) rather than stay stuck in `HALF_OPEN`.

### Hints

**Hint 1 (gentle):** Look at `_trip()` — it's a two-line pattern
(`self._state = ...`, `self._opened_at = ...`) you'll reuse in more than one
place.

**Hint 2 (moderate):** For TODO 1, compute
`elapsed = time.monotonic() - self._opened_at`. Compare it to
`self.recovery_timeout`. If recovery hasn't elapsed yet, raise
`CircuitBreakerOpenError` with a message that includes how many seconds remain
— that number is genuinely useful to whoever's debugging a live incident.

**Hint 3 (strong):** `_call_half_open` needs a `try`/`except`/`else`/`finally`
shape: set the in-flight flag before the call, clear it in `finally` no matter
what happens, and put the reset-to-CLOSED logic in `else` (only runs if no
exception was raised) and the reopen logic in `except`.

**Solution:** `solution.py` has the complete implementation if you want to check
your approach or you're well and truly stuck. Try to get your own version
passing first — the point of this lab is building the state machine, not reading
one.

## Self-check

```
python3 test_lab.py
```

All 7 tests should print `ok`. You're done when this file passes without you
having edited it. If a test fails, its name tells you which TODO to revisit:
`TestOpenState` tests exercise TODO 1; `test_half_open_rejects_concurrent_trial`
exercises TODO 2; the other two `TestHalfOpenState` tests exercise TODO 3.

## Independent challenge (optional, no solution provided)

The breaker above only tracks one downstream dependency. TicketFlow actually
calls three tools (`verify_payment`, `check_inventory`, `send_notification`).
Extend `CircuitBreaker` (or wrap it) so a caller can hold one breaker instance
per tool name, with per-tool `failure_threshold`/`window_size`/`recovery_timeout`
configuration, and a single method to check whether *any* tool is currently
OPEN (useful for a health-check endpoint). There's no solution file for this —
decide for yourself what "done" looks like, informed by the objectives above.

## Key Points

- A circuit breaker's job is to stop calling a failing dependency, not to make
  the call succeed — pair it with a fallback or a user-facing degraded mode at
  the call site, which is a separate concern from what this lab implements.
- The CLOSED → OPEN transition is about a *rate* of failure over a window, not
  any single failure — that's what keeps a breaker from tripping on one
  unlucky timeout.
- HALF_OPEN exists so recovery is a single controlled probe, not a stampede of
  every caller retrying the instant `recovery_timeout` elapses.
- Rejecting fast (raising immediately in OPEN) is the entire value proposition
  under load — a slow rejection is barely better than the outage it's meant to
  prevent.

## Citation: Objectives → Practice → Check

| Objective | Chapter knowledge item | Practiced in | Verified by |
|---|---|---|---|
| Explain why naive retries amplify a downstream failure | 8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions | Scenario | N/A — conceptual; carried into the design read in We Do |
| Implement OPEN-state fast-fail rejection and timed transition to HALF_OPEN | 8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions | You Do, TODO 1 | `test_lab.py::TestOpenState` |
| Implement HALF_OPEN single-trial dispatch, reset-to-CLOSED on success, reopen on failure | 8.2B: failure-threshold and sliding-window tuning for trip/reset behavior | You Do, TODOs 2-3 | `test_lab.py::TestHalfOpenState` |
| Verify state-transition logic with a deterministic test suite | 8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions | Self-Check | Running `python3 test_lab.py` and reading its pass/fail output directly |
