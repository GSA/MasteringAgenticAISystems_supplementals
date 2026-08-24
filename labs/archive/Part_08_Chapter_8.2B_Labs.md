## 8.2.4 "You Do" Guided Practice: Implementing Circuit Breakers

### The Cascading Failure Problem

Distributed systems fail in predictable ways. When a downstream service experiences degradation—whether an external API endpoint, database connection, or LLM provider—continuing to send requests creates a destructive cycle. Each failing request consumes client-side resources (connection pools, thread pools, memory), increases latency for end users, and amplifies load on the already-struggling downstream service. The result: cascading failures that propagate across your architecture, transforming a single unhealthy dependency into system-wide outages.

Circuit breakers prevent this cascade by implementing fast-fail behavior. When a downstream service exhibits failures exceeding a configured threshold, the circuit breaker "opens," immediately rejecting subsequent requests without attempting to contact the failing service. This serves two critical functions: first, it preserves client-side resources by avoiding doomed network calls; second, it reduces load on the downstream service, giving it time to recover. After a configured timeout period, the circuit breaker enters a "half-open" state, allowing a single test request to probe for recovery. If the test succeeds, normal operation resumes. If it fails, the circuit reopens for another timeout cycle.

Consider this production scenario: Your customer service agent invokes an external product recommendation API as part of processing user queries about purchase suggestions. During peak shopping hours, this API experiences intermittent failures with a 15% error rate. Without circuit breaker protection, your agent's retry logic amplifies the problem—each of 1,000 concurrent agent sessions retries failed requests up to 3 times, generating 3,000 additional requests to an already overwhelmed service. The API response time degrades from 200ms to 8 seconds, causing connection pool exhaustion in your agent infrastructure. What began as a 15% error rate in one dependency cascades into complete agent unavailability.

### The Guided Implementation Challenge

Your task: implement a circuit breaker pattern to protect against this cascading failure scenario. The circuit breaker must track failure rates over a sliding time window and implement three-state behavior (CLOSED, OPEN, HALF_OPEN) with automatic state transitions based on observed service health.

**Specific Requirements:**
- **Failure Threshold:** After 50% of requests fail within a 10-second sliding window, transition from CLOSED to OPEN state
- **Timeout Duration:** Remain in OPEN state for 30 seconds before attempting recovery
- **Recovery Protocol:** In HALF_OPEN state, allow a single test request; if successful, transition to CLOSED; if failed, return to OPEN
- **State Tracking:** Maintain counts of failures and successes with timestamps for sliding window calculation

This exercise uses progressive scaffolding. Start by examining the provided code skeleton, then attempt implementation using the gentle hints. If you need additional guidance, reveal the moderate hints. The strong hints provide detailed implementation strategy. After attempting the implementation yourself, review the complete solution to compare approaches.

### Scaffolded Code Starting Point

Please see code example Part_08_Chapter_8.2B_scaffolded_code_01_circuit_breaker_setup.py

### Progressive Hints for Implementation

**Hint 1 (Gentle - Start Here):**

The sliding window calculation is the core challenge. You need to track when requests occurred and whether they succeeded, then calculate the failure rate considering only requests within the `window_size` period. Consider using a `deque` (double-ended queue) to store tuples of `(timestamp, success_flag)`. Before calculating the failure rate, iterate through the deque from the left (oldest entries) and remove any requests with `timestamp < (now - window_size)`. Then count failures vs. total requests in the remaining entries.

For state transitions, remember that each state has specific entry/exit actions. When transitioning TO OPEN, record `last_failure_time = datetime.now()`. When transitioning FROM OPEN to HALF_OPEN, reset both `failure_count` and `success_count` to prepare for the recovery test.

**Hint 2 (Moderate - Detailed Structure):**

Please see code example Part_08_Chapter_8.2B_moderate_hints_02_call_method_structure.py for the detailed structure of the call() method and failure rate calculation patterns.

**Hint 3 (Strong - Implementation Strategy):**

Please see code example Part_08_Chapter_8.2B_strong_hints_03_half_open_and_closed_logic.py for detailed patterns for handling HALF_OPEN state test requests and CLOSED state failure tracking.

### Complete Solution with Explanation

After attempting the implementation yourself using the hints above, compare your approach to this complete solution. Pay particular attention to the state transition logic and sliding window calculation—these are the most common sources of implementation bugs.

Please see code example Part_08_Chapter_8.2B_complete_solution_04_full_circuit_breaker.py

**Key Implementation Details Explained:**

1. **Sliding Window Cleanup:** The `_remove_old_requests()` method uses `popleft()` to efficiently remove the oldest entries first. Since requests are appended in chronological order, all "too old" requests cluster at the left side of the deque. We remove until finding a request within the window, ensuring O(k) complexity where k = removed entries.

2. **State Transition Timing:** When transitioning TO OPEN, we immediately record `last_failure_time`. This timestamp serves as the anchor for timeout calculation. When checking timeout expiration in the OPEN state, we calculate `datetime.now() - self.last_failure_time > timeout_duration` BEFORE transitioning to HALF_OPEN, ensuring the full timeout period elapses.

3. **HALF_OPEN Success Counting:** The `success_threshold` parameter (default 2) prevents premature circuit closure after a single successful probe. Transient recovery might result in one successful request followed by continued failures. Requiring multiple consecutive successes provides higher confidence that the downstream service genuinely recovered.

4. **Exception Propagation:** The circuit breaker raises the original exception after recording failures, preserving stack traces and error context for upstream handlers. This differs from OPEN state behavior, where we raise `CircuitBreakerError` to signal fast-fail behavior rather than actual downstream failures.

5. **Failure Rate Calculation During Success:** When a request succeeds in CLOSED state, we still call `_calculate_failure_rate()` to check if the circuit should open. This might seem counterintuitive—why check after success? The answer: the sliding window might contain many failures from the recent past. A single success doesn't negate a 60% failure rate over the last 10 seconds.

### Testing the Circuit Breaker Implementation

To validate your implementation, test these scenarios demonstrating state transitions:

**Test 1: Circuit Opens After Threshold Exceeded**
```python
import time

# Simulated failing service
def failing_api_call():
    """Fails 60% of the time."""
    import random
    if random.random() < 0.6:
        raise Exception("API timeout")
    return "Success"

config = CircuitBreakerConfig(
    failure_threshold=0.5,
    window_size=timedelta(seconds=10),
    timeout_duration=timedelta(seconds=5)
)
cb = CircuitBreaker(config)

# Generate requests until circuit opens
for i in range(20):
    try:
        result = cb.call(failing_api_call)
        print(f"Request {i}: {result}")
    except CircuitBreakerError as e:
        print(f"Request {i}: Circuit OPEN (fast-failed)")
        break
    except Exception as e:
        print(f"Request {i}: Failed ({e})")

    time.sleep(0.5)  # 500ms between requests

# Expected output: After ~10 requests (60% failure rate), circuit opens
# Subsequent requests immediately raise CircuitBreakerError
```

**Test 2: Circuit Recovers After Timeout**
```python
# After circuit opens, wait for timeout and observe HALF_OPEN transition
print("\nWaiting 5 seconds for timeout...")
time.sleep(6)

# Simulated recovered service (now succeeds)
def recovered_api_call():
    return "Success - service recovered"

# First request after timeout should enter HALF_OPEN
result = cb.call(recovered_api_call)
print(f"After timeout: {result} (state: {cb.state})")

# Second successful request should close circuit
result = cb.call(recovered_api_call)
print(f"Second success: {result} (state: {cb.state})")

# Expected output:
# "Circuit breaker entering HALF_OPEN (testing recovery)"
# After timeout: Success - service recovered (state: HALF_OPEN)
# "Circuit breaker CLOSED (recovery confirmed)"
# Second success: Success - service recovered (state: CLOSED)
```

**Test 3: Circuit Reopens if Test Request Fails**
```python
# Reset to OPEN state
cb._transition_to_open()
time.sleep(6)  # Wait for timeout

# Attempt recovery with still-failing service
def still_failing_api():
    raise Exception("Service still unhealthy")

try:
    cb.call(still_failing_api)
except Exception:
    print(f"Test request failed - circuit state: {cb.state}")

# Expected output: Circuit transitions OPEN → HALF_OPEN → OPEN
# "Circuit breaker entering HALF_OPEN (testing recovery)"
# "Circuit breaker OPENED at [timestamp]"
# Test request failed - circuit state: OPEN
```

These test scenarios validate the three critical behaviors: opening after threshold violations, recovering after timeout with successful probes, and reopening if recovery tests fail. When implementing circuit breakers in production, add metrics tracking state durations (time spent OPEN, HALF_OPEN transitions per hour) and alert on circuits that remain OPEN beyond expected recovery windows.

---