# Hint 3 (Strong - Implementation Strategy)
# Code snippets showing HALF_OPEN and CLOSED state handling patterns

# HALF_OPEN state handling pattern:
"""
if self.state == CircuitState.HALF_OPEN:
    try:
        result = func(*args, **kwargs)
        # Success - increment counter and potentially close circuit
        self.success_count += 1
        if self.success_count >= self.config.success_threshold:
            self._transition_to_closed()
        return result
    except Exception as e:
        # Failure - immediately reopen circuit
        self._transition_to_open()
        raise
"""

# CLOSED state handling pattern:
"""
try:
    result = func(*args, **kwargs)
    self.request_history.append((datetime.now(), True))  # Success
    self._remove_old_requests()
    # Failure rate check happens with existing failures, not this success
    return result
except Exception as e:
    self.request_history.append((datetime.now(), False))  # Failure
    self.failure_count += 1
    self._remove_old_requests()

    if self._calculate_failure_rate() >= self.config.failure_threshold:
        self._transition_to_open()

    raise  # Re-raise original exception
"""
