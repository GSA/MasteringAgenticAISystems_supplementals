# Hint 2 (Moderate - Detailed Structure)

def call(self, func: Callable, *args, **kwargs) -> Any:
    # State 1: OPEN - Check timeout and potentially transition to HALF_OPEN
    if self.state == CircuitState.OPEN:
        if datetime.now() - self.last_failure_time > self.config.timeout_duration:
            self._transition_to_half_open()
        else:
            raise CircuitBreakerError("Circuit breaker is OPEN")

    # State 2: HALF_OPEN - Execute test request and transition based on outcome
    if self.state == CircuitState.HALF_OPEN:
        # Execute and handle success/failure...
        pass

    # State 3: CLOSED - Normal execution with failure tracking
    if self.state == CircuitState.CLOSED:
        # Execute, track outcome, check failure threshold...
        pass


# Failure rate calculation helper
def _calculate_failure_rate(self) -> float:
    """Calculate failure rate from request_history."""
    if not self.request_history:
        return 0.0
    failures = sum(1 for timestamp, success in self.request_history if not success)
    return failures / len(self.request_history)
