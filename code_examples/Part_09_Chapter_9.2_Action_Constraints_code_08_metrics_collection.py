from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Token bucket rate limiting to control operation velocity"""

    def __init__(self, tokens_per_minute: int = 60, burst_size: int = 10):
        self.tokens_per_minute = tokens_per_minute
        self.burst_size = burst_size
        self.buckets = defaultdict(lambda: {
            'tokens': burst_size,
            'last_update': datetime.utcnow()
        })

    def allow_request(self, user_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        bucket = self.buckets[user_id]
        now = datetime.utcnow()

        # Refill tokens based on elapsed time
        time_elapsed = (now - bucket['last_update']).total_seconds()
        tokens_to_add = (time_elapsed / 60.0) * self.tokens_per_minute
        bucket['tokens'] = min(
            self.burst_size,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_update'] = now

        # Check if sufficient tokens available
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        else:
            return False

# Usage: Limit agent to 60 operations per minute with burst capacity
limiter = RateLimiter(tokens_per_minute=60, burst_size=10)
if not limiter.allow_request(user_id='agent_123'):
    raise RateLimitError("Rate limit exceeded - operation velocity too high")