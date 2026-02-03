# Token Tracking Infrastructure
import tiktoken
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class TokenUsage:
    """Records token consumption and cost for individual requests"""
    request_id: str
    timestamp: datetime
    feature: str
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: float

class TokenTracker:
    """Tracks token usage with accurate cost calculation including caching"""

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000}
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens using model-specific tokenizer for accuracy"""
        return len(self.tokenizer.encode(text))

    def track_request(self,
                     request_id: str,
                     feature: str,
                     user_id: str,
                     model: str,
                     prompt: str,
                     completion: str,
                     cached_tokens: int = 0,
                     latency_ms: float = 0) -> TokenUsage:
        """Track comprehensive token usage with caching support"""
        # Calculate token counts
        input_tokens = self.count_tokens(prompt) - cached_tokens
        output_tokens = self.count_tokens(completion)

        # Calculate cost with caching discount
        input_cost = input_tokens * self.pricing[model]["input"]
        cached_cost = cached_tokens * self.pricing[model]["input"] * 0.5  # 50% discount
        output_cost = output_tokens * self.pricing[model]["output"]
        total_cost = input_cost + cached_cost + output_cost

        usage = TokenUsage(
            request_id=request_id,
            timestamp=datetime.now(),
            feature=feature,
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=total_cost,
            latency_ms=latency_ms
        )

        # Store usage data for analysis
        self.store_usage(usage)

        # Update real-time cost metrics
        self.update_metrics(usage)

        return usage

    def update_metrics(self, usage: TokenUsage):
        """Update Prometheus monitoring metrics"""
        # Counter: Total tokens by model and type
        token_counter.labels(model=usage.model, type="input").inc(usage.input_tokens)
        token_counter.labels(model=usage.model, type="output").inc(usage.output_tokens)

        # Counter: Cumulative cost by feature
        cost_counter.labels(feature=usage.feature).inc(usage.cost_usd)

        # Histogram: Cost distribution per request
        cost_histogram.labels(feature=usage.feature).observe(usage.cost_usd)

        # Gauge: Current hourly burn rate
        hourly_cost_gauge.labels(feature=usage.feature).set(
            self.calculate_hourly_burn_rate(usage.feature)
        )
