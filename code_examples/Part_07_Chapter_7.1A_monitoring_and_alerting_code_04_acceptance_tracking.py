# Track acceptance rate degradation in production
from prometheus_client import Histogram

acceptance_histogram = Histogram(
    'speculative_decode_acceptance_rate',
    'Acceptance rate distribution',
    buckets=[0.2, 0.4, 0.6, 0.8, 0.95]
)

def track_acceptance(accepted_tokens, total_speculated):
    acceptance_rate = accepted_tokens / total_speculated
    acceptance_histogram.observe(acceptance_rate)

    if acceptance_rate < 0.5:
        alert("Acceptance rate degraded below threshold")
