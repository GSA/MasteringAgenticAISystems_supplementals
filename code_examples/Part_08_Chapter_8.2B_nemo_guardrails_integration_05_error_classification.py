# NeMo Guardrails with Error Tracking Integration
from nemoguardrails import RailsConfig, LLMRails
from opentelemetry import trace
from prometheus_client import Counter

# Initialize OpenTelemetry tracer for distributed tracing
tracer = trace.get_tracer(__name__)

# Define separate Prometheus counters for safety vs infrastructure errors
safety_violations = Counter(
    'agent_safety_violations_total',
    'Count of NeMo Guardrails safety policy violations',
    ['violation_type', 'agent_id']
)

infrastructure_errors = Counter(
    'agent_infrastructure_errors_total',
    'Count of infrastructure execution failures',
    ['error_type', 'agent_id']
)

# Configure NeMo Guardrails from configuration directory
# Expects config/ to contain:
# - config.yml: Guardrails policies (PII detection, jailbreak prevention, factuality checking)
# - prompts.yml: System prompts and safety instructions
config = RailsConfig.from_path("config")
rails = LLMRails(config)

async def guarded_agent_call(user_input: str, agent_id: str = "customer_service"):
    """
    Execute agent request with NeMo Guardrails safety monitoring.

    This function demonstrates three-layer error handling:
    1. Guardrails execution (safety policy validation)
    2. Safety violation classification (blocked responses)
    3. Infrastructure failure classification (execution exceptions)

    Args:
        user_input: User query to process
        agent_id: Identifier for agent instance (used in metrics labels)

    Returns:
        dict: Response with content or error details

    Raises:
        Exception: Infrastructure failures (timeouts, capacity errors) propagated to caller
    """
    # Create OpenTelemetry span for distributed tracing
    with tracer.start_as_current_span("guarded_agent_execution") as span:
        span.set_attribute("agent.id", agent_id)
        span.set_attribute("input.length", len(user_input))

        try:
            # Execute request through NeMo Guardrails
            # Guardrails performs:
            # 1. Input validation (jailbreak detection, prompt injection checks)
            # 2. LLM invocation with safety-enhanced system prompts
            # 3. Output validation (PII detection, factuality checking, hallucination detection)
            response = await rails.generate_async(messages=[{
                "role": "user",
                "content": user_input
            }])

            # Check if guardrails BLOCKED the response (safety violation detected)
            if response.get("blocked", False):
                # This is a SAFETY failure, not infrastructure failure
                violation_type = response.get("block_reason", "unknown_violation")

                # Record safety violation in span attributes
                span.set_attribute("guardrail.blocked", True)
                span.set_attribute("guardrail.violation_type", violation_type)

                # Increment safety violation counter (separate from infrastructure errors)
                safety_violations.labels(
                    violation_type=violation_type,
                    agent_id=agent_id
                ).inc()

                # Return structured error response
                # Do NOT raise exception - this is expected behavior for adversarial inputs
                return {
                    "error": "Request blocked by safety guardrails",
                    "violation_type": violation_type,
                    "category": "safety_violation"
                }

            # Guardrails validation passed - return successful response
            span.set_attribute("guardrail.passed", True)
            span.set_attribute("response.length", len(response.get("content", "")))

            return {
                "content": response.get("content"),
                "category": "success"
            }

        except Exception as e:
            # This is an INFRASTRUCTURE failure (timeout, API error, capacity limit)
            error_type = type(e).__name__

            # Record infrastructure failure in span
            span.set_attribute("error.infrastructure", True)
            span.set_attribute("error.type", error_type)
            span.record_exception(e)

            # Increment infrastructure error counter (separate from safety violations)
            infrastructure_errors.labels(
                error_type=error_type,
                agent_id=agent_id
            ).inc()

            # Propagate exception to caller for retry logic / circuit breaker handling
            raise
