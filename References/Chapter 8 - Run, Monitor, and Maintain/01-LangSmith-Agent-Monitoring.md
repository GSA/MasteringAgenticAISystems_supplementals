# LangSmith: Agent Monitoring and Observability

**Source:** https://www.langchain.com/langsmith

**Platform:** LangSmith by LangChain
**Focus:** Complete observability for agent systems
**Components:** Tracing, monitoring, evaluation, insights

---

## Overview

LangSmith provides complete visibility into agent behavior through **tracing, real-time monitoring, alerting, and insights** into agent execution patterns and failure modes.

## Core Components

### 1. Tracing

**What It Does:**
Records complete execution trace from input to output

**Captured Information:**
- Individual steps (runs)
- Model calls
- Tool invocations
- Retriever operations
- Sub-chains
- Time measurements
- Token counts
- Costs

**Benefit:** Debug agent behavior step-by-step

### 2. Monitoring

**Real-Time Metrics:**
- Cost tracking (by run, model, agent)
- Latency measurements (p50, p95, p99)
- Token usage
- Response quality metrics
- Error rates

**Live Dashboards:** Track key metrics continuously

### 3. Evaluation

**Automatic Evaluation:**
- LLM-as-a-judge scoring
- Custom rubric testing
- Multi-turn conversation evaluation
- Tool calling correctness assessment

**Human Evaluation:**
- Mark traces for manual review
- Collaborate on feedback
- Build evaluation datasets

### 4. Insights Agent

**Analysis:**
Discovers common patterns, behaviors, and failure modes

**Provides:**
- Usage patterns
- Common agent paths
- Error clustering
- Performance segments

---

## Implementation Pattern

```python
from langsmith import Client
from langchain.agents import AgentExecutor

# Initialize LangSmith tracing
from langsmith import set_workspace
set_workspace("my-workspace")

# Create and run agent (automatically traced)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools
)

# Execute (automatically sent to LangSmith)
result = agent_executor.invoke({
    "input": "User question here"
})
```

## Tracing Structure

### Run Hierarchy

```
Conversation (highest level)
    ├─ Agent Decision
    │   ├─ LLM Call
    │   ├─ Tool 1 Invocation
    │   │   ├─ Tool Processing
    │   │   └─ Result
    │   └─ Tool 2 Invocation
    ├─ Agent Decision (second step)
    └─ Final Response
```

### Information Captured

- **Inputs/Outputs:** Full conversation
- **Duration:** Latency per step
- **Tokens:** Model consumption
- **Cost:** Compute cost per run
- **Errors:** Exception stack traces
- **Metadata:** Custom tags and attributes

---

## Evaluation Types

### Type 1: Automated LLM Evaluation

```python
from langsmith import evaluate

# Define evaluation criteria
criteria = """
Evaluate if the agent response correctly answers the user question.
Rate on scale 1-5.
"""

# Evaluate traces automatically
results = evaluate(
    traces,
    evaluators=[
        ("answer_quality", lambda x: {
            "score": llm_evaluate(x, criteria)
        })
    ]
)
```

### Type 2: Multi-Turn Evaluation

Measure if agent accomplished goal across entire interaction

**Captures:**
- Initial intent
- Agent's path to solution
- Success metrics
- Effort (steps taken)

### Type 3: Custom Metrics

Test custom rubrics for domain-specific evaluation

---

## Monitoring Setup

### Key Metrics to Track

**Operational:**
- Availability (uptime)
- Latency (response time)
- Error rate
- Queue depth

**Financial:**
- Cost per request
- Cost per successful request
- Cost by agent
- Cost by model

**Quality:**
- Accuracy metrics
- User satisfaction
- Goal achievement rate
- Tool call success rate

### Alert Configuration

```python
# Alert when error rate exceeds threshold
alert_config = {
    "metric": "error_rate",
    "threshold": 0.05,
    "window": "5m",
    "severity": "high"
}

# Alert on cost anomalies
alert_config = {
    "metric": "cost_per_request",
    "type": "anomaly_detection",
    "severity": "medium"
}
```

---

## Analysis with Insights Agent

### Discovers

**Usage Patterns:**
- Common agent flows
- Frequently asked questions
- Popular tool combinations

**Behavioral Insights:**
- Success rates by scenario
- Failure patterns
- Performance bottlenecks

**Improvement Opportunities:**
- Low-performing steps
- High-cost operations
- Error clustering

---

## Integration Patterns

### Pattern 1: Development

```python
# Enable debug mode for development
from langsmith import enable_debugging
enable_debugging()

# Run agent locally
result = agent.invoke(input)

# All traces sent to LangSmith
# Review in dashboard for debugging
```

### Pattern 2: Production Monitoring

```python
# Production with persistent workspace
from langsmith import Client

client = Client(api_key="your-key")

# Run agent
result = agent.invoke(input)

# Metrics automatically collected
# Live dashboard shows performance
```

### Pattern 3: Evaluation Pipeline

```python
# Test suite with automatic evaluation
test_cases = [
    {"input": "Q1", "expected": "Expected answer 1"},
    {"input": "Q2", "expected": "Expected answer 2"},
]

# Run evaluations
for test in test_cases:
    result = agent.invoke({"input": test["input"]})
    evaluate(result, test["expected"])

# Results in LangSmith
```

---

## Best Practices

### For Development

- [ ] Enable tracing in dev environment
- [ ] Review traces regularly
- [ ] Identify slow paths
- [ ] Mark promising solutions
- [ ] Build evaluation dataset

### For Production

- [ ] Configure key metrics
- [ ] Set up alerting
- [ ] Monitor latency & cost
- [ ] Review error logs
- [ ] Periodic manual review

### For Evaluation

- [ ] Automatic evaluation on all traces
- [ ] Mark traces for improvement
- [ ] Build evaluation dataset from samples
- [ ] A/B test improvements

---

## Common Queries

**Find slow agent executions:**
```python
slow_traces = client.list_runs(
    session_name="my-agent",
    latency_ms=(1000, 5000)
)
```

**Find error patterns:**
```python
error_traces = client.list_runs(
    session_name="my-agent",
    error=True
)
```

**Find expensive runs:**
```python
expensive = client.list_runs(
    session_name="my-agent",
    sort_by_cost_desc=True
)[:10]
```

---

## References

- **Evaluation Framework:** Related to Chapter 3 evaluation content
- **Agent Architecture:** See Chapters 1-2
- **Monitoring Best Practices:** See Chapter 7/16

---

## Conclusion

LangSmith provides essential observability for agent systems, enabling debugging, monitoring, and continuous improvement through comprehensive tracing, real-time metrics, and automated evaluation. By leveraging LangSmith's insights, teams build higher-quality, more efficient agent systems.

For detailed LangSmith documentation: https://www.langchain.com/langsmith
