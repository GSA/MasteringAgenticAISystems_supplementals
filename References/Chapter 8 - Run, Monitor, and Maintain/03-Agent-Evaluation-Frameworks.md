# Agent Evaluation Frameworks and Metrics

**Source:** Industry best practices for agent evaluation and NVIDIA documentation

**Focus:** Comprehensive evaluation approaches for deployed agent systems
**Scope:** Metrics, benchmarks, evaluation methodologies, automation

---

## Agent-Specific Evaluation Metrics

### Task Completion Metrics

**Success Rate**
- Percentage of agent runs that successfully completed the goal
- Target: >95% for production systems

**Steps to Completion**
- Number of steps required to reach goal
- Indicates efficiency and prompt quality
- Track trends for optimization

**Retry Rate**
- How often agent retries failed operations
- High retry rate indicates instability or unclear instructions

### Quality Metrics

**Accuracy of Agent Decisions**
- Does agent make correct choices given context?
- Measured against ground truth when available
- Per-decision category tracking

**Tool Call Correctness**
- Are tool invocations appropriate for the task?
- Do parameters match the context?
- Rate of invalid tool calls

**Hallucination Rate**
- How often does agent invent information?
- Critical for RAG systems
- Measured against retrieval source documents

### User Experience Metrics

**User Satisfaction Score**
- Explicit rating (1-5) or thumbs up/down
- Track satisfaction by task type
- Identify problem areas

**Helpful Response Rate**
- Percentage of responses users found useful
- Different from accuracy - user perception matters

**Query Resolution Rate**
- Fraction of queries fully resolved without escalation
- Complement to success rate

### Efficiency Metrics

**Response Time Distribution**
- p50, p95, p99 latency
- Budget-aware: Response time vs. response quality
- Flag: p95 > SLA threshold

**Token Efficiency**
- Tokens generated per successful task
- Identify verbose reasoning paths
- Compare against baseline

**Cost per Task**
- Total cost (tokens × price) per completed task
- Most important for business operations
- Target cost reduction over time

---

## Evaluation Methodologies

### Method 1: Automated LLM-as-Judge Evaluation

```python
from langsmith import evaluate

# Define evaluation criteria
criteria = """
Evaluate if the agent response correctly answers the user's question.
- Accuracy: Is the information correct?
- Completeness: Does it address all aspects?
- Clarity: Is the answer well-explained?
Rate on scale 1-5.
"""

# Run evaluation automatically
evaluator = LLMEvaluator(
    criteria=criteria,
    model="gpt-4"
)

results = evaluate(
    dataset=test_traces,
    evaluators=[(
        "response_quality",
        evaluator
    )]
)
```

**Advantages:**
- Scalable - evaluate thousands of traces
- Consistent - same rubric applied uniformly
- Fast - evaluations run in minutes

**Limitations:**
- Depends on judge model quality
- May not catch subtle errors
- Cannot evaluate subjective quality well

### Method 2: Human-in-the-Loop Evaluation

**Sampling Strategy:**
```
Sample 5-10% of production traces for review
- Random sampling for baseline quality
- Error sampling for problem investigation
- Edge case sampling for boundary conditions
```

**Review Scorecard:**
```
For each trace:
[ ] Task completed successfully
[ ] Tool calls were appropriate
[ ] No hallucinations detected
[ ] Response was clear and helpful
[ ] Any issues found: [description]
```

**When to Use:**
- Validate LLM-as-judge results
- Catch systemic issues humans notice but LLMs miss
- Build training data for fine-tuning

### Method 3: Multi-Turn Conversation Evaluation

**Measure:**
- Initial problem statement
- Agent's reasoning across multiple steps
- Final outcome vs. goal
- Success metrics

**Example:**
```
User: "Help me optimize my database queries"

Agent Step 1: Asks about current performance
Agent Step 2: Requests schema information
Agent Step 3: Analyzes queries
Agent Step 4: Provides optimization recommendations

Success: User confirms recommendations improved performance
```

### Method 4: Regression Testing

**Approach:**
- Build dataset of known good/bad traces
- Re-run after model updates
- Alert if performance degrades

**Test Categories:**
```
Core functionality tests
- Basic tasks that must always work
- Test: Success rate >99%

Edge case tests
- Boundary conditions and complex queries
- Test: Success rate >90%

Regression tests
- Tasks from historical failures
- Test: All previously fixed issues remain fixed
```

---

## Benchmark Datasets

### Public Benchmarks

**ToolLLaMA**
- Tool-using capability benchmark
- Covers 16 different tool categories
- Success rate: 50-90% depending on model

**API Bank**
- Banking/finance APIs
- Tests real-world API usage
- ~1000 test cases

**ReAct Benchmark**
- Multi-step reasoning tasks
- Combines retrieval with reasoning
- Standard baseline: 70-85%

### Custom Benchmarks

**Building Enterprise Benchmarks:**

1. **Collect representative tasks** from production
   - Sample high-value queries
   - Capture diverse use cases
   - Include edge cases

2. **Get ground truth labels**
   - Domain expert validation
   - Reference implementations
   - User confirmation

3. **Create test dataset**
   - 50-100 test cases minimum
   - Stratify by difficulty
   - Include negative cases

---

## Real-Time Evaluation in Production

### Continuous Quality Monitoring

```python
# After each agent execution
def evaluate_execution(agent_result):
    metrics = {
        "success": agent_result.success,
        "steps_taken": len(agent_result.steps),
        "hallucination_detected": check_hallucination(agent_result),
        "execution_cost": calculate_cost(agent_result),
        "latency_ms": agent_result.latency_ms
    }

    # Compare against baseline
    baseline = get_baseline_metrics()
    if metrics["success_rate"] < baseline * 0.95:
        alert("Success rate degradation detected")

    return metrics
```

**Metrics Tracked:**
- Success/failure for every execution
- Cost tracking for trend analysis
- Latency percentiles
- Error categorization

**Alerting Thresholds:**
- Success rate drops >5%
- Cost per request doubles
- Latency p95 exceeds SLA
- Hallucination rate >2%

---

## Evaluation Dashboards

### Executive Dashboard

Shows:
- Overall system health (green/yellow/red)
- Success rate trend
- Cost trend
- Key business metrics
- Comparison vs. baseline

### Quality Dashboard

Shows:
- Success rate by task type
- Hallucination rate
- User satisfaction scores
- Error distribution
- Improvement areas

### Optimization Dashboard

Shows:
- Cost trends (per request, per token)
- Performance optimization opportunities
- Efficiency metrics
- Resource utilization
- Areas for improvement

---

## A/B Testing Agents

### Testing Framework

```python
# Route requests to two agents
def route_request(request):
    if random.random() < 0.5:
        result = agent_v1.execute(request)
        log_result(result, version="v1")
    else:
        result = agent_v2.execute(request)
        log_result(result, version="v2")
    return result

# After sufficient samples
def analyze_results():
    stats_v1 = aggregate_metrics(version="v1")
    stats_v2 = aggregate_metrics(version="v2")

    if stats_v2.success_rate > stats_v1.success_rate * 1.05:
        return "v2 is significantly better"
```

**Sample Size Requirements:**
- Minimum 100-200 samples per variant
- Longer for smaller effect sizes
- Consider traffic volume and rarity

**Metrics for Comparison:**
- Success rate (primary)
- Cost per request
- User satisfaction
- Latency
- Hallucination rate

---

## Evaluation Automation

### Automated Testing Pipeline

```
New Agent Version
    ↓
Run Against Test Dataset (1,000 traces)
    ↓
LLM-as-Judge Evaluation
    ↓
Regression Test Suite
    ↓
Performance Benchmarks
    ↓
↙ If Results Good ↘
Report for Review    Automated Rollout
    ↓
Human Review
    ↓
Deploy to Production
```

### CI/CD Integration

**Pre-deployment Checks:**
- [ ] Success rate >95%
- [ ] No regression in key metrics
- [ ] Cost per request within bounds
- [ ] Latency within SLA
- [ ] All regression tests pass
- [ ] Human reviewer approval

**Post-deployment Monitoring:**
- [ ] Track first hour metrics carefully
- [ ] Compare against baseline
- [ ] Alert on any anomalies
- [ ] Easy rollback if needed

---

## Cost of Evaluation

### Sampling Trade-offs

```
100% Evaluation (All Traces)
├─ Cost: ~2-5% of inference cost
├─ Data: Comprehensive ground truth
└─ Latency: Adds ~500ms per request

10% Random Sample
├─ Cost: ~0.2-0.5% of inference cost
├─ Data: Representative sample
└─ Latency: No impact on production

5% Strategic Sample (errors + random)
├─ Cost: ~0.1-0.3% of inference cost
├─ Data: Focuses on problem areas
└─ Latency: No impact on production
```

**Recommendation:** Use 5-10% sampling with error oversampling for production systems.

---

## Best Practices

### Evaluation Discipline

- [ ] Define evaluation criteria before deployment
- [ ] Use consistent evaluation across versions
- [ ] Automate evaluation as much as possible
- [ ] Sample production traces regularly
- [ ] Maintain evaluation datasets
- [ ] A/B test before full rollout
- [ ] Track evaluation coverage

### Data Quality

- [ ] Validate human annotations
- [ ] Inter-rater agreement >85%
- [ ] Document evaluation methodology
- [ ] Version evaluation criteria
- [ ] Archive evaluation results

### Iteration

- [ ] Weekly evaluation review
- [ ] Monthly trend analysis
- [ ] Quarterly benchmark updates
- [ ] Quarterly eval methodology review
- [ ] Build improvement backlog

---

## Common Issues and Solutions

### Issue 1: Low Success Rate

**Diagnosis:**
- Review failed traces in LangSmith
- Analyze error patterns
- Check if external services down

**Solutions:**
- Improve prompts with examples
- Retrain with recent failures
- Add error handling paths
- Implement fallback mechanisms

### Issue 2: High Hallucination Rate

**Diagnosis:**
- Check if agent accessing retrieval
- Verify retrieval quality
- Analyze when hallucinations occur

**Solutions:**
- Improve retrieval system
- Add citation requirements
- Require source verification
- Use smaller models if appropriate

### Issue 3: Cost Exploding

**Diagnosis:**
- Track tokens per successful task
- Identify high-cost request patterns
- Check for infinite loops/retries

**Solutions:**
- Optimize prompts (shorter context)
- Implement early stopping
- Use smaller models for simple tasks
- Implement budget limits

---

## Tools and Platforms

**Evaluation Tools:**
- LangSmith (agent-specific evaluation)
- Weights & Biases (ML evaluation)
- Hugging Face Hub (benchmark datasets)

**Benchmark Platforms:**
- HELM (Holistic Evaluation of Language Models)
- SuperGLUE/GLUE (language understanding)
- OpenAI Evals (multi-modal)

---

## References

- **LangSmith Evaluation:** See Chapter 8/01-LangSmith-Agent-Monitoring.md
- **ML Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md
- **Agent Architecture:** See Chapters 1-3

---

## Conclusion

Comprehensive evaluation frameworks enable continuous improvement of agent systems. By combining automated evaluation with human review, establishing clear metrics, and systematically testing improvements, teams ensure agent quality remains high throughout the system lifecycle.

**Key Principle:** What gets measured gets improved. Establish evaluation discipline early.
