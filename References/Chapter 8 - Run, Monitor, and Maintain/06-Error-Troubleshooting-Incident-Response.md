# Error Troubleshooting and Incident Response for Agent Systems

**Source:** Production operations best practices and NVIDIA platform documentation

**Focus:** Identifying and resolving issues in deployed agent systems
**Scope:** Error types, debugging techniques, incident response, recovery procedures

---

## Common Agent System Errors

### Error Category 1: Agent Execution Errors

**Hallucination/False Information**
```
Symptom: Agent generates information not present in context
Example: "The 2024 World Cup winner was France" (when docs say Argentina)

Root Causes:
├─ Model defaulting to training data
├─ Insufficient retrieval quality
├─ Ambiguous user query
├─ Out-of-domain question

Severity: HIGH - Impacts correctness
Detection: Automated fact-checking against retrieval sources
```

**Tool Call Errors**
```
Symptom: Invalid tool invocations or parameter errors
Example:
├─ Calling non-existent function
├─ Wrong parameter types
├─ Missing required parameters
└─ Invalid parameter values

Root Causes:
├─ Model confused about tool schema
├─ Ambiguous tool descriptions
├─ Tool not properly initialized
└─ Context window overflow

Severity: MEDIUM - Causes task failure
Detection: Schema validation, parameter type checking
```

**Decision Loop/Infinite Retry**
```
Symptom: Agent retries same action repeatedly
Example: Tool fails, agent retries 50 times without fallback

Root Causes:
├─ No fallback mechanism
├─ Retry logic too aggressive
├─ Tool in persistent failure state
└─ Prompt not handling errors well

Severity: HIGH - Consumes resources, impacts costs
Detection: Iteration count limits, timeout tracking
```

### Error Category 2: System Errors

**Out of Memory (OOM)**
```
Symptom: CUDA Out of Memory error
Example: "RuntimeError: CUDA out of memory"

Root Causes:
├─ Context window too large
├─ Batch size too large
├─ Memory fragmentation
├─ Model doesn't fit on GPU
├─ Long-running accumulation

Severity: CRITICAL - Service degradation
Detection: Monitor GPU memory usage, catch exceptions
```

**Latency Timeout**
```
Symptom: Agent execution exceeds SLA
Example: Response takes >30 seconds

Root Causes:
├─ Model inference slow
├─ Tool call timeout
├─ Waiting for external API
├─ Network issues
└─ High system load

Severity: HIGH - User-facing impact
Detection: Latency monitoring, timeout enforcement
```

**Database Connection Errors**
```
Symptom: Cannot connect to knowledge base or memory store
Example: "Connection refused: localhost:5432"

Root Causes:
├─ Service down
├─ Network connectivity
├─ Authentication failure
├─ Resource limit exceeded
└─ Connection pool exhausted

Severity: CRITICAL - Service outage
Detection: Health checks, connection pooling
```

### Error Category 3: Data Errors

**Missing or Corrupt Data**
```
Symptom: Agent retrieves invalid or incomplete data
Example: Knowledge base returns null values

Root Causes:
├─ Data pipeline failure
├─ Encoding issues
├─ Truncation in storage
├─ Schema mismatch
└─ Indexing failure

Severity: MEDIUM - Quality degradation
Detection: Schema validation, null checks
```

**API Response Errors**
```
Symptom: External tool returns error or unexpected format
Example: Search API returns 500 error

Root Causes:
├─ Third-party service down
├─ Rate limiting
├─ Invalid request format
├─ Authentication expired
└─ Business logic error in tool

Severity: MEDIUM-HIGH - Depends on criticality
Detection: Response code checking, schema validation
```

---

## Debugging Workflows

### Workflow 1: Quick Diagnosis

**Steps:**
```
1. Reproduce the error
   ├─ Get exact input that caused error
   ├─ Verify it fails consistently
   └─ Note error message and timestamp

2. Check recent changes
   ├─ Code deployments in last 24h?
   ├─ Model version changes?
   ├─ Prompt modifications?
   ├─ External service changes?
   └─ Data pipeline changes?

3. Check system health
   ├─ GPU/memory available?
   ├─ Services running?
   ├─ Network connectivity?
   ├─ Disk space available?
   └─ Rate limits hit?

4. Check logs
   ├─ Error message in application logs?
   ├─ Stack trace available?
   ├─ Related log entries?
   └─ Timestamps consistent?

5. Initial assessment
   ├─ Is this a known issue?
   ├─ Can we replicate in test environment?
   ├─ What's the scope (all users or specific)?
   └─ Is workaround available?
```

### Workflow 2: Deep Dive Analysis

**For Complex Issues:**

```python
def debug_agent_execution(request_id):
    # Get full trace from LangSmith
    trace = langsmith_client.get_run(request_id)

    # Analyze each step
    for step in trace.child_runs:
        print(f"\n--- Step: {step.name} ---")
        print(f"Status: {step.status}")
        print(f"Latency: {step.end_time - step.start_time}")
        print(f"Input: {step.inputs}")
        print(f"Output: {step.outputs}")

        if step.error:
            print(f"ERROR: {step.error}")

    # Check for patterns
    print("\n--- Summary ---")
    total_latency = trace.end_time - trace.start_time
    step_count = len(trace.child_runs)
    print(f"Total latency: {total_latency}s")
    print(f"Steps taken: {step_count}")
    print(f"Average step latency: {total_latency/step_count}s")

    # Identify bottlenecks
    slowest = max(trace.child_runs,
                  key=lambda s: s.end_time - s.start_time)
    print(f"\nSlowest step: {slowest.name} ({slowest.end_time - slowest.start_time}s)")
```

---

## Incident Response Procedure

### Phase 1: Detection and Triage (0-5 minutes)

**Detection:**
```
Error Symptom Detected
    ↓
Automated Alert Triggered
    ↓
Incident Ticket Created
    ↓
Page on-call Engineer
```

**Initial Triage:**
```
Questions to Answer:
1. Is service down? (availability monitoring)
2. How many users affected? (error rate, affected regions)
3. Is it growing? (trending alerts)
4. Do we have workaround? (documented procedures)
5. What changed recently? (deployment, config changes)
```

### Phase 2: Investigation (5-30 minutes)

**Data Collection:**
```
Gather:
├─ Affected user samples
├─ Error logs and stack traces
├─ Performance metrics (before/after)
├─ Recent changes/deployments
├─ External service status
└─ Resource utilization
```

**Root Cause Hypothesis:**
```
Based on evidence:
1. Is it code-related? (logic bug, new feature)
2. Is it infrastructure? (resources, connectivity)
3. Is it external? (API down, rate limit)
4. Is it data-related? (bad data, missing schema)
5. Is it configuration? (wrong settings, env var)
```

**Verification:**
```
Test Hypothesis:
├─ Reproduce in test environment?
├─ Can we isolate the component?
├─ Does reverting recent change fix it?
└─ Do logs support this theory?
```

### Phase 3: Resolution (30 minutes - 2+ hours)

**Fix Options by Category:**

**Quick Mitigation (if immediate fix unavailable):**
```
├─ Scale up resources (if capacity issue)
├─ Scale down traffic (rate limit or queue)
├─ Disable problematic feature
├─ Route to fallback service
├─ Revert to previous version
└─ Route to smaller, slower model
```

**Permanent Fix:**
```
Code Fix:
├─ Identify problematic code
├─ Implement fix
├─ Test in staging
├─ Deploy with monitoring

Config Fix:
├─ Identify wrong configuration
├─ Update to correct value
├─ Verify services pick up change
├─ Monitor for impact

Data Fix:
├─ Identify bad data
├─ Repair/reload data
├─ Verify correctness
├─ Monitor queries
```

### Phase 4: Monitoring and Recovery (2+ hours)

**Verification:**
```
After Fix Deployed:
├─ Error rate returning to normal?
├─ Latency back to baseline?
├─ User reports stopping?
├─ Resource utilization normal?
└─ No new errors introduced?
```

**Gradual Rollout (if applicable):**
```
1% of traffic → 10% → 50% → 100%
Monitor at each step for regression
```

### Phase 5: Post-Incident Review

**Documentation:**
```
Incident Report should include:
1. What happened (incident description)
2. When it happened (timeline)
3. Impact (users affected, duration, cost)
4. Root cause (what actually went wrong)
5. Detection (how we found it)
6. Resolution (what we did to fix)
7. Prevention (how to prevent recurrence)
```

**Example Report:**
```
INCIDENT REPORT: Service Latency Spike

WHAT HAPPENED:
Agent response time increased from 2s to 15s

TIMELINE:
- 14:32 UTC: Latency spike detected by alert
- 14:35 UTC: Incident declared
- 14:42 UTC: Root cause identified
- 14:55 UTC: Fix deployed
- 15:05 UTC: Normal latency restored
- Duration: 33 minutes

IMPACT:
- 8,450 affected requests
- 2.5% error rate (vs 0.1% normal)
- 12 users reported slowness
- Estimated $150 in excess cost

ROOT CAUSE:
Vector database (Pinecone) became unresponsive due to spike in search QPS.
Service was not properly handling timeouts, retrying indefinitely.

DETECTION:
Automated latency alert at p95 threshold

RESOLUTION:
1. Increased timeout to 5s (with retry logic)
2. Added circuit breaker for vector DB
3. Deployed at 14:55 UTC

PREVENTION:
- Add health checks for vector DB
- Implement request timeout limits
- Add fallback to approximate search
- Test under load
```

---

## Common Error Patterns and Solutions

### Pattern 1: Cascading Failures

**Symptom:**
```
One tool fails → Agent retries
→ Overloads tool with requests
→ Tool gets slower/fails more
→ More retries → Complete failure
```

**Detection:**
```python
def detect_cascading_failure(traces):
    for trace in traces:
        if trace.retry_count > 10:
            # Excessive retries suggest cascade
            alert(f"Potential cascade in {trace.request_id}")

        if trace.error_rate > 20:
            # High error rate compounds
            alert(f"High error rate cascade detected")
```

**Solution:**
```python
# Implement circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.is_open = False

    def call(self, func, *args):
        if self.is_open:
            raise Exception("Circuit breaker open")

        try:
            result = func(*args)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise
```

### Pattern 2: Resource Exhaustion

**Symptom:**
```
Growing memory usage over time
→ Eventually OOM error
→ Service crash
→ No recovery without restart
```

**Detection:**
```python
def monitor_memory_leak():
    baseline = get_memory_usage()

    for i in range(10):
        time.sleep(10)
        current = get_memory_usage()
        growth = current - baseline

        if growth > baseline * 0.1:  # 10% growth
            alert(f"Potential memory leak: {growth}MB growth")
```

**Solution:**
```python
# 1. Profile to find leak
with memory_profiler.profile():
    for _ in range(100):
        trace = execute_agent(query)

# 2. Fix common issues
# ├─ Clear caches periodically
cache.clear()  # Don't let cache grow unbounded

# ├─ Close connections
db.close_connection()

# ├─ Delete large temp objects
del large_temp_data

# 3. Implement auto-restart
# Periodic worker restart (e.g., every 1000 requests)
```

### Pattern 3: Model Drift/Quality Degradation

**Symptom:**
```
Success rate declining gradually
Success rate: Week 1: 95%, Week 2: 92%, Week 3: 88%
```

**Detection:**
```python
def detect_quality_drift(window_days=7):
    current_metrics = get_metrics(last_n_days=1)
    baseline_metrics = get_metrics(
        start_date=today - window_days,
        end_date=today - 1
    )

    success_decline = (
        baseline_metrics['success_rate'] -
        current_metrics['success_rate']
    )

    if success_decline > 0.05:  # 5% decline
        alert(f"Quality drift detected: {success_decline:.1%}")
```

**Solution:**
```python
# Trigger automatic retraining
if detect_quality_drift():
    # Collect recent failure cases
    recent_failures = get_recent_failures(days=7)

    # Prepare training data
    training_data = prepare_sft_dataset(recent_failures)

    # Fine-tune model
    new_model = finetune_model(model, training_data)

    # A/B test new model
    route_10_percent_to_new_model(new_model)

    # Monitor and rollout
    if success_rate_improves(new_model):
        fully_deploy(new_model)
```

---

## Recovery Procedures

### Quick Recovery Steps

**For OOM Error:**
```
1. Reduce batch size by 50%
2. Clear GPU memory cache
3. Restart inference container
4. Scale down concurrent requests
5. If persistent: reduce context window length
```

**For Latency Timeout:**
```
1. Check external service health
2. Increase timeout threshold (if safe)
3. Scale up resources
4. Check for resource bottleneck
5. Consider model downgrade
```

**For Tool Call Failure:**
```
1. Verify tool configuration
2. Check tool API status
3. Verify authentication tokens
4. Test tool manually
5. Implement fallback behavior
```

**For Database Errors:**
```
1. Check database service status
2. Verify network connectivity
3. Check authentication
4. Restart connection pool
5. Failover to replica if available
```

---

## Runbook Examples

### Runbook: High Latency Response

```
ISSUE: Response latency >10 seconds

DIAGNOSIS (2 min):
1. Check LangSmith dashboard for slow traces
2. Identify bottleneck step (LLM, tool, retrieval)
3. Check corresponding service health

IF LLM INFERENCE SLOW:
├─ Check GPU memory usage
├─ Check model serving status
├─ Check request queue depth
└─ Scale up if needed

IF TOOL CALL SLOW:
├─ Check tool service health
├─ Check for rate limiting
├─ Check network latency
└─ Try fallback tool

IF RETRIEVAL SLOW:
├─ Check vector DB health
├─ Check vector DB query latency
├─ Check cache hit rate
└─ Reduce search scope if possible
```

### Runbook: High Error Rate

```
ISSUE: Error rate >2%

DIAGNOSIS (2 min):
1. Check error rate by error type
2. Identify when error started
3. Check for recent changes

ERROR TYPES:

A. Tool Call Errors:
   ├─ Verify tool schemas
   ├─ Check tool responses
   └─ Roll back if recent change

B. Hallucination Errors:
   ├─ Check retrieval quality
   ├─ Reduce context size
   └─ Add fact-checking

C. System Errors:
   ├─ Restart services
   ├─ Scale up resources
   └─ Revert last deployment

D. Input Validation Errors:
   ├─ Check input validation rules
   ├─ Review recent prompt changes
   └─ Add more robust validation
```

---

## Best Practices

### Prevention

- [ ] Comprehensive error handling
- [ ] Timeout limits on all operations
- [ ] Circuit breakers for external services
- [ ] Resource quotas and limits
- [ ] Automated health checks
- [ ] Regular testing and load testing

### Detection

- [ ] Alerting on key metrics
- [ ] Error rate monitoring
- [ ] Latency distribution tracking
- [ ] Resource utilization limits
- [ ] Regular log review

### Response

- [ ] Clear incident procedures
- [ ] Runbooks for common issues
- [ ] On-call rotation
- [ ] Easy rollback procedures
- [ ] Communication channels

---

## References

- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md
- **Observability:** See Chapter 8/01-LangSmith-Agent-Monitoring.md
- **Cost Optimization:** See Chapter 8/05-Cost-Optimization-Resource-Monitoring.md

---

## Conclusion

Effective troubleshooting and incident response minimize downtime and maintain system reliability. By establishing clear procedures, documenting common issues, and practicing incident response, teams develop the muscle memory to handle production problems quickly and effectively.

**Core Principle:** Detect early, diagnose methodically, resolve decisively, learn thoroughly.
