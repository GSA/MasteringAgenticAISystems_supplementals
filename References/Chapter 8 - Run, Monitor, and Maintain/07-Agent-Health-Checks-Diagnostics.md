# Agent Health Checks and Diagnostics

**Source:** Production operations and NVIDIA platform best practices

**Focus:** Continuous health assessment and diagnostic tools for agent systems
**Scope:** Health check frameworks, diagnostic procedures, automated testing

---

## Health Check Layers

### Layer 1: Service-Level Health

**Basic Availability Check:**
```python
def health_check_basic():
    """Check if agent service is responding"""
    try:
        response = requests.get(
            "http://agent-service:8000/health",
            timeout=2
        )
        return response.status_code == 200
    except Exception:
        return False
```

**Response Time Health:**
```python
def health_check_latency():
    """Check if response time is acceptable"""
    start = time.time()
    response = agent.invoke({"input": "test query"})
    latency = time.time() - start

    if latency > 5.0:  # SLA threshold
        return False, f"Latency {latency}s exceeds SLA"
    return True, f"Latency OK: {latency}s"
```

**Resource Availability:**
```python
def health_check_resources():
    """Check if resources are available"""
    checks = {}

    # GPU availability
    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_free = torch.cuda.mem_get_info()[0]
        checks['gpu'] = gpu_free > gpu_memory * 0.1  # At least 10% free
    except:
        checks['gpu'] = False

    # Memory availability
    import psutil
    memory_percent = psutil.virtual_memory().percent
    checks['memory'] = memory_percent < 85

    # Disk space
    disk_percent = psutil.disk_usage('/').percent
    checks['disk'] = disk_percent < 90

    return all(checks.values()), checks
```

### Layer 2: Model-Level Health

**Model Loading Check:**
```python
def health_check_model_load():
    """Verify model loaded correctly"""
    try:
        # Check model parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0, "No model parameters"

        # Quick inference test
        with torch.no_grad():
            output = model(test_input)

        return True, f"Model OK: {param_count} parameters"
    except Exception as e:
        return False, str(e)
```

**Model Output Quality:**
```python
def health_check_model_quality():
    """Check if model outputs reasonable results"""
    test_cases = [
        ("What is 2+2?", "4"),  # Should contain 4
        ("What is your name?", "assistant"),  # Should be humble
        ("Say hello", "hello"),  # Should contain greeting
    ]

    results = []
    for query, expected_pattern in test_cases:
        response = model.generate(query)
        matches = expected_pattern.lower() in response.lower()
        results.append(matches)

    success_rate = sum(results) / len(results)
    return success_rate > 0.8, f"Quality: {success_rate:.1%}"
```

### Layer 3: Integration Health

**Tool Integration Check:**
```python
def health_check_tools():
    """Verify all tools are accessible"""
    tool_health = {}

    for tool_name, tool_func in agent.tools.items():
        try:
            # Quick test of tool
            if tool_name == "search":
                result = tool_func("test query")
            elif tool_name == "database":
                result = tool_func("SELECT 1")
            else:
                result = tool_func()

            tool_health[tool_name] = (True, "OK")
        except Exception as e:
            tool_health[tool_name] = (False, str(e))

    all_ok = all(status for status, _ in tool_health.values())
    return all_ok, tool_health
```

**Memory Store Health:**
```python
def health_check_memory():
    """Check knowledge base and memory systems"""
    checks = {}

    # Knowledge base
    try:
        doc_count = knowledge_base.count()
        checks['kb_available'] = doc_count > 0
        checks['kb_size'] = doc_count
    except:
        checks['kb_available'] = False

    # Vector DB
    try:
        vector_db.search("test")
        checks['vector_db'] = True
    except:
        checks['vector_db'] = False

    # Long-term memory
    try:
        memory = agent.memory.load()
        checks['memory'] = len(memory) >= 0
    except:
        checks['memory'] = False

    return all(checks.values()), checks
```

### Layer 4: End-to-End Health

**Full Agent Workflow Test:**
```python
def health_check_e2e():
    """Complete agent execution test"""
    test_scenario = {
        "input": "What is the capital of France?",
        "expected": "Paris",
        "tools_used": ["retrieval"],
        "max_latency": 5.0
    }

    start = time.time()
    result = agent.invoke({"input": test_scenario["input"]})
    latency = time.time() - start

    checks = {
        "task_completed": result.success,
        "answer_correct": test_scenario["expected"].lower() in result.output.lower(),
        "latency_ok": latency < test_scenario["max_latency"],
        "tools_called": len(result.tool_calls) > 0,
    }

    return all(checks.values()), checks
```

---

## Comprehensive Health Check Framework

### Unified Health Check Endpoint

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/health")
async def health_check():
    """Complete system health check"""
    checks = {
        "service": await check_service(),
        "model": await check_model(),
        "tools": await check_tools(),
        "memory": await check_memory(),
        "resources": await check_resources(),
        "e2e": await check_e2e(),
    }

    # Overall status
    all_ok = all(status for check in checks.values() for status, _ in check.items())
    overall_status = "healthy" if all_ok else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "details": generate_summary(checks)
    }

@app.get("/health/deep")
async def deep_diagnostics():
    """Detailed diagnostics for debugging"""
    return {
        "memory_usage": get_memory_stats(),
        "model_metrics": get_model_metrics(),
        "tool_latencies": measure_tool_latencies(),
        "cache_stats": get_cache_stats(),
        "request_queue": get_queue_stats(),
        "recent_errors": get_recent_errors(),
    }
```

### Health Status Codes

```
Green (Healthy): All checks pass
├─ Availability: >99.5%
├─ Latency: p95 < SLA
├─ Resources: >10% available
└─ Quality: >95% success

Yellow (Degraded): Some issues
├─ Availability: 95-99.5%
├─ Latency: p95 at/near SLA
├─ Resources: 5-10% available
└─ Quality: 80-95% success

Red (Unhealthy): Critical issues
├─ Availability: <95%
├─ Latency: p95 > SLA
├─ Resources: <5% available
└─ Quality: <80% success
```

---

## Diagnostic Procedures

### Procedure 1: Performance Diagnostics

**Identify Performance Bottleneck:**

```python
def diagnose_performance(slow_query):
    """Find where time is spent"""
    trace = execute_with_timing(slow_query)

    print("=== Performance Breakdown ===")
    for step in trace.steps:
        duration = step.end_time - step.start_time
        percentage = (duration / trace.total_time) * 100

        print(f"{step.name:30} {duration:8.2f}s {percentage:5.1f}%")

    # Identify bottleneck
    slowest = max(trace.steps, key=lambda s: s.end_time - s.start_time)
    print(f"\nBottleneck: {slowest.name}")
    print(f"Optimization opportunity: {slowest.description}")
```

**Example Output:**
```
=== Performance Breakdown ===
Input Validation              0.05s   0.5%
Tool Selection               0.10s   1.0%
Tool Execution               3.50s  35.0%  ← Bottleneck
LLM Generation               5.00s  50.0%  ← Main bottleneck
Output Formatting            1.35s  13.5%

Bottleneck: LLM Generation
Optimization opportunity: Use faster model or implement caching
```

### Procedure 2: Quality Diagnostics

**Identify Quality Issues:**

```python
def diagnose_quality(recent_failures):
    """Categorize quality issues"""
    categories = {
        "hallucination": [],
        "tool_error": [],
        "incomplete": [],
        "timeout": [],
        "other": []
    }

    for failure in recent_failures:
        if "not found in context" in failure.error:
            categories["hallucination"].append(failure)
        elif "tool call failed" in failure.error:
            categories["tool_error"].append(failure)
        elif "incomplete response" in failure.error:
            categories["incomplete"].append(failure)
        elif "timeout" in failure.error:
            categories["timeout"].append(failure)
        else:
            categories["other"].append(failure)

    print("=== Quality Issues Breakdown ===")
    for category, failures in categories.items():
        print(f"{category:15} {len(failures):5} ({len(failures)/len(recent_failures)*100:5.1f}%)")

    # Recommendations
    print("\nRecommendations:")
    if categories["hallucination"]:
        print("- Improve retrieval quality")
        print("- Add fact-checking step")

    if categories["tool_error"]:
        print("- Verify tool configurations")
        print("- Add error handling")
```

### Procedure 3: Resource Diagnostics

**Analyze Resource Consumption:**

```python
def diagnose_resources():
    """Detailed resource analysis"""
    import psutil

    print("=== Resource Diagnostics ===\n")

    # GPU
    print("GPU Status:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9

        print(f"  GPU {i} ({props.name}):")
        print(f"    Allocated: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")
        print(f"    Reserved:  {reserved:.1f}GB")

    # CPU and Memory
    print("\nHost Resources:")
    print(f"  CPU Usage: {psutil.cpu_percent()}%")
    vm = psutil.virtual_memory()
    print(f"  RAM Usage: {vm.used/1e9:.1f}GB / {vm.total/1e9:.1f}GB ({vm.percent:.1f}%)")

    # Disk
    print("\nStorage:")
    du = psutil.disk_usage('/')
    print(f"  Root:     {du.used/1e9:.1f}GB / {du.total/1e9:.1f}GB ({du.percent:.1f}%)")

    # Network (if available)
    print("\nNetwork:")
    net = psutil.net_io_counters()
    print(f"  Bytes sent:     {net.bytes_sent/1e9:.2f}GB")
    print(f"  Bytes received: {net.bytes_recv/1e9:.2f}GB")
```

---

## Automated Health Monitoring

### Scheduled Health Checks

```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

# Every minute - basic health
scheduler.add_job(
    health_check_basic,
    'interval',
    minutes=1,
    id='health_basic'
)

# Every 5 minutes - detailed health
scheduler.add_job(
    health_check_detailed,
    'interval',
    minutes=5,
    id='health_detailed'
)

# Every hour - comprehensive diagnostics
scheduler.add_job(
    run_comprehensive_diagnostics,
    'interval',
    hours=1,
    id='health_comprehensive'
)

scheduler.start()
```

### Continuous Monitoring Dashboard

```python
def build_monitoring_dashboard():
    """Real-time health status dashboard"""
    from datetime import datetime

    while True:
        status = health_check()

        print("\033[2J")  # Clear screen
        print(f"=== Agent System Health ===")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Status: {status['status'].upper()}\n")

        print("Component Status:")
        for component, checks in status['checks'].items():
            ok = all(v for v in checks.values())
            symbol = "✓" if ok else "✗"
            print(f"  {symbol} {component}")

        print("\nMetrics:")
        print(f"  Latency p95: {get_latency_p95()}ms")
        print(f"  Error rate: {get_error_rate():.2f}%")
        print(f"  GPU util: {get_gpu_utilization():.1f}%")

        time.sleep(10)
```

---

## Diagnostic CLI Tools

### Health Check Command

```bash
# Check service health
agent-cli health check

# Get detailed metrics
agent-cli health details

# Run diagnostics
agent-cli health diagnose

# Monitor in real-time
agent-cli health monitor --interval 5s

# Check specific component
agent-cli health check --component model
agent-cli health check --component tools
agent-cli health check --component memory
```

### Diagnostic Tools

```bash
# Profile performance
agent-cli profile --queries 100

# Analyze failures
agent-cli analyze-failures --days 7

# Test tools
agent-cli test-tools --verbose

# Load test
agent-cli load-test --qps 100 --duration 60

# Memory analysis
agent-cli memory-analysis
```

---

## Common Diagnostics Patterns

### Pattern 1: Intermittent Failures

**Diagnosis:**
```
Symptom: Some requests fail, others succeed
Frequency: Random, not reproducible

Questions:
- Is it rate-dependent (happens at high load)?
- Is it time-dependent (specific hours)?
- Is it input-dependent (certain query types)?
- Is it resource-dependent (when memory low)?
```

**Debugging:**
```python
def analyze_intermittent_failures():
    """Identify pattern in failures"""
    failures = get_recent_failures()

    # Check for patterns
    patterns = {
        "by_time": analyze_by_time(failures),
        "by_load": analyze_by_load(failures),
        "by_input_type": analyze_by_input(failures),
        "by_resource": analyze_by_resource(failures),
    }

    # Find correlation
    for pattern_type, correlation in patterns.items():
        if correlation > 0.7:  # Strong correlation
            print(f"Pattern found: {pattern_type}")
            print(f"Correlation: {correlation:.2f}")
```

### Pattern 2: Gradual Degradation

**Diagnosis:**
```
Symptom: Performance slowly gets worse over time
Timeline:
- Hour 1: 2.0s latency
- Hour 3: 2.5s latency
- Hour 5: 3.2s latency
- Hour 8: 5.0s latency (timeout)

Likely causes:
- Memory leak
- Cache growth
- Connection pool exhaustion
- Accumulation of state
```

**Debugging:**
```python
def monitor_degradation_over_time():
    """Track metrics hourly to catch degradation"""
    metrics_history = []

    for hour in range(24):
        metrics = {
            "time": datetime.now(),
            "latency": get_latency_p95(),
            "memory": get_memory_usage(),
            "cache_size": get_cache_size(),
            "error_rate": get_error_rate(),
        }
        metrics_history.append(metrics)

        # Check for trend
        if len(metrics_history) > 3:
            trend = calculate_trend(metrics_history[-3:])
            if trend > 0.1:  # >10% hourly increase
                alert(f"Degradation detected: {trend:.1%}/hour")

        time.sleep(3600)  # Wait 1 hour
```

---

## Best Practices

### Health Check Design

- [ ] Start with simple checks (availability)
- [ ] Layer in complexity (performance, quality, diagnostics)
- [ ] Set appropriate thresholds
- [ ] Make checks fast (<1 second for basic)
- [ ] Don't let checks slow down system
- [ ] Test health checks themselves

### Diagnostics

- [ ] Collect sufficient data for debugging
- [ ] Preserve error context and logs
- [ ] Record timing information
- [ ] Track resource usage
- [ ] Document common issues

### Automation

- [ ] Run health checks frequently
- [ ] Alert on degradation trends
- [ ] Collect metrics for analysis
- [ ] Build dashboards for visibility
- [ ] Make it easy to investigate

---

## References

- **Incident Response:** See Chapter 8/06-Error-Troubleshooting-Incident-Response.md
- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md
- **Observability:** See Chapter 8/01-LangSmith-Agent-Monitoring.md

---

## Conclusion

Comprehensive health checks and diagnostics enable operators to quickly identify and resolve issues before they impact users. By implementing layered health checks, automated monitoring, and systematic diagnostic procedures, teams maintain high availability and reliability.

**Key Principle:** Monitor proactively, diagnose systematically, resolve decisively.
