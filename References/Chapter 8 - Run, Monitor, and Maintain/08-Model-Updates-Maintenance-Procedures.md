# Model Updates and Maintenance Procedures

**Source:** MLOps best practices and NVIDIA platform guidelines

**Focus:** Safe and effective model updates and system maintenance
**Scope:** Update strategies, testing procedures, rollout processes, rollback procedures

---

## Model Update Types and Strategies

### Type 1: Critical Security Patches

**Examples:**
- Security vulnerability in model or dependencies
- Critical bug affecting correctness
- Data breach or privacy issue

**Strategy: Immediate Deployment**
```
1. Prepare patch (fix the issue)
2. Test in staging (smoke tests)
3. Deploy immediately (minimal rollout)
4. Monitor closely for issues
5. Have immediate rollback ready
```

**Example Deployment:**
```bash
# Critical patch deployment
git checkout -b hotfix/security-patch
# Fix issue
git commit -m "SECURITY: Fix critical vulnerability"
git push origin hotfix/security-patch

# Fast-track review
git pull-request --urgency critical

# Deploy immediately after review
kubectl apply -f deployment.yaml --record
kubectl rollout status deployment/agent-service
```

### Type 2: Feature Updates

**Examples:**
- New agent capabilities
- Improved reasoning patterns
- Additional tool integration
- UI/UX improvements

**Strategy: Staged Rollout**
```
Canary Deployment:
└─ 5% of traffic → 25% → 50% → 100%

At each stage:
├─ Monitor success rate
├─ Compare vs. baseline
├─ Check for regressions
└─ Alert on any issues
```

**Example Staged Rollout:**
```python
def staged_rollout(new_model, stages=[0.05, 0.25, 0.50, 1.0]):
    """Deploy new model in stages"""
    for traffic_percentage in stages:
        # Route traffic_percentage to new model
        update_traffic_split(
            model_v1=(1 - traffic_percentage),
            model_v2=traffic_percentage
        )

        # Monitor for 30 minutes
        wait_time = 30  # minutes
        print(f"Monitoring {traffic_percentage:.0%} traffic for {wait_time}min")
        time.sleep(wait_time * 60)

        # Check metrics
        metrics = compare_models()

        if metrics["new_model_better"]:
            print(f"✓ Success rate improved at {traffic_percentage:.0%}")
            continue
        elif metrics["no_regression"]:
            print(f"✓ No regression at {traffic_percentage:.0%}")
            continue
        else:
            # Rollback
            print(f"✗ Regression detected at {traffic_percentage:.0%}")
            rollback(new_model)
            return False

    print("✓ Full rollout successful")
    return True
```

### Type 3: Dependency Updates

**Examples:**
- Python package updates
- CUDA/cuDNN version upgrades
- Infrastructure updates
- Tool/API changes

**Strategy: Careful Testing**
```
1. Update in isolated environment
2. Run full test suite
3. Performance benchmark
4. Check for breakages
5. Staged deployment
```

---

## Pre-Update Testing

### Test Level 1: Unit Tests

```python
def run_unit_tests():
    """Quick validation that basic functionality works"""
    import pytest

    # Run fast unit tests
    result = pytest.main([
        "tests/unit/",
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])

    return result == 0

def test_basic_functionality():
    """Example unit test"""
    agent = create_agent()
    result = agent.invoke({"input": "hello"})

    assert result.success
    assert len(result.output) > 0
```

### Test Level 2: Integration Tests

```python
def run_integration_tests():
    """Test complete workflows"""
    test_cases = [
        {
            "input": "What is 2+2?",
            "expected_tools": ["calculator"],
            "expected_answer": "4",
        },
        {
            "input": "Search for Python documentation",
            "expected_tools": ["search"],
            "min_results": 1,
        },
        {
            "input": "Tell me a joke",
            "tools_needed": [],
            "checks": ["contains_humor"],
        }
    ]

    for test in test_cases:
        result = agent.invoke({"input": test["input"]})

        # Verify tools were called correctly
        for tool_name in test.get("expected_tools", []):
            assert tool_name in [t.name for t in result.tool_calls]

        # Verify quality
        for check in test.get("checks", []):
            assert run_check(check, result)
```

### Test Level 3: Performance Benchmarks

```python
def run_performance_benchmarks(baseline_metrics):
    """Compare performance against baseline"""
    test_queries = load_test_queries(count=100)

    latencies = []
    success_count = 0

    for query in test_queries:
        start = time.time()
        result = agent.invoke({"input": query})
        latency = time.time() - start

        latencies.append(latency)
        if result.success:
            success_count += 1

    # Compute metrics
    metrics = {
        "latency_p50": np.percentile(latencies, 50),
        "latency_p95": np.percentile(latencies, 95),
        "latency_p99": np.percentile(latencies, 99),
        "success_rate": success_count / len(test_queries),
    }

    # Compare to baseline
    print("=== Performance Comparison ===")
    for metric, new_value in metrics.items():
        baseline = baseline_metrics[metric]
        change = (new_value - baseline) / baseline * 100

        status = "✓" if change < 5 else "✗"  # <5% regression OK
        print(f"{status} {metric:20} {new_value:8.2f} (baseline: {baseline:8.2f}, {change:+.1f}%)")

    return metrics
```

### Test Level 4: Quality Evaluation

```python
def run_quality_evaluation(sample_size=50):
    """Automated quality check"""
    from langsmith import evaluate

    # Get test dataset
    test_dataset = load_test_dataset(sample_size)

    # Define evaluators
    evaluators = [
        ("correctness", check_correctness),
        ("completeness", check_completeness),
        ("clarity", check_clarity),
    ]

    # Run evaluation
    results = evaluate(
        dataset=test_dataset,
        evaluators=evaluators,
        model=agent
    )

    # Report results
    for evaluator_name, scores in results.items():
        avg_score = np.mean(scores)
        print(f"{evaluator_name}: {avg_score:.2f}/5.0")

    return results
```

---

## Update Deployment Process

### Stage 1: Preparation

**Checklist:**
```
Pre-Deployment Checklist:
├─ [ ] Code reviewed and approved
├─ [ ] All tests passing
├─ [ ] Performance benchmarks acceptable
├─ [ ] Rollback plan prepared
├─ [ ] Team notified
├─ [ ] Monitoring dashboards ready
└─ [ ] Incident response on call
```

**Prepare Deployment Package:**
```bash
# Tag version
git tag -a v1.2.0 -m "Agent system update v1.2.0"
git push origin v1.2.0

# Build Docker image
docker build -t agent:v1.2.0 .
docker push gcr.io/my-project/agent:v1.2.0

# Prepare Kubernetes manifests
kubectl apply -f deployments/agent-v1.2.0.yaml --dry-run=client
```

### Stage 2: Staging Deployment

**Test in Production-Like Environment:**
```bash
# Deploy to staging environment
kubectl apply -f staging/agent-v1.2.0.yaml

# Run smoke tests
pytest tests/smoke/ --env=staging

# Run soak tests (24 hour stability test)
python scripts/soak_test.py --env=staging --duration=24h

# Load test
python scripts/load_test.py --env=staging --qps=100
```

### Stage 3: Canary Deployment

**Deploy to 5% of Production Traffic:**
```bash
# Create canary deployment
kubectl apply -f deployments/agent-v1.2.0-canary.yaml

# Update traffic split (5% new, 95% old)
kubectl patch service agent-service \
  -p '{"spec":{"trafficPolicy":{"canary":{"weight":5}}}}'

# Monitor for 1 hour
watch kubectl get pods -l app=agent-canary
```

**Monitor Canary Metrics:**
```python
def monitor_canary(duration_minutes=60):
    """Monitor canary deployment"""
    start_time = time.time()

    while time.time() - start_time < duration_minutes * 60:
        metrics = get_metrics_comparison("v1.1.0", "v1.2.0")

        print(f"\n=== Canary Monitor (5% traffic) ===")
        print(f"Success Rate v1.1: {metrics['v1.1']['success_rate']:.2%}")
        print(f"Success Rate v1.2: {metrics['v1.2']['success_rate']:.2%}")

        # Check for regression
        if metrics['v1.2']['success_rate'] < metrics['v1.1']['success_rate'] * 0.95:
            alert("ROLLBACK: Success rate regression detected")
            return False

        if metrics['v1.2']['latency_p95'] > metrics['v1.1']['latency_p95'] * 1.1:
            alert("ROLLBACK: Latency regression detected")
            return False

        time.sleep(60)  # Check every minute

    return True
```

### Stage 4: Progressive Rollout

**Increase Traffic in Steps:**
```python
def progressive_rollout(stages=[0.1, 0.25, 0.5, 1.0]):
    """Gradually increase traffic to new version"""
    for stage_percentage in stages:
        print(f"\n=== Rolling out to {stage_percentage:.0%} traffic ===")

        # Update traffic split
        kubectl patch service agent-service \
          -p f'{{"spec":{{"trafficPolicy":{{"canary":{{"weight":{stage_percentage * 100}}}}}}}}'

        # Monitor for extended period
        success = monitor_canary(duration_minutes=30)

        if not success:
            print(f"✗ Regression at {stage_percentage:.0%}, rolling back")
            rollback()
            return False

        print(f"✓ Healthy at {stage_percentage:.0%}, proceeding")

    print("✓ Full rollout complete")
    return True
```

### Stage 5: Cleanup

**Remove Old Version:**
```bash
# After successful rollout
kubectl delete deployment agent-v1.1.0

# Keep old version available for quick rollback (4 hours)
kubectl delete deployment agent-v1.1.0 --grace-period=14400
```

---

## Rollback Procedures

### Immediate Rollback (Critical Issue)

```python
def emergency_rollback():
    """Rollback to previous version immediately"""
    print("EMERGENCY ROLLBACK INITIATED")

    # 1. Instant traffic switch
    kubectl patch service agent-service \
      -p '{"spec":{"selector":{"version":"v1.1.0"}}}'

    # 2. Scale down new version immediately
    kubectl scale deployment agent-v1.2.0 --replicas=0

    # 3. Scale up old version
    kubectl scale deployment agent-v1.1.0 --replicas=20

    # 4. Notify team
    send_alert("CRITICAL: Rolled back to v1.1.0")

    # 5. Monitor
    wait_for_stability()

    print("Rollback complete")
```

**When to Use:**
- Error rate >5%
- Latency >200% baseline
- System crash/unavailability
- Data corruption detected

### Controlled Rollback (Minor Issue)

```python
def controlled_rollback():
    """Gradually rollback to previous version"""
    # Reverse the progressive rollout
    stages = [0.5, 0.25, 0.1, 0.0]

    for stage in stages:
        print(f"Rolling back to {stage:.0%} new version")
        update_traffic_split(new_version=stage)
        time.sleep(600)  # Monitor 10 min

        metrics = get_metrics()
        if not is_stable(metrics):
            print("Stability lost, accelerating rollback")
            break

    print("Rollback complete")
```

**When to Use:**
- Moderate quality degradation
- Performance issues
- Non-critical bugs

---

## Scheduled Maintenance

### Maintenance Windows

```python
def schedule_maintenance(maintenance_type, duration_hours=4):
    """Schedule system maintenance"""
    maintenance_window = {
        "type": maintenance_type,  # e.g., "database_upgrade"
        "start": get_next_maintenance_window(),
        "duration": duration_hours,
        "impact": "Expected degraded performance",
    }

    # Notify users
    notify_scheduled_maintenance(maintenance_window)

    # Prepare
    backup_system_state()
    prepare_rollback_point()

    # Execute maintenance
    execute_maintenance()

    # Verify
    verify_system_health()

    # Notify users - complete
    notify_maintenance_complete()
```

### Maintenance Checklist

```
Scheduled Maintenance Checklist:
└─ Database Upgrade
   ├─ [ ] Backup database
   ├─ [ ] Test upgrade in staging
   ├─ [ ] Plan rollback
   ├─ [ ] Notify users
   ├─ [ ] Execute upgrade
   ├─ [ ] Verify data integrity
   ├─ [ ] Run smoke tests
   ├─ [ ] Monitor for 1 hour
   └─ [ ] Update documentation
```

---

## Update Tracking and Versioning

### Version Management

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent
    version: v1.2.0
    release-date: "2024-01-15"
    updated-by: "engineering-team"
spec:
  template:
    spec:
      containers:
      - name: agent
        image: gcr.io/my-project/agent:v1.2.0
        env:
        - name: VERSION
          value: "v1.2.0"
        - name: BUILD_ID
          value: "12345"
        - name: BUILD_TIME
          value: "2024-01-15T10:30:00Z"
```

### Update History Tracking

```python
class UpdateTracker:
    def __init__(self, db):
        self.db = db

    def record_update(self, version, update_type, status, details):
        """Record an update event"""
        self.db.insert({
            "version": version,
            "type": update_type,  # "feature", "fix", "security"
            "status": status,      # "success", "rollback"
            "timestamp": datetime.now(),
            "duration": details.get("duration"),
            "affected_users": details.get("affected_users", 0),
            "previous_version": details.get("previous_version"),
            "changelog": details.get("changelog"),
        })

    def get_update_history(self, days=30):
        """Get recent update history"""
        return self.db.query(
            f"SELECT * FROM updates WHERE timestamp > NOW() - INTERVAL {days} day"
        )
```

---

## Best Practices

### Update Safety

- [ ] Always test before deploying
- [ ] Use staged rollout strategy
- [ ] Monitor metrics closely during rollout
- [ ] Have rollback plan ready
- [ ] Test rollback procedure regularly

### Update Timing

- [ ] Schedule during low-traffic periods
- [ ] Avoid Friday/holiday deployments
- [ ] Inform users in advance
- [ ] Have support team on standby
- [ ] Allow time for monitoring

### Documentation

- [ ] Maintain detailed changelog
- [ ] Document known issues
- [ ] Keep runbooks updated
- [ ] Record update procedures
- [ ] Share lessons learned

---

## References

- **Testing and Evaluation:** See Chapter 8/03-Agent-Evaluation-Frameworks.md
- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md
- **Incident Response:** See Chapter 8/06-Error-Troubleshooting-Incident-Response.md

---

## Conclusion

Safe and effective model updates require careful planning, thorough testing, and staged deployment. By following systematic update procedures with comprehensive rollback capabilities, teams maintain system stability while continuously improving their agent systems.

**Key Principle:** Test thoroughly, deploy cautiously, monitor closely, rollback quickly.
