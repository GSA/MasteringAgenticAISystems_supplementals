# Cost Optimization and Resource Monitoring for Agent Systems

**Source:** NVIDIA platform cost optimization and operational best practices

**Focus:** Financial efficiency and resource utilization in production
**Scope:** Cost tracking, optimization strategies, resource scheduling, budgeting

---

## Cost Model for Agent Systems

### Components of Agent System Costs

**Infrastructure Costs (60-70% of total)**
```
Monthly Cost Breakdown:
├─ GPU compute time
│  ├─ H100: $2-3/hour per GPU
│  ├─ A100: $1.50-2/hour per GPU
│  └─ L40S: $1.50-2/hour per GPU
│
├─ Memory (host RAM): $0.05-0.10/GB/month
├─ Storage: $0.10-0.25/GB/month
└─ Network: $0.01-0.15/GB transferred
```

**API and Third-Party Costs (20-30% of total)**
```
├─ LLM inference (if cloud provider)
│  └─ GPT-4: $0.03/input, $0.06/output token
│
├─ External tools/APIs
│  └─ Search API, data services, payment processors
│
└─ Observability/Monitoring
    ├─ LangSmith: $0.001-0.01/trace
    └─ Datadog/New Relic: $10-100+/month
```

**Operational Costs (10-15% of total)**
```
├─ Personnel (engineering/operations)
├─ License fees
└─ Professional services
```

### Cost Calculation Framework

**Per-Request Cost:**
```
Cost = (GPU Hours × GPU Rate) + (Tokens Generated × Token Price) + Other Costs
      = (Latency_seconds / 3600) × $2.50/hour + (1000 tokens × $0.00001) + $0.001
      = ~$0.001-0.005 per request
```

**Cost by Customer/Feature:**
```python
def calculate_cost_breakdown():
    costs = {
        "by_customer": {},
        "by_feature": {},
        "by_model": {},
        "by_time_period": {}
    }

    for trace in production_traces:
        # Track by customer
        customer = trace.user_id
        costs["by_customer"][customer] = (
            costs["by_customer"].get(customer, 0) +
            trace.gpu_cost + trace.api_cost
        )

        # Track by feature
        feature = trace.agent_type
        costs["by_feature"][feature] = (
            costs["by_feature"].get(feature, 0) +
            trace.total_cost
        )

    return costs
```

---

## Cost Optimization Strategies

### Strategy 1: Model Selection and Sizing

**Cost-Quality Trade-off:**

```
Model          Inference Cost  Quality  Recommended For
Nemotron 4B    $0.0001/1K      75%      Simple tasks, high volume
Nemotron 8B    $0.0005/1K      85%      General purpose
Nemotron 70B   $0.003/1K       95%      Complex reasoning
Nemotron 405B  $0.01/1K        98%      Enterprise critical
```

**Optimization:**
```python
def route_to_optimal_model(query):
    # Estimate complexity
    complexity = estimate_query_complexity(query)

    if complexity < 0.3:
        return route_to_model("nemotron-4b")  # 30x cheaper
    elif complexity < 0.6:
        return route_to_model("nemotron-8b")  # 10x cheaper
    else:
        return route_to_model("nemotron-70b")  # Higher quality
```

**Typical Savings:** 40-60% by using appropriate model size

### Strategy 2: Prompt Optimization

**Token Reduction Techniques:**

```
Optimization                    Savings
===========================================
Remove unnecessary examples     10-15%
Compress system prompt          5-10%
Use structured output format    15-20%
Remove redundant instructions   10-15%

Total potential savings: 40-60% tokens
```

**Example Optimization:**

```python
# Before: 1,200 tokens for similar queries
bad_prompt = """You are an expert assistant...
[12 examples of good responses]
Answer the following carefully...
[detailed instructions]"""

# After: 400 tokens
good_prompt = """You are a helpful assistant.
Instructions: [concise, clear]
Format: JSON with fields [list]"""
```

### Strategy 3: Batch Processing

**Request Batching:**

```
Single Request Model:
Request 1: 100 tokens → $0.001
Request 2: 100 tokens → $0.001
Request 3: 100 tokens → $0.001
Total: 300 tokens, 3 GPU activations

Batched Model:
Batch [Req1, Req2, Req3]: 300 tokens → $0.002
Savings: 33% fewer GPU activations
```

**Implementation:**
```python
class BatchProcessor:
    def __init__(self, batch_size=32, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []

    async def process(self, request):
        self.queue.append(request)

        # Process when batch full or timeout
        if len(self.queue) >= self.batch_size:
            return await self.flush()

    async def flush(self):
        if not self.queue:
            return None

        batch = self.queue[:self.batch_size]
        results = model.batch_inference(batch)
        self.queue = self.queue[self.batch_size:]

        return results
```

**Typical Savings:** 20-40% through batching

### Strategy 4: Caching and Memoization

**Query-Level Caching:**
```
Baseline: Same query asked 100 times → 100 inferences
With cache: First inference + 99 cache hits → 1 inference + 99 cache lookups
Savings: 99%
```

**Implementation:**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_agent_response(query: str, context: str) -> str:
    # Only recompute if query/context unique
    return agent.process(query, context)

# Monitor cache effectiveness
cache_info = get_agent_response.cache_info()
hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses)
print(f"Cache hit rate: {hit_rate:.1%}")
```

**Semantic Caching:**
```python
# Cache by query similarity, not exact match
def semantic_cache_lookup(query):
    # Find similar cached queries
    similar = vector_db.search(
        embedding(query),
        top_k=5,
        threshold=0.95
    )

    if similar:
        return cached_responses[similar[0]]
    else:
        # Compute and cache
        response = agent.process(query)
        vector_db.add(embedding(query), response)
        return response
```

**Typical Savings:** 30-60% with good caching strategy

### Strategy 5: Load-Based Scaling

**Right-Sizing Infrastructure:**

```
Peak demand: 1000 req/s
Average demand: 100 req/s
Off-hours demand: 10 req/s

Static Sizing (peak):
├─ 40 GPUs × $2/hour × 24h = $1,920/day
└─ Wasteful during off-peak

Dynamic Sizing (autoscale):
├─ Peak hours: 40 GPUs
├─ Average hours: 4 GPUs
├─ Off-peak: 1 GPU
└─ Weighted average: 12 GPUs → $240/day

Savings: 87.5% daily compute cost
```

**Kubernetes Autoscaling Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-autoscaler
spec:
  scaleTargetRef:
    kind: Deployment
    name: agent-deployment
  minReplicas: 1
  maxReplicas: 40
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50  # Scale down 50% at a time
```

**Typical Savings:** 50-70% through dynamic scaling

### Strategy 6: Spot Instances and Preemptible VMs

**Cost Reduction:**
```
On-demand GPU: $2/hour
Spot instance: $0.60/hour (70% savings!)

Tradeoff: Spot can be interrupted

Use for:
├─ Batch processing (can pause and resume)
├─ Non-critical services
├─ Development/testing
└─ Load spikes (burst capacity)

Don't use for:
├─ User-facing critical requests
├─ Long-running inference
└─ Time-sensitive operations
```

---

## Resource Monitoring and Optimization

### GPU Utilization Tracking

**Key Metrics:**

```python
def track_gpu_metrics():
    metrics = {
        "utilization_percentage": nvidia_smi.gpu_util,
        "memory_used_gb": nvidia_smi.memory_used,
        "memory_free_gb": nvidia_smi.memory_free,
        "power_watts": nvidia_smi.power_draw,
        "temperature_c": nvidia_smi.temperature,
        "clock_speed_mhz": nvidia_smi.clock_speed,
    }

    # Efficiency calculation
    if metrics["memory_used_gb"] > 0:
        efficiency = (
            metrics["utilization_percentage"] *
            metrics["memory_used_gb"]
        ) / metrics["memory_free_gb"]
    else:
        efficiency = 0

    return metrics, efficiency
```

**Target Utilization:**
```
Optimal Range: 75-85% GPU utilization
├─ Below 50%: Opportunity for right-sizing
├─ 50-75%: Reasonable utilization
├─ 75-85%: Efficient use (good range)
├─ 85-95%: Running hot, monitor closely
└─ >95%: Risk of throttling/failures

Thermal Target: 70-80°C
└─ Above 85°C: Thermal throttling likely
```

### Memory Optimization

**Memory Profiling:**
```python
def analyze_memory_usage(model, batch_size):
    import tracemalloc

    tracemalloc.start()

    # Run inference
    with torch.no_grad():
        output = model(sample_batch)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Current memory: {current / 1024**3:.1f} GB")
    print(f"Peak memory: {peak / 1024**3:.1f} GB")
```

**Memory Reduction Techniques:**
```
Technique                   Savings     Implementation
======================================================
Mixed precision (FP16)     30-50%      torch.autocast
Gradient checkpointing     25-40%      Enable for training
Quantization (INT8/INT4)   50-75%      TensorRT
Model distillation         40-60%      Train smaller model
Attention optimization     10-20%      Flash Attention
```

### CPU and Host Memory Monitoring

```python
def monitor_host_resources():
    import psutil

    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "network_io": psutil.net_io_counters(),
    }

    # Alert if approaching limits
    if metrics["memory_percent"] > 85:
        alert("Host memory usage critical")

    if metrics["disk_percent"] > 90:
        alert("Disk space critically low")

    return metrics
```

---

## Cost Dashboards and Reporting

### Daily Cost Dashboard

Shows:
- Total daily cost
- Cost by customer/feature
- Cost per request
- Cost trend (vs. yesterday, vs. baseline)
- Projected monthly cost
- Top cost drivers

### Weekly Cost Review

```
Week of Nov 1-7:
├─ Total cost: $4,850
├─ vs. Budget: +8% (investigate)
├─ vs. Last week: +15% (traffic up)
├─ Cost per request: $0.0045
├─ Top customer: Customer A ($1,200)
└─ Top feature: RAG queries (40% of cost)

Recommendations:
├─ Cache more RAG responses
├─ Optimize Customer A's prompts
└─ Consider model downgrade for simple queries
```

### Monthly Cost Analysis

```
October Summary:
├─ Compute: $12,000 (60%)
├─ APIs: $5,000 (25%)
├─ Storage: $2,000 (10%)
├─ Operations: $1,000 (5%)
│
├─ vs. Budget: -5% (under budget)
├─ vs. Last month: +20% (seasonal)
└─ Optimization opportunities: -$3,000 potential
```

---

## Cost Allocation and Chargeback

### Multi-Tenant Cost Tracking

```python
class CostTracker:
    def log_request(self, request_id, customer_id, metrics):
        cost = (
            metrics['gpu_hours'] * GPU_RATE +
            metrics['tokens'] * TOKEN_RATE +
            metrics['api_calls'] * API_RATE
        )

        # Store by customer
        self.customer_costs[customer_id] += cost

        # Store by feature
        feature = metrics['agent_type']
        self.feature_costs[feature] += cost

        return cost

    def generate_customer_invoice(self, customer_id, month):
        total = self.customer_costs.get(customer_id, 0)
        markup = total * 0.20  # 20% markup
        return {
            "customer_id": customer_id,
            "base_cost": total,
            "markup": markup,
            "total": total + markup
        }
```

### Budget Management

```python
def enforce_budget(customer_id, monthly_budget):
    current_cost = get_customer_cost_ytd(customer_id)
    utilization = current_cost / monthly_budget

    if utilization > 0.95:
        # Hard limit at 95%
        reject_requests(customer_id)
        alert(f"Customer {customer_id} approaching budget")

    elif utilization > 0.80:
        # Warn at 80%
        warn(f"Customer {customer_id} at 80% of budget")

    return utilization
```

---

## Best Practices

### Cost Management

- [ ] Track costs at multiple levels (request, customer, feature)
- [ ] Set budgets and enforce limits
- [ ] Review costs weekly
- [ ] Optimize high-cost operations
- [ ] Plan for seasonal variations

### Resource Optimization

- [ ] Monitor GPU/CPU utilization continuously
- [ ] Right-size infrastructure for typical load
- [ ] Implement aggressive caching
- [ ] Use autoscaling for peak handling
- [ ] Regular performance profiling

### Cost-Quality Balance

- [ ] Don't sacrifice quality for cost
- [ ] Target sweet spot: 75-85% GPU utilization
- [ ] Optimize prompts iteratively
- [ ] A/B test model variants
- [ ] Monitor cost vs. user satisfaction

---

## Common Issues and Solutions

### Issue 1: Unexpectedly High Costs

**Diagnosis:**
- Check for cache misses
- Analyze token usage trends
- Review recent model changes
- Check for infinite loops/retries

**Solutions:**
- Improve caching strategy
- Optimize prompts
- Implement request timeouts
- Implement rate limiting

### Issue 2: Low GPU Utilization

**Diagnosis:**
- Check request arrival rate
- Monitor batch sizes
- Check for I/O bottlenecks
- Review scheduling efficiency

**Solutions:**
- Implement batching
- Optimize I/O paths
- Consider right-sizing down
- Combine workloads

### Issue 3: Budget Overruns

**Diagnosis:**
- Track cost by customer
- Identify high-cost features
- Review pricing assumptions
- Check for traffic spikes

**Solutions:**
- Enforce rate limiting
- Optimize problematic features
- Renegotiate pricing
- Plan capacity more conservatively

---

## References

- **Infrastructure Deployment:** See Chapter 7/04-NVIDIA-NIM-Deployment.md
- **Performance Optimization:** See Chapter 7/05-Mastering-LLM-Inference-Optimization.md
- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md

---

## Conclusion

Cost optimization and resource monitoring are essential for sustainable production systems. By systematically tracking costs, implementing optimization strategies, and maintaining efficient resource utilization, organizations ensure their agent systems remain economically viable while maintaining quality.

**Key Principle:** Measure costs, optimize iteratively, balance quality and efficiency.
