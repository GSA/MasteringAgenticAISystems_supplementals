# Data Quality and Drift Detection for Agent Systems

**Source:** Production ML best practices and NVIDIA monitoring guidelines

**Focus:** Detecting data issues that affect agent performance
**Scope:** Quality checks, drift detection, anomaly detection, root cause analysis

---

## Data Quality Dimensions for Agents

### Input Quality Metrics

**User Query Characteristics**
- Query length distribution
- Language diversity (multilingual?)
- Query complexity scores
- Topic distribution

**Example Tracking:**
```
Metric: Query Length
Baseline: Mean 45 tokens, 95th percentile 150 tokens
Alert Condition: Mean increases to 80+ tokens (30% spike)
Action: May indicate different user population or prompt engineering
```

### Context Quality Metrics

**Retrieved Documents (RAG Systems)**
- Document relevance to query
- Document freshness/staleness
- Number of chunks retrieved
- Citation completeness

**Knowledge Base Health:**
```
Metric: Retrieval Quality
Baseline: Top-1 relevance score >0.85
Current: Average 0.62
Impact: Agent cannot find good sources
Action: Investigate knowledge base updates or query changes
```

**Tool Outputs**
- API response validity
- Data completeness
- Unexpected null values
- Rate limiting or throttling

### Agent State Quality

**Memory Consistency**
- Previous context matches current state
- Tool results properly integrated
- No contradictions in reasoning trace

**Conversation Flow**
- Coherence of multi-turn interactions
- Consistency in entity references
- Logical flow of reasoning

---

## Distribution Drift Detection

### Type 1: Feature Distribution Drift

**Detecting Query Shift:**

```python
from scipy.stats import ks_2samp

def detect_query_drift(baseline_queries, current_queries):
    # Extract query embedding statistics
    baseline_lengths = [len(q.split()) for q in baseline_queries]
    current_lengths = [len(q.split()) for q in current_queries]

    # Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(baseline_lengths, current_lengths)

    if p_value < 0.05:
        print(f"Drift detected! p-value: {p_value}")
        print(f"Baseline mean: {np.mean(baseline_lengths)}")
        print(f"Current mean: {np.mean(current_lengths)}")
```

**What This Detects:**
- Changes in user population
- Seasonal variations
- Bot/spam increase
- Language/domain shifts

**Alert Threshold:**
- Statistical significance (p < 0.05)
- Practical significance: >20% mean shift
- Visual inspection of distribution plots

### Type 2: Prediction Distribution Drift

**Agent Output Changes:**

```
Metric: Agent Decision Distribution
Baseline (Jan):    60% TaskA, 25% TaskB, 15% TaskC
Current (Feb):     45% TaskA, 40% TaskB, 15% TaskC

Interpretation: Significant shift from TaskA to TaskB
Cause: Likely change in user needs or prompts
Action: Investigate what changed, retrain if intentional
```

**Detection:**
- Jensen-Shannon divergence for categorical distributions
- Track proportions over time
- Flag >15% shift in any category

### Type 3: Concept Drift

**Performance Degradation Over Time:**

```
Week 1: Success rate 96%
Week 2: Success rate 95%
Week 3: Success rate 93%
Week 4: Success rate 91%

Pattern: Steady decline indicates concept drift
Not sudden failure = suggests gradual environmental change
Action: Retrain with recent data or adjust thresholds
```

**Causes:**
- User behavior evolution
- Business logic changes
- Tool API modifications
- Market shifts

---

## Data Quality Checks

### Check 1: Structural Validation

**Schema Verification:**
```
Input: User Query Object
├─ user_id: string (required)
├─ query_text: string (required, length 1-5000)
├─ timestamp: ISO-8601 datetime
├─ conversation_id: uuid (optional)
└─ context: dict (optional)

Validation:
✓ user_id present and non-empty
✓ query_text between 1-5000 chars
✓ timestamp valid ISO-8601
✓ conversation_id valid UUID if present
```

**Alert Conditions:**
- >5% records with missing required fields
- >2% records with invalid types
- >1% records with constraint violations

### Check 2: Value Range Validation

**Detecting Out-of-Range Values:**

```python
def validate_ranges(current_data, baseline_stats):
    issues = []

    # Query length
    if mean(current_data.query_lengths) > baseline_stats.q_length_99th:
        issues.append("Query length abnormally high")

    # Latency (response time)
    if mean(current_data.latencies) > baseline_stats.latency_p99 * 1.5:
        issues.append("Latency spike detected")

    # Confidence scores
    if mean(current_data.confidence) < baseline_stats.confidence_mean - 2*std:
        issues.append("Confidence significantly lower")

    return issues
```

**Common Ranges:**
- Query length: 5-500 tokens (agent-dependent)
- Response time: 0.5-30 seconds
- Confidence score: 0.0-1.0
- Token count per request: 100-8000

### Check 3: Anomaly Detection

**Statistical Approach - Isolation Forest:**

```python
from sklearn.ensemble import IsolationForest

def detect_anomalies(data, contamination=0.01):
    # Feature engineering
    features = [
        'query_length',
        'response_latency',
        'num_tool_calls',
        'confidence_score',
        'tokens_generated'
    ]

    X = data[features].values

    # Train anomaly detector
    model = IsolationForest(contamination=contamination)
    predictions = model.fit_predict(X)

    anomalies = data[predictions == -1]
    return anomalies
```

**Use Cases:**
- Identify unusual queries (injections, adversarial inputs)
- Detect performance anomalies
- Find statistical outliers

### Check 4: Freshness Validation

**For Time-Dependent Data:**

```
Knowledge Base Freshness:
├─ Articles updated: Last 24 hours
├─ Prices current: Last 1 hour
├─ Weather data: Last 30 minutes
└─ Stock prices: Real-time

Alert if:
├─ Articles not updated for 7+ days
├─ Prices older than 1 day
├─ Weather data older than 2 hours
└─ Stock prices delayed >5 minutes
```

**Impact:**
- Stale data → incorrect agent decisions
- Outdated context → hallucinations
- Wrong prices → financial errors

---

## Monitoring Retrieval Quality (RAG Systems)

### Retrieval Metrics

**Relevance Scoring:**
```
For each query:
├─ Top-1 document relevance
├─ Top-5 document average relevance
├─ Document diversity (avoid duplicates)
└─ Citation coverage of agent response

Target Metrics:
├─ Top-1 relevance >0.85
├─ Top-5 average >0.70
├─ Diversity score >0.6
└─ 80%+ of response supported by retrieval
```

**Monitoring:**
```python
def assess_retrieval_quality(query, retrieved_docs, response):
    metrics = {
        "top1_relevance": compute_relevance(query, retrieved_docs[0]),
        "top5_avg": mean([
            compute_relevance(query, doc)
            for doc in retrieved_docs[:5]
        ]),
        "diversity": compute_diversity(retrieved_docs),
        "citation_coverage": count_cited_sources(response) / len(retrieved_docs)
    }

    # Alert if degradation
    if metrics["top1_relevance"] < 0.75:
        alert("Retrieval quality degraded")

    return metrics
```

### Knowledge Base Health

**Document Statistics:**
```
Total documents: 50,000
├─ By freshness:
│  ├─ Updated today: 2,400
│  ├─ Updated this week: 8,200
│  └─ Older than month: 39,400
│
├─ By quality:
│  ├─ High quality: 35,000
│  ├─ Medium quality: 12,000
│  └─ Low quality: 3,000
│
└─ Indexing health:
   ├─ Properly indexed: 49,800
   ├─ Missing embeddings: 150
   └─ Duplicate entries: 50
```

**Maintenance Actions:**
- Remove or update stale documents
- Improve low-quality documents
- Fix missing embeddings
- Deduplicate corpus

---

## Tool Integration Monitoring

### API Health Tracking

**For External Tool Calls:**

```python
def monitor_tool_health(tool_name, recent_calls):
    metrics = {
        "success_rate": success_count / len(recent_calls),
        "avg_latency": mean([c.duration for c in recent_calls]),
        "error_rate": error_count / len(recent_calls),
        "rate_limit_hits": count_rate_limits(recent_calls),
        "timeout_rate": count_timeouts(recent_calls)
    }

    # Alert conditions
    if metrics["success_rate"] < 0.95:
        alert(f"{tool_name}: Success rate {metrics['success_rate']:.1%}")

    if metrics["avg_latency"] > 5000:  # 5 seconds
        alert(f"{tool_name}: Latency increased to {metrics['avg_latency']:.0f}ms")

    if metrics["rate_limit_hits"] > 10:
        alert(f"{tool_name}: Rate limit being hit frequently")

    return metrics
```

**Baseline Expectations:**
- Success rate: >98% for critical tools
- Latency: p95 <2 seconds
- Rate limit incidents: <5/hour
- Timeout rate: <1%

---

## Drift Response Procedures

### Tier 1: Monitor and Log

**Conditions:**
- Small distribution shifts (<10%)
- Slight performance dips (<2%)
- Isolated quality issues

**Action:**
- Log the anomaly
- Track in monitoring dashboard
- Review in weekly analysis

### Tier 2: Investigate and Adjust

**Conditions:**
- Moderate shifts (10-25%)
- Performance dip (2-5%)
- Consistent pattern

**Action:**
1. Analyze root cause
2. Determine if intentional change
3. Adjust thresholds if appropriate
4. Consider fine-tuning dataset

### Tier 3: Immediate Action

**Conditions:**
- Large shifts (>25%)
- Major performance drop (>5%)
- System health at risk

**Action:**
1. Page on-call engineer
2. Investigate immediately
3. Prepare rollback plan
4. Consider reverting changes
5. Retrain emergency model

---

## Root Cause Analysis Framework

### 5-Step Debugging Process

**Step 1: Confirm the Issue**
```
✓ Is the alert real or false positive?
✓ What metric changed and by how much?
✓ When did it start?
✓ Is it still occurring?
```

**Step 2: Isolate the Change**
```
Recent changes to investigate:
├─ Code deployments
├─ Model updates
├─ Prompt changes
├─ Data pipeline modifications
├─ External system changes
└─ Input distribution shifts
```

**Step 3: Analyze Affected Traces**
```
Compare recent traces with baseline:
├─ What's different in the input?
├─ How did agent decisions change?
├─ Which tool calls failed?
├─ What was the output quality?
```

**Step 4: Test Hypotheses**
```
Test in isolated environment:
├─ Revert last code change
├─ Use previous model version
├─ Test with baseline data
├─ Modify suspected parameters
```

**Step 5: Implement Fix**
```
Action plan:
├─ Implement verified fix
├─ Test in staging environment
├─ Monitor closely on rollout
├─ Have rollback ready
└─ Post-incident review
```

---

## Quality Dashboards

### Data Quality Dashboard

Shows:
- Validation pass rates by check type
- Data anomaly trends
- Distribution drift metrics
- Tool health status
- Alert history

### Drift Detection Dashboard

Shows:
- Distribution comparisons (baseline vs. current)
- Drift metrics by feature
- Concept drift progression
- Recovery status
- Correlation with performance

### Impact Dashboard

Shows:
- Correlation between data quality and agent success
- Impact of detected issues on business metrics
- Cost of quality issues
- ROI of quality monitoring

---

## Best Practices

### Quality Monitoring

- [ ] Define quality checks upfront
- [ ] Set appropriate alert thresholds
- [ ] Monitor at multiple time scales
- [ ] Correlate metrics with performance
- [ ] Regular threshold review

### Drift Handling

- [ ] Establish drift response procedures
- [ ] Test drift detection on historical data
- [ ] Monitor multiple drift signals
- [ ] Implement automated retrain triggers
- [ ] Track drift causes and patterns

### Continuous Improvement

- [ ] Track false positive rate
- [ ] Adjust thresholds based on experience
- [ ] Add checks for newly discovered issues
- [ ] Document quality issues encountered
- [ ] Share learnings across team

---

## References

- **ML Monitoring in Production:** See Chapter 8/02-ML-Monitoring-Production.md
- **Inference Optimization:** See Chapter 7/05-Mastering-LLM-Inference-Optimization.md
- **Agent Architecture:** See Chapters 1-3

---

## Conclusion

Comprehensive data quality and drift detection systems enable rapid identification and response to data-driven issues affecting agent performance. By monitoring at multiple levels—structural, statistical, and semantic—teams maintain high-quality data and catch problems before they impact users.

**Core Principle:** Detect drift early, respond fast, improve continuously.
