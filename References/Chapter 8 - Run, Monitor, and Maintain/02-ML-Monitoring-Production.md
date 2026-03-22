# Machine Learning Monitoring in Production

**Source:** https://www.evidentlyai.com/ and industry best practices

**Focus:** Monitoring strategies for production ML/AI systems
**Scope:** Data quality, drift, performance, and observability

---

## Why Monitoring Matters

Production ML systems face unique challenges:

**Silent Failures:**
- Models produce predictions even with unreliable inputs
- No system errors or exceptions raised
- Quality degrades without alerting

**Delayed Ground Truth:**
- Labels arrive late (days, weeks)
- Cannot immediately assess accuracy
- Must rely on proxy metrics

**Concept Drift:**
- Data patterns change over time
- Model performance degrades gradually
- Requires continuous monitoring

**Data Distribution Shifts:**
- Input features change statistically
- Model sees data it wasn't trained on
- Performance varies by segment

---

## Key Monitoring Metrics

### Model Quality Metrics

**Classification:**
- Accuracy
- Precision / Recall
- F1-Score
- ROC-AUC
- Per-class performance

**Regression:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

**Performance by Segment:**
- Performance for different customer cohorts
- Performance by data ranges
- Performance by time periods

### Data Drift Metrics

**Feature Distribution Drift:**
- Detect changes in input distributions
- Compare against baseline
- Statistical tests or distance metrics

**Prediction Drift:**
- Track prediction distributions
- Detect if model is making different predictions
- Indicates potential issue

**Label Drift:**
- Monitor distribution of ground truth
- Indicates changing environment
- May require model retraining

### Data Quality Metrics

**Checks:**
- Missing value percentages
- Schema validation
- Feature range constraints
- Type correctness
- Outlier detection

**Issues:**
- Unexpected missing data
- Corrupt values
- Type mismatches
- Out-of-range values
- Format errors

### Bias & Fairness

**Predictive Parity:**
- Equal performance across demographic groups
- Ensure fair treatment

**Equalized Odds:**
- Equal true positive and false positive rates
- Prevent discriminatory outcomes

---

## Monitoring Architecture

### Batch Monitoring

**Approach:**
- Collect predictions in batch
- Run monitoring jobs (hourly, daily)
- Generate reports
- Alert on anomalies

**When to Use:**
- Non-real-time applications
- Cost-sensitive monitoring
- Historical analysis

### Real-Time Streaming

**Approach:**
- Monitor each prediction immediately
- Stream metrics to monitoring system
- Alert on anomalies in real-time
- Track metrics continuously

**When to Use:**
- Critical applications
- Real-time alerts needed
- High-volume systems

### Hybrid Approach

**Combination:**
- Real-time basic metrics
- Batch deep analysis
- Best of both

---

## Monitoring Strategy Framework

### Step 1: Define Objectives

**Identify:**
- Key stakeholders
- Business-critical metrics
- Risk tolerance
- SLA requirements

### Step 2: Select Metrics

**Choose:**
- Primary metrics (directly measure quality)
- Proxy metrics (when ground truth delayed)
- Leading indicators (predict future issues)
- Business metrics (revenue, satisfaction)

### Step 3: Choose Reference Dataset

**Baseline:**
- Use stable historical data
- Represents good performance
- Used for drift comparison
- Should span variability

### Step 4: Pick Architecture

**Decide:**
- Batch vs. streaming
- Tools and platforms
- Storage requirements
- Alert mechanisms

### Step 5: Set Alert Thresholds

**Define:**
- Absolute thresholds
- Relative changes
- Statistical significance
- Alert severity levels

---

## Special Considerations

### For Large Language Models

**Challenges:**
- Direct accuracy evaluation difficult
- Qualitative outputs (generation)
- Variable correctness criteria
- Expensive evaluation

**Solutions:**
- LLM-as-a-judge evaluation
- User satisfaction surveys
- Hallucination detection
- Semantic similarity metrics
- Token-level analysis

### For Retrieval-Augmented Generation (RAG)

**Monitor:**
- Retrieval quality (relevance of documents)
- Generation quality (appropriate responses)
- Hallucination rate (factual accuracy)
- Citation correctness

### For Agentic Systems

**Monitor:**
- Agent decision correctness
- Tool calling accuracy
- Tool parameter correctness
- Interaction success rate
- Goal achievement

### For Critical Applications

**Examples:** Finance, healthcare, autonomous systems

**Requirements:**
- Rigorous monitoring
- High alerting sensitivity
- Manual review of critical decisions
- Compliance tracking
- Explainability validation

### For Non-Critical Applications

**Examples:** Recommendation systems, content ranking

**Approach:**
- Simpler monitoring
- Less frequent evaluation
- Cost-conscious methods
- Best-effort optimization

---

## Data Quality Checks

### Missing Values
```
Alert if: missing_percentage > 5%
Action: Investigate data pipeline
```

### Schema Validation
```
Alert if: type mismatch detected
Action: Review data transformation
```

### Range Constraints
```
Alert if: feature_value > max_expected
Action: Check for data corruption
```

### Outlier Detection
```
Alert if: outlier_percentage > threshold
Action: Investigate source
```

---

## Common Issues & Solutions

### Issue 1: Data Drift

**Symptom:** Feature distributions changed significantly

**Diagnosis:**
- Compare current vs. baseline distributions
- Run statistical tests
- Identify specific features

**Solutions:**
- Retrain model
- Implement correction layer
- Collect more data
- Adjust alert thresholds

### Issue 2: Concept Drift

**Symptom:** Model performance degrading

**Diagnosis:**
- Ground truth arriving
- Compare recent accuracy to baseline
- Analyze error patterns

**Solutions:**
- Retrain model
- Update training data
- Switch to new model
- Implement ensemble

### Issue 3: Silent Failures

**Symptom:** Model predictions on corrupted data

**Diagnosis:**
- Quality checks fail
- But predictions still generated
- Users affected without alert

**Solutions:**
- Pre-prediction quality gates
- Validation before serving
- Reject bad predictions
- Alert immediately

---

## Tools for Monitoring

**Evidently AI:** ML observability platform
**Arize:** ML monitoring for enterprises
**Monte Carlo:** Data quality and observability
**Datadog:** General ML monitoring
**Custom Solutions:** Prometheus + Grafana

---

## Best Practices

### Design

- [ ] Start simple, add complexity as needed
- [ ] Choose relevant metrics
- [ ] Set realistic baselines
- [ ] Plan for delays in ground truth

### Implementation

- [ ] Automate metric collection
- [ ] Set up persistent storage
- [ ] Create dashboards
- [ ] Configure alerting

### Operations

- [ ] Review alerts regularly
- [ ] Tune thresholds based on experience
- [ ] Investigate anomalies thoroughly
- [ ] Document incidents

### Improvement

- [ ] Track monitoring effectiveness
- [ ] Improve metric selection
- [ ] Reduce false alerts
- [ ] Build evaluation dataset

---

## Integration with Agent Systems

For agentic AI specifically:

1. **Monitor agent decisions**
2. **Track tool calls and success**
3. **Measure goal achievement**
4. **Detect failure patterns**
5. **Evaluate response quality**

---

## Conclusion

Effective ML monitoring is essential for maintaining model quality in production. By systematically selecting metrics, implementing monitoring infrastructure, and responding to alerts, organizations ensure their AI systems remain accurate, reliable, and fair over time.

The key principle: **Measure systematically, alert intelligently, respond decisively.**
