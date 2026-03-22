# Risk Assessment and Management for AI Systems

**Source:** ISO 31000 risk management, NIST AI RMF, safety engineering best practices

**Focus:** Systematically identifying and managing risks in agent systems
**Scope:** Risk identification, assessment, mitigation, monitoring, residual risk acceptance

---

## Risk Assessment Framework

### Phase 1: Risk Identification

**Identify Potential Risks:**

```
Risk Categories for AI Systems:
├─ Safety Risks
│  ├─ System failures causing harm
│  ├─ Incorrect predictions causing injury
│  ├─ Unintended consequences
│  └─ Edge case failures
│
├─ Performance Risks
│  ├─ Model degradation over time
│  ├─ Accuracy below acceptable level
│  ├─ Latency exceeding SLA
│  └─ System unavailability
│
├─ Security Risks
│  ├─ Unauthorized access
│  ├─ Data breach
│  ├─ Model poisoning
│  └─ Adversarial attacks
│
├─ Bias & Fairness Risks
│  ├─ Demographic disparities
│  ├─ Discrimination in decisions
│  ├─ Amplification of historical bias
│  └─ Representation issues
│
├─ Privacy Risks
│  ├─ Personal data exposure
│  ├─ Unauthorized use of data
│  ├─ Unintended data collection
│  └─ Data retention violations
│
├─ Regulatory Risks
│  ├─ Non-compliance with laws
│  ├─ Regulatory penalties
│  ├─ License revocation
│  └─ Legal liability
│
└─ Reputational Risks
   ├─ Public controversy
   ├─ Loss of user trust
   ├─ Negative media coverage
   └─ Brand damage
```

**Risk Identification Methods:**
```python
def identify_risks(system):
    """Systematic risk identification"""
    risks = []

    # Brainstorming with diverse team
    risks.extend(brainstorm_with_team(system))

    # Review historical incidents
    risks.extend(learn_from_similar_systems())

    # Stakeholder interviews
    risks.extend(interview_stakeholders())

    # Threat modeling
    risks.extend(threat_modeling(system))

    # Review regulatory requirements
    risks.extend(identify_regulatory_gaps())

    # Remove duplicates and consolidate
    return deduplicate_risks(risks)
```

### Phase 2: Risk Assessment

**Assess Probability and Impact:**

```
Risk Matrix:
               Unlikely    Possible    Likely      Almost Certain
               (5-15%)     (15-40%)    (40-75%)    (75-95%)
────────────────────────────────────────────────────────────────
Negligible     LOW         LOW         LOW         MEDIUM
(No impact)

Minor          LOW         LOW         MEDIUM      MEDIUM
(Limited)

Moderate       LOW         MEDIUM      MEDIUM      HIGH
(Significant)

Major          MEDIUM      MEDIUM      HIGH        HIGH
(Severe)

Critical       MEDIUM      HIGH        HIGH        CRITICAL
(Catastrophic)
```

**Risk Scoring:**
```python
def assess_risk(risk_description):
    """Quantify risk level"""
    # Estimate probability (0-1)
    probability = estimate_probability(risk_description)

    # Estimate impact magnitude (0-1)
    impact = estimate_impact(risk_description)

    # Calculate risk score
    risk_score = probability * impact

    # Determine risk level
    if risk_score > 0.25:
        return "CRITICAL"
    elif risk_score > 0.15:
        return "HIGH"
    elif risk_score > 0.06:
        return "MEDIUM"
    else:
        return "LOW"
```

### Phase 3: Risk Mitigation

**Design Controls:**

```
Mitigation Strategies (ordered by effectiveness):

1. Eliminate
   └─ Remove the risk entirely (redesign)

2. Reduce
   ├─ Reduce probability (better testing, training)
   └─ Reduce impact (safety features, limits)

3. Transfer
   └─ Shift to insurance/third party

4. Accept
   └─ Accept residual risk, monitor closely
```

**Mitigation Planning:**
```python
def mitigation_plan(risk):
    """Create mitigation strategy"""
    return {
        "risk": risk.description,
        "current_score": risk.current_score,
        "target_score": 0.05,  # < 5% acceptable
        "mitigation": [
            {
                "control": "Implement input validation",
                "reduces_probability": 0.3,  # 30% reduction
                "timeline": "2 weeks",
                "owner": "Engineering",
            },
            {
                "control": "Add human review gate",
                "reduces_impact": 0.5,  # 50% reduction
                "timeline": "1 week",
                "owner": "Operations",
            },
        ],
        "residual_score": 0.04,  # After mitigations
        "acceptance": "Accepted",
        "review_date": "Q3 2024",
    }
```

---

## Risk Register and Tracking

### Risk Register Format

```yaml
Risk Register for [System Name]
Updated: [Date]

Risk 1: Model Accuracy Degradation
─────────────────────────────────
ID: PERF-001
Category: Performance
Description: Model accuracy decreases over time as data distribution shifts
Current Probability: High (60%)
Current Impact: Major
Current Risk Score: 0.60 (HIGH)

Mitigation:
├─ Implement drift detection (reduces prob by 40%)
├─ Automated retraining pipeline (reduces prob by 20%)
├─ Monitoring and alerts (reduces impact by 30%)
└─ Manual review process (reduces impact by 20%)

Residual Risk Score: 0.15 (MEDIUM)
Owner: ML Team
Status: In Progress
Target Resolution: Q4 2024

Risk 2: Privacy Data Breach
──────────────────────────
ID: PRIV-002
Category: Privacy
Description: User personal data exposed due to security vulnerability
Current Probability: Medium (20%)
Current Impact: Critical
Current Risk Score: 0.20 (HIGH)

Mitigation:
├─ Encryption at rest (AES-256)
├─ Encryption in transit (TLS 1.3)
├─ Access controls (least privilege)
├─ Security monitoring
└─ Incident response plan

Residual Risk Score: 0.02 (LOW)
Owner: Security Team
Status: Complete
Monitoring: Continuous
```

### Risk Tracking Dashboard

```
Risk Management Dashboard
==========================

Total Risks: 15
├─ CRITICAL: 0
├─ HIGH: 3
├─ MEDIUM: 7
└─ LOW: 5

Trends:
├─ New risks this month: 1
├─ Resolved risks: 2
├─ Escalated risks: 0
└─ Overall trend: Improving ↓

Top Risks (by score):
1. Model Degradation (0.15) - In Progress
2. Data Privacy (0.12) - In Progress
3. System Unavailability (0.10) - Monitoring

Mitigation Progress:
├─ Planned: 5 controls
├─ In Progress: 3 controls
├─ Complete: 7 controls
└─ On Schedule: Yes
```

---

## Risk Monitoring

### Continuous Risk Monitoring

```python
class RiskMonitor:
    def __init__(self, risk_register):
        self.risks = risk_register
        self.monitoring_interval = 3600  # Every hour

    def monitor_risks(self):
        """Continuously assess risk status"""
        for risk in self.risks:
            # Update probability based on new data
            new_probability = self.estimate_current_probability(risk)

            # Update impact based on environment changes
            new_impact = self.estimate_current_impact(risk)

            # Recalculate risk score
            new_score = new_probability * new_impact

            # Compare to threshold
            if new_score > risk.threshold:
                self.escalate_risk(risk, new_score)

            # Update risk record
            self.update_risk_record(risk, new_score)

    def estimate_current_probability(self, risk):
        """Update probability estimate"""
        if risk.category == "model_degradation":
            # Check if drift detected
            drift_detected = self.check_for_drift()
            return 0.7 if drift_detected else 0.3

        elif risk.category == "security_breach":
            # Check vulnerability scan results
            vuln_score = self.get_vulnerability_score()
            return min(0.9, vuln_score)

        return risk.current_probability
```

### Risk Monitoring Metrics

```
For Each Risk, Monitor:

1. Probability Indicators
   ├─ Detection of root cause
   ├─ Status of mitigating controls
   ├─ Historical trend data
   └─ Early warning signals

2. Impact Indicators
   ├─ Potential affected users
   ├─ Potential financial loss
   ├─ Regulatory implications
   └─ Reputational impact

3. Control Effectiveness
   ├─ Is control working?
   ├─ Coverage percentage
   ├─ Design gaps
   └─ Implementation gaps

4. Residual Risk
   ├─ Current vs target
   ├─ Acceptable?
   ├─ Trend
   └─ Review date
```

---

## Risk Escalation and Acceptance

### Escalation Criteria

```
Risk Escalates to Executive When:

├─ CRITICAL risks detected (score > 0.30)
├─ Multiple HIGH risks (3+)
├─ Trend worsening (probability increasing)
├─ Control effectiveness declining
├─ Regulatory requirements tightening
└─ Unplanned risk emergence
```

### Risk Acceptance Decision

**Risk Acceptance Process:**

```
1. Risk Identified and Assessed
   └─ Probability and impact quantified
   └─ Mitigation options evaluated
   └─ Residual risk calculated

2. Stakeholder Review
   └─ Technical team reviews
   └─ Compliance team reviews
   └─ Business team reviews

3. Decision Gate
   └─ Is residual risk acceptable?
   └─ Can we accept monitoring burden?
   └─ Does business value justify?

4. Formal Acceptance
   └─ Documented decision
   └─ Signature approval
   └─ Conditions stated
   └─ Review date set

5. Active Management
   └─ Continuous monitoring
   └─ Regular reporting
   └─ Mitigation track
   └─ Re-assessment at review date
```

**Risk Acceptance Record:**
```yaml
Risk Acceptance Form

Risk: Model Accuracy Below Threshold
Risk ID: PERF-001
Residual Risk Score: 0.18

Business Justification:
└─ System too new to retrain
└─ Benefit of deployment outweighs risk
└─ Monitoring in place

Monitoring Plan:
├─ Weekly accuracy reports
├─ Monthly retraining evaluation
├─ Real-time drift monitoring
└─ User feedback tracking

Conditions:
├─ Human review gate required
├─ User satisfaction >4.0/5.0
├─ Error rate <2%
└─ Review in 6 months

Approval:
├─ Engineering Lead: [Signature]
├─ Product Manager: [Signature]
├─ Chief Risk Officer: [Signature]
└─ CEO: [Signature]

Review Date: [6 months from now]
```

---

## Best Practices

### Risk Management Program
- [ ] Formal risk assessment process
- [ ] Documented risk register
- [ ] Regular risk reviews
- [ ] Clear escalation procedures
- [ ] Executive sponsorship
- [ ] Integration with other processes

### Risk Assessment
- [ ] Diverse team perspectives
- [ ] Quantified probability/impact
- [ ] Multiple assessment methods
- [ ] Regular updates
- [ ] Root cause analysis
- [ ] Mitigation validation

### Monitoring and Reporting
- [ ] Automated monitoring where possible
- [ ] Regular risk dashboards
- [ ] Escalation alerts
- [ ] Monthly reporting
- [ ] Trend analysis
- [ ] Lessons learned

---

## References

- **Safety Frameworks:** See Chapter 9/01-AI-Safety-Frameworks.md
- **Compliance:** See Chapter 9/03-Regulatory-Compliance-Frameworks.md
- **Incident Response:** See Chapter 8/06-Error-Troubleshooting-Incident-Response.md

---

## Conclusion

Effective risk management requires systematic identification, assessment, and monitoring of risks throughout the system lifecycle. By maintaining a comprehensive risk register, implementing targeted mitigations, and continuously monitoring risk status, organizations proactively manage uncertainties and maintain acceptable risk levels.

**Core Principle:** Know your risks, manage them actively, accept them consciously.
