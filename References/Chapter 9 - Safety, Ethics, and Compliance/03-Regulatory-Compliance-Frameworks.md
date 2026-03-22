# Regulatory Compliance Frameworks for AI Systems

**Source:** Government regulations, industry standards, NVIDIA compliance guidelines

**Focus:** Meeting legal and regulatory requirements for AI deployment
**Scope:** EU AI Act, FedRAMP, HIPAA, GDPR, industry-specific regulations

---

## Global Regulatory Landscape

### Region 1: European Union (EU AI Act)

**Status:** Enacted (2024), enforcement phased

**Risk-Based Classification:**

```
Prohibited Risk (Banned):
├─ Social scoring systems
├─ Cognitive impairment exploitation
├─ Subliminal manipulation
└─ Facial recognition in public (with exceptions)

High-Risk:
├─ Educational/vocational access decisions
├─ Employment decisions
├─ Critical infrastructure control
├─ Law enforcement (risk assessment, biometric ID)
├─ Immigration and asylum
├─ Facial recognition identification
└─ Eligibility for public services (benefits, loans)

Limited Risk:
└─ Chatbots and generative AI (transparency required)

Minimal Risk:
└─ Everything else
```

**High-Risk Requirements:**

```
If Your Agent is High-Risk, You Must:

1. Risk Assessment
   ├─ Document AI system and its risks
   ├─ Analyze impacts on rights/freedoms
   ├─ Identify risk mitigation measures
   └─ Regular reassessment

2. Technical Documentation
   ├─ Training and test data
   ├─ Algorithm logic and decisions
   ├─ Performance metrics (accuracy, bias)
   ├─ Limitations and failure modes
   └─ Human oversight mechanisms

3. Data Management
   ├─ Data is complete and accurate
   ├─ Data is representative
   ├─ Data is free of bias
   ├─ Data is properly labeled
   └─ Data access is controlled

4. Transparency and Information
   ├─ Disclose to stakeholders that AI is used
   ├─ Explain how decisions are made
   ├─ Describe intended purpose
   └─ Describe known limitations

5. Human Oversight
   ├─ Humans understand system
   ├─ Humans can intervene
   ├─ Humans can override decisions
   ├─ System supports human judgment
   └─ Humans are appropriately trained

6. Quality Assurance
   ├─ Continuous monitoring
   ├─ Testing in real conditions
   ├─ Logging and record-keeping
   └─ Incident management procedures

7. Record Keeping
   ├─ Maintain all required documentation
   ├─ Records available for regulators
   ├─ Maintain for 5-7 years
   └─ Access controls in place
```

**Penalties:**
```
Violation Class         Fine
────────────────────────────────────
Prohibited Risk         Up to €30M or 6% revenue
High-Risk Non-Compliance Up to €20M or 4% revenue
Transparency Violations  Up to €10M or 2% revenue
```

### Region 2: United States

**Federal Regulation:**
```
├─ No single comprehensive law
├─ Sector-specific regulations apply
├─ Executive Order on AI Governance
└─ Proposed regulations in development
```

**By Sector:**

```
Healthcare (FDA):
├─ AI used in medical diagnosis/treatment
├─ Pre-market review required
├─ Real-world performance monitoring
└─ Black-box algorithms require validation

Finance (SEC/Federal Reserve):
├─ AI in financial decisions (credit, pricing)
├─ Fair lending laws apply
├─ Model risk management required
└─ Vendor management/oversight

Government (NIST AI Risk Management):
├─ Risk management framework required
├─ Impact assessment mandatory
├─ Regular auditing
└─ Documentation required

Employment (EEOC):
├─ Discrimination laws apply to AI
├─ Adverse impact testing required
├─ Fairness validation mandatory
└─ Hiring decisions auditable
```

### Region 3: China

**Algorithm Governance Law:**
```
├─ Content recommendation systems regulated
├─ Algorithm audit by government
├─ Data collection restrictions
├─ User consent required
└─ Transparency requirements
```

**Data Security Law:**
```
├─ Sensitive data residency in China
├─ Government access to data
├─ Data minimization principles
└─ Export restrictions on AI models
```

---

## Key Regulatory Requirements

### Requirement 1: Impact Assessment

**Data Protection Impact Assessment (DPIA) for GDPR:**

```
1. System Description
   ├─ What is the AI system?
   ├─ What does it do?
   ├─ What data does it use?
   └─ Who can access it?

2. Necessity and Proportionality
   ├─ Why is this AI system necessary?
   ├─ Are the benefits worth the risks?
   └─ Are there less invasive alternatives?

3. Privacy Impact
   ├─ What personal data is processed?
   ├─ How is it collected?
   ├─ How long is it retained?
   └─ Who has access?

4. Risk Assessment
   ├─ What could go wrong?
   ├─ How likely is it?
   ├─ What would be the impact?
   ├─ Can we mitigate?
   └─ What's the residual risk?

5. Mitigation Measures
   ├─ Technical safeguards (encryption, access control)
   ├─ Procedural safeguards (audit, training)
   ├─ Organizational safeguards (governance, review)
   └─ Privacy by design principles

6. Conclusion
   ├─ Overall risk assessment
   ├─ Mitigation effectiveness
   ├─ Approval decision
   └─ Monitoring plan
```

### Requirement 2: Documentation and Record-Keeping

**Required Documentation:**

```
1. System Documentation
   ├─ Technical design document
   ├─ Algorithm description
   ├─ Training process
   ├─ Data specifications
   └─ Performance metrics

2. Risk and Compliance Documentation
   ├─ Risk assessment
   ├─ Impact assessment (DPIA, EU AI Act)
   ├─ Fairness/bias audit
   ├─ Compliance checklist
   └─ Exception log

3. Operational Documentation
   ├─ Monitoring procedures
   ├─ Incident procedures
   ├─ Human oversight procedures
   ├─ Update procedures
   └─ Support procedures

4. Training and Governance
   ├─ Team training records
   ├─ Ethics review documentation
   ├─ Board approval records
   └─ Stakeholder consultation log

Record Retention:
├─ Production systems: 5-7 years
├─ Incident documentation: Full lifecycle
├─ Regulatory requests: Indefinitely
└─ Decommissioned systems: 3 years post-decommission
```

### Requirement 3: Transparency and Disclosure

**User Disclosure Requirements:**

```
1. Basic Disclosure
   ├─ Inform user that AI is being used
   ├─ Explain its purpose
   ├─ Describe decision being made
   └─ Indicate confidence/reliability

2. Functional Transparency
   ├─ Explain how the AI works (at appropriate level)
   ├─ List factors considered
   ├─ Describe limitations
   └─ Provide examples

3. Data Transparency
   ├─ What data is collected
   ├─ How it will be used
   ├─ Who can access it
   ├─ How long it's retained
   └─ User rights over data

4. Algorithmic Accountability
   ├─ Who made this decision?
   ├─ Why was this decision made?
   ├─ Can I appeal?
   ├─ How do I appeal?
   └─ What happens if I appeal?
```

### Requirement 4: Bias and Fairness Testing

**Mandatory Testing:**

```
1. Development Phase
   ├─ Training data bias audit
   ├─ Feature importance analysis
   ├─ Prototype fairness testing
   └─ Limitation documentation

2. Deployment Phase
   ├─ Pre-deployment fairness audit
   ├─ Demographic parity testing
   ├─ Equalized odds testing
   ├─ Bias mitigation validation
   └─ Documentation

3. Operations Phase
   ├─ Continuous monitoring
   ├─ Quarterly fairness audits
   ├─ User complaint tracking
   ├─ Bias trend analysis
   └─ Regular reporting

Thresholds:
├─ Disparate Impact <80%: Investigation required
├─ Disparate Impact <70%: Remediation required
├─ False Positive Rate diff >10%: Review required
└─ Documentation: Always required
```

---

## Compliance by Use Case

### Use Case 1: Medical AI System

**Applicable Regulations:**
```
US:
├─ FDA regulation (Class II/III device likely)
├─ HIPAA for patient data
├─ State medical board oversight
└─ Medical malpractice law

EU:
├─ EU AI Act (High-Risk)
├─ GDPR for patient data
├─ Medical Device Regulation (MDR)
└─ Data Protection Directive
```

**Compliance Requirements:**

```
1. Regulatory Approval
   ├─ FDA 510(k) or PMA (US)
   ├─ CE marking (EU)
   └─ Clinical validation data

2. Clinical Evidence
   ├─ Clinical studies showing safety
   ├─ Clinical studies showing efficacy
   ├─ Comparison to standard care
   └─ Long-term monitoring data

3. Data Privacy
   ├─ HIPAA compliance (US)
   ├─ GDPR compliance (EU)
   ├─ Secure transmission (encryption)
   └─ Access controls

4. Transparency
   ├─ Doctor informed of AI involvement
   ├─ Patient informed of AI involvement
   ├─ Limitations disclosed
   └─ Appeal/override process documented

5. Monitoring
   ├─ Post-market surveillance
   ├─ Adverse event reporting
   ├─ Performance monitoring
   └─ Continuous validation
```

### Use Case 2: Financial Services AI

**Applicable Regulations:**
```
US:
├─ Fair Lending Act (ECOA, FHA)
├─ Equal Credit Opportunity Act
├─ Truth in Lending Act
├─ SEC oversight (if securities-related)
└─ State regulation varies

EU:
├─ EU AI Act (High-Risk)
├─ GDPR
├─ Banking Regulation
└─ Consumer Rights Directive
```

**Compliance Requirements:**

```
1. Fairness Testing
   ├─ FCRA Section 1022(b) compliance
   ├─ Disparate impact testing
   ├─ Regular fairness audits
   └─ Bias remediation

2. Explainability
   ├─ Adverse action notices explain why
   ├─ Feature importance disclosed
   ├─ Appeal process exists
   └─ Human review available

3. Data Compliance
   ├─ Consumer reporting agency compliance
   ├─ Data accuracy requirements
   ├─ Dispute resolution procedures
   └─ Data access rights

4. Model Risk Management
   ├─ Model governance framework
   ├─ Validation and monitoring
   ├─ Change management
   └─ Vendor oversight (if vendor model)
```

### Use Case 3: HR/Employment AI

**Applicable Regulations:**
```
US:
├─ Title VII of Civil Rights Act (1964)
├─ Age Discrimination in Employment Act
├─ Americans with Disabilities Act
├─ Equal Pay Act
├─ State-specific employment laws
└─ EEOC guidance on AI

EU:
├─ EU AI Act (High-Risk)
├─ GDPR
├─ Employment Equality Directives
└─ Data Protection directive
```

**Compliance Requirements:**

```
1. Bias Testing
   ├─ Disparate impact analysis
   ├─ Hiring rate by protected class
   ├─ Interview scoring fairness
   ├─ Promotion rate fairness
   └─ Documented testing

2. Transparency
   ├─ Applicant informed AI is used
   ├─ Explanation of decision
   ├─ Appeal/override process
   ├─ Human reviewer available
   └─ Feedback loop to improve

3. Records and Documentation
   ├─ Hiring records by demographic
   ├─ Model performance metrics
   ├─ Fairness audit reports
   ├─ Incident logs
   └─ Complaint resolution records

4. Impact Assessment
   ├─ Job relevance validation
   ├─ Validity evidence
   ├─ Predictive accuracy
   ├─ Limitations documentation
   └─ Risk mitigation
```

---

## Compliance Monitoring and Reporting

### Compliance Audit Process

```
1. Scope Definition
   ├─ What systems are in scope?
   ├─ What regulations apply?
   ├─ What documentation is required?
   └─ What metrics to assess?

2. Documentation Review
   ├─ Check all required docs exist
   ├─ Verify completeness
   ├─ Assess quality
   └─ Identify gaps

3. Technical Testing
   ├─ Verify requirements are met
   ├─ Test safeguards
   ├─ Validate monitoring
   └─ Check logging

4. Operational Testing
   ├─ Process compliance
   ├─ Human oversight
   ├─ Incident response
   └─ Training effectiveness

5. Report and Remediation
   ├─ Document findings
   ├─ Prioritize issues
   ├─ Create remediation plan
   ├─ Track completion
   └─ Schedule follow-up
```

### Compliance Reporting Template

```yaml
Compliance Report: [System Name]
Date: [YYYY-MM-DD]
Auditor: [Name]

---
Executive Summary
-----------------
System: [Description]
Regulations: [Applicable regulations]
Overall Status: [Compliant/Non-Compliant/Partial]

---
Detailed Findings
-----------------

Regulation 1: [Regulation Name]
Status: [Compliant/Non-Compliant]
Findings:
  ├─ Requirement A: [Status + evidence]
  ├─ Requirement B: [Status + evidence]
  └─ Requirement C: [Status + evidence]

---
Gap Analysis
------------
Gap 1: [Description]
  Severity: [Critical/High/Medium/Low]
  Remediation: [Action plan]
  Timeline: [When to fix]

---
Risk Assessment
---------------
Residual Risk: [High/Medium/Low]
Mitigation: [What we're doing about it]

---
Recommendations
----------------
1. [Priority action]
2. [Next priority]
```

---

## Compliance Checklist

### Before Deployment

- [ ] Identify applicable regulations
- [ ] Complete impact assessment (DPIA/EU AI Act)
- [ ] Bias and fairness audit
- [ ] Documentation complete and accurate
- [ ] Legal review completed
- [ ] Compliance sign-off obtained
- [ ] Transparency/disclosure prepared
- [ ] Appeals process documented

### During Operations

- [ ] Monitoring system active
- [ ] Regular fairness audits (quarterly)
- [ ] Incident logging and response
- [ ] User feedback collection
- [ ] Regulatory change monitoring
- [ ] Staff training current
- [ ] Documentation updates

### Annually

- [ ] Comprehensive compliance audit
- [ ] Impact reassessment
- [ ] Fairness metrics review
- [ ] Risk assessment update
- [ ] Stakeholder reporting
- [ ] Regulatory requirements review
- [ ] Documentation archival

---

## Best Practices

### Compliance Program
- [ ] Designate compliance officer
- [ ] Establish governance committee
- [ ] Document all requirements
- [ ] Implement monitoring systems
- [ ] Regular audits
- [ ] Staff training

### Documentation
- [ ] Maintain complete records
- [ ] Version control
- [ ] Audit trail
- [ ] Accessible to regulators
- [ ] Properly archived
- [ ] Secure access

### Continuous Improvement
- [ ] Monitor regulatory changes
- [ ] Update procedures as needed
- [ ] Learn from compliance issues
- [ ] Proactive remediation
- [ ] Stakeholder engagement
- [ ] Transparency reporting

---

## References

- **Safety Frameworks:** See Chapter 9/01-AI-Safety-Frameworks.md
- **Ethical Principles:** See Chapter 9/02-Responsible-AI-Ethical-Principles.md
- **Guardrails:** See Chapter 7/03-NeMo-Guardrails-Safety-Framework.md

---

## Conclusion

Regulatory compliance is not a burden—it's a framework that ensures AI systems are developed and deployed responsibly. By understanding applicable regulations, implementing required safeguards, and maintaining clear documentation, organizations ensure their AI systems are compliant, trustworthy, and ready for deployment.

**Core Principle:** Compliance is not optional—it's required and demonstrates commitment to responsible AI.
