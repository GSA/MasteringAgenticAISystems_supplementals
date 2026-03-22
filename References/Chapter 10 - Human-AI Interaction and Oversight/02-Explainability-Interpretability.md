# Explainability and Interpretability in Agent Systems

**Source:** Interpretable ML research, XAI frameworks, user research

**Focus:** Making agent decisions understandable to users and stakeholders
**Scope:** Explanation types, technical methods, user communication, testing

---

## Explanation Levels

### Level 1: What (The Decision)
**For Users:** "The system approved your loan application"
- Clear statement of what was decided
- Easy to understand outcome
- No technical details

### Level 2: Why (The Reasoning)
**For Users:** "Based on your excellent credit history and stable employment, the system approved your loan"
- Key factors considered
- Relative importance of factors
- Connection to decision

### Level 3: How (The Process)
**For Experts:** "System evaluated 15 features using gradient boosting model, with these top features contributing 60% to the decision..."
- Technical details of process
- Model architecture
- Feature importance
- Confidence metrics

### Level 4: Why Not (The Alternatives)
**For Verification:** "Alternative decision would have been rejection due to debt-to-income ratio >40%, but credit history outweighed this factor"
- Why other decisions weren't chosen
- Trade-offs in decision
- Factors pushing other ways
- Robustness of decision

---

## Explanation Techniques

### Technique 1: Feature Importance

```python
def explain_by_importance(decision):
    """Explain using important features"""
    # Calculate feature importance for this prediction
    important_features = model.get_feature_importance(decision.inputs)

    explanation = f"""
    Your application was approved based on these factors:

    Most Important Factors (positive):
    ├─ Credit Score: 750 (Excellent) - Very positive
    ├─ Employment Stability: 8 years - Strong
    └─ Debt-to-Income: 25% - Good

    Least Important Factors:
    └─ College Degree: Yes - Neutral
    """

    return explanation
```

### Technique 2: Example-Based Explanation

```python
def explain_by_examples(decision, k=3):
    """Show similar past decisions"""
    # Find similar historical cases
    similar_cases = find_similar_historical_cases(decision, k=k)

    explanation = f"""
    Your application is similar to these approved applications:

    Case 1: Similar applicant with score 740, approved
    Case 2: Similar applicant with score 755, approved
    Case 3: Similar applicant with debt 27%, approved

    Your case is more favorable than all of these.
    """

    return explanation
```

### Technique 3: Counterfactual Explanation

```python
def explain_by_counterfactual(decision):
    """Show what would need to change"""
    # Find minimal changes to flip decision
    counterfactual = find_minimal_change_to_flip(decision)

    explanation = f"""
    Your application was approved.

    If your situation were different:
    - If credit score were 620 (instead of 750): Would be rejected
    - If debt-to-income were 45% (instead of 25%): Would be rejected
    - If employment < 2 years (instead of 8): Uncertain

    Your strongest factor is your credit history.
    """

    return explanation
```

### Technique 4: Local Explanation (LIME/SHAP)

```python
def explain_with_lime(decision):
    """Use LIME for local approximation"""
    from lime.tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        feature_names=['credit_score', 'employment_years', ...],
        feature_types=['numeric', 'numeric', ...],
        class_names=['approved', 'rejected']
    )

    explanation = explainer.explain_instance(
        decision.input_features,
        model.predict_proba,
        num_features=5
    )

    return explanation.as_list()
```

---

## User-Centered Explanations

### Designing Explanations for Different Users

**For Loan Applicants:**
```
"Your application was approved because:
- Your credit score (750) shows you pay bills on time
- You've worked at your company for 8 years, showing stable income
- Your debt compared to income (25%) is very reasonable

If you want to improve your credit further, you could:
- Keep paying bills on time
- Reduce your overall debt"
```

**For Financial Managers:**
```
Model Decision: APPROVED
Confidence: 87%
Risk Score: 0.12 (LOW)
Key Drivers:
- Credit: +0.35
- Income: +0.25
- Debt Ratio: +0.15
- Employment: +0.12
```

**For Regulators:**
```
Decision Analysis:
- Feature importance verified
- No protected attribute correlation detected
- Disparate impact ratio: 0.92 (compliant)
- Historical accuracy: 94%
- Monitoring: Active
```

---

## Transparency vs. Complexity Trade-off

### Finding the Right Balance

```
Too Simple:
├─ Problem: Lacks credibility
├─ Users: Don't trust decisions
└─ Solution: Add more detail

Too Complex:
├─ Problem: Users confused
├─ Users: Can't understand explanation
└─ Solution: Simplify and tier

Just Right:
├─ Clear main points
├─ Technical details available
├─ Multiple explanation types
└─ Matches user needs
```

### Progressive Disclosure

```
User Interface Design:

First Screen (Simple):
┌─────────────────────────────┐
│ Your application: APPROVED  │
│ Reason: Strong application  │
│ [View More Details]         │
└─────────────────────────────┘

Second Screen (Moderate):
┌─────────────────────────────┐
│ APPROVED                    │
│ Top factors:                │
│ • Credit Score: Good        │
│ • Income: Stable            │
│ • Debt: Low                 │
│ [View Technical Details]    │
└─────────────────────────────┘

Third Screen (Detailed):
┌─────────────────────────────┐
│ Feature Importance:         │
│ credit_score: 0.35          │
│ employment_years: 0.25      │
│ debt_ratio: 0.15            │
│ ...                         │
└─────────────────────────────┘
```

---

## Testing Explanations

### Test 1: Explanation Fidelity

**Does explanation match actual decision?**

```python
def test_explanation_fidelity(model, decision, explanation):
    """Verify explanation matches decision"""
    # Remove features mentioned in explanation
    modified_input = remove_features(decision.input, explanation.features)

    # Predict with modified input
    modified_prediction = model.predict(modified_input)

    # Compare to original
    # If explanation correct: prediction should change significantly
    change = abs(original_prediction - modified_prediction)

    assert change > 0.3, "Explanation doesn't match decision"
```

### Test 2: User Understanding

**Do users understand the explanation?**

```
Usability Testing:
1. Show explanation to representative users
2. Ask them to:
   - Explain decision in their own words
   - Predict outcome for similar case
   - Identify key factors
   - Answer: "Do you trust this decision?"
3. Score comprehension 0-100
4. Target: >85% comprehension
```

### Test 3: Explanation Stability

**Are explanations consistent across similar cases?**

```python
def test_explanation_stability():
    """Similar inputs should get similar explanations"""

    # Slightly perturb input (noise)
    for noise in [0.01, 0.05, 0.1]:
        perturbed = add_noise(decision.input, noise)
        expl1 = explain(decision)
        expl2 = explain(perturbed)

        # Check similarity
        similarity = compare_explanations(expl1, expl2)
        assert similarity > 0.9, "Explanation unstable"
```

---

## Common Explanation Pitfalls

### Pitfall 1: Post-hoc Rationalization

**Problem:** Explanation doesn't actually explain the decision
**Example:** "System approved because credit score is good" (but it's really the employment history)
**Solution:** Verify explanation with feature importance testing

### Pitfall 2: Overconfidence

**Problem:** Explanation makes user overconfident in system
**Example:** "System made this decision, so it must be right"
**Solution:** Include uncertainty, explain limitations, allow appeals

### Pitfall 3: Cognitive Biases

**Problem:** Users misinterpret explanation due to cognitive biases
**Example:** Users focus on first mentioned factor, ignore others
**Solution:** Test explanations with users, iterate design

### Pitfall 4: Information Overload

**Problem:** Too many details confuse rather than clarify
**Example:** Showing all 100 features' importances
**Solution:** Use progressive disclosure, show top 5-7 factors

---

## Best Practices

### Explanation Design
- [ ] Tailor to audience
- [ ] Start simple, allow deeper dive
- [ ] Use plain language
- [ ] Show uncertainty/confidence
- [ ] Include context
- [ ] Test with users

### Implementation
- [ ] Multiple explanation types available
- [ ] Explanations tied to real system logic
- [ ] Consistent explanations
- [ ] Explanations update with feedback
- [ ] Performance tracking
- [ ] Continuous improvement

---

## References

- **Safety Frameworks:** See Chapter 9/01-AI-Safety-Frameworks.md
- **Evaluation:** See Chapter 3 for quality assessment
- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md

---

## Conclusion

Explainability makes agent systems trustworthy and accountable. By providing clear, accurate, and understandable explanations tailored to user needs, organizations enable informed decision-making and maintain human trust in AI systems.

**Core Principle:** If you can't explain it, you shouldn't deploy it.
