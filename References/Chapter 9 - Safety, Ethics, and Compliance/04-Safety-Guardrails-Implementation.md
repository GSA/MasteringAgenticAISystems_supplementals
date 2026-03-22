# Safety Guardrails Implementation for Agent Systems

**Source:** NVIDIA NeMo Guardrails, industry safety practices, responsible AI frameworks

**Focus:** Implementing practical safety controls and guardrails
**Scope:** Input/output guardrails, decision gates, safety policies, monitoring integration

---

## Guardrails Architecture

### Multi-Layer Guardrail Design

```
User Input
    ↓
┌─────────────────────┐
│ Input Guardrails    │ (Validate, sanitize, reject harmful input)
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Agent Processing    │ (Agent decision making)
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Decision Guardrails │ (Validate tool calls, check permissions)
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Output Guardrails   │ (Filter output, add disclaimers)
└─────────────────────┘
    ↓
User Response
```

---

## Input Guardrails

### Type 1: Content Filtering

```python
class InputGuardrails:
    def __init__(self):
        # Load content filters
        self.toxicity_detector = load_toxicity_model()
        self.pii_detector = load_pii_model()
        self.injection_patterns = load_injection_patterns()

    def check_input(self, text):
        """Validate user input"""
        issues = []

        # Check for toxic content
        if self.check_toxicity(text):
            issues.append("toxic_content")

        # Check for personal information
        if self.detect_pii(text):
            issues.append("pii_detected")

        # Check for injection attempts
        if self.detect_injection(text):
            issues.append("injection_attempt")

        # Check for restricted topics
        if self.is_restricted_topic(text):
            issues.append("restricted_topic")

        return issues

    def check_toxicity(self, text):
        """Check for hateful, offensive content"""
        from perspective import Client
        client = Client()
        response = client.analyze_comment(text)

        # Flag high toxicity
        return response['TOXICITY'] > 0.8

    def detect_pii(self, text):
        """Detect personally identifiable information"""
        pii_patterns = {
            "ssn": r"\d{3}-\d{2}-\d{4}",
            "credit_card": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(\+?1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        }

        for pii_type, pattern in pii_patterns.items():
            if re.search(pattern, text):
                return True

        return False

    def detect_injection(self, text):
        """Detect prompt injection attempts"""
        injection_patterns = [
            r"(?i)(ignore|forget|override).*(instruction|prompt|rule)",
            r"(?i)(you are now|pretend|act as).*(unrestricted|admin|root)",
            r"(?i)(system\s*:\s*|admin\s*mode|developer\s*mode)",
            r"(?i)(role\s*?play|jailbreak|break.*free)",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text):
                return True

        return False

    def is_restricted_topic(self, text):
        """Check if topic is restricted"""
        restricted_keywords = [
            "how to make explosives",
            "how to synthesize drugs",
            "how to commit fraud",
            "sexual content involving minors",
        ]

        text_lower = text.lower()
        for keyword in restricted_keywords:
            if keyword in text_lower:
                return True

        return False
```

---

## Decision Guardrails

### Type 1: Permission Checking

```python
class PermissionGuardrails:
    def __init__(self, user_permissions):
        self.permissions = user_permissions

    def can_execute_tool(self, user_id, tool_name):
        """Check if user can use this tool"""
        # Get user permissions
        user_perms = self.permissions.get(user_id, {})

        # Check tool access
        if tool_name not in user_perms.get("allowed_tools", []):
            return False

        # Check if tool is restricted
        if tool_name in ["delete_database", "modify_user", "access_pii"]:
            return user_perms.get("admin", False)

        return True

    def check_parameter_safety(self, tool_name, parameters):
        """Validate tool parameters"""
        # Database query safety
        if tool_name == "execute_query":
            query = parameters.get("query", "")
            if "DROP" in query.upper() or "DELETE" in query.upper():
                return False  # Prevent destructive queries

        # File operations safety
        if tool_name == "read_file":
            filepath = parameters.get("path", "")
            if not self.is_safe_path(filepath):
                return False  # Prevent directory traversal

        # API call safety
        if tool_name == "make_request":
            url = parameters.get("url", "")
            if not self.is_safe_url(url):
                return False  # Prevent SSRF

        return True

    def is_safe_path(self, filepath):
        """Check if path access is safe"""
        # Resolve to absolute path
        resolved = os.path.abspath(filepath)

        # Check against whitelist
        allowed_dirs = ["/data/documents", "/data/uploads"]
        return any(resolved.startswith(d) for d in allowed_dirs)

    def is_safe_url(self, url):
        """Check if URL is safe to request"""
        blocked_domains = ["127.0.0.1", "localhost", "169.254"]
        parsed = urllib.parse.urlparse(url)

        # Block localhost/internal
        if any(blocked in parsed.netloc for blocked in blocked_domains):
            return False

        return True
```

### Type 2: Decision Validation

```python
class DecisionValidation:
    def validate_agent_decision(self, decision, context):
        """Validate that agent decision is appropriate"""
        validations = {}

        # Check confidence level
        if decision.get("confidence", 0) < 0.5:
            validations["low_confidence"] = True

        # Check for hallucination risk
        if self.detect_hallucination_risk(decision, context):
            validations["hallucination_risk"] = True

        # Check for policy violations
        if self.violates_policy(decision):
            validations["policy_violation"] = True

        # Determine if human review is needed
        if any(validations.values()):
            return {
                "valid": False,
                "needs_review": True,
                "issues": validations,
            }

        return {"valid": True, "needs_review": False}

    def detect_hallucination_risk(self, decision, context):
        """Check if decision is based on retrieval"""
        # If decision lacks supporting sources, flag as hallucination risk
        sources = decision.get("sources", [])
        if not sources and decision["type"] in ["factual_claim", "recommendation"]:
            return True

        return False

    def violates_policy(self, decision):
        """Check against organization policies"""
        policies = {
            "medical": r"diagnose|prescribe|treat",
            "legal": r"legal advice|lawsuit|attorney",
            "financial": r"investment advice|buy|sell",
        }

        decision_text = str(decision)
        for policy_type, pattern in policies.items():
            if re.search(pattern, decision_text, re.IGNORECASE):
                if policy_type in self.restricted_domains:
                    return True

        return False
```

---

## Output Guardrails

### Type 1: Output Filtering and Masking

```python
class OutputGuardrails:
    def sanitize_output(self, response):
        """Clean and filter agent response"""
        # Remove any detected sensitive info
        response = self.remove_pii(response)

        # Add safety disclaimers where needed
        response = self.add_disclaimers(response)

        # Filter potentially harmful content
        response = self.filter_harmful_content(response)

        # Add confidence levels if available
        response = self.add_confidence_indicators(response)

        return response

    def remove_pii(self, text):
        """Remove or mask PII from output"""
        pii_patterns = {
            "ssn": (r"\d{3}-\d{2}-\d{4}", "XXX-XX-####"),
            "credit_card": (r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "XXXX-XXXX-XXXX-####"),
            "phone": (r"(\+?1)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "###-###-####"),
        }

        for pii_type, (pattern, replacement) in pii_patterns.items():
            text = re.sub(pattern, replacement, text)

        return text

    def add_disclaimers(self, response):
        """Add appropriate disclaimers"""
        # Medical content disclaimer
        if self.contains_medical_content(response):
            disclaimer = "⚠️ This is for informational purposes only. Consult a healthcare provider for medical advice."
            return f"{response}\n\n{disclaimer}"

        # Financial content disclaimer
        if self.contains_financial_content(response):
            disclaimer = "⚠️ This is not financial advice. Consult a financial advisor before making decisions."
            return f"{response}\n\n{disclaimer}"

        # Legal content disclaimer
        if self.contains_legal_content(response):
            disclaimer = "⚠️ This is not legal advice. Consult an attorney for legal matters."
            return f"{response}\n\n{disclaimer}"

        return response

    def filter_harmful_content(self, text):
        """Remove or flag harmful content"""
        if self.detector.is_harmful(text):
            return "[Content filtered due to safety concerns]"

        return text

    def add_confidence_indicators(self, response):
        """Add confidence level to output"""
        if hasattr(response, 'confidence'):
            confidence = response.confidence
            indicator = self.get_confidence_indicator(confidence)
            return f"{response}\n\n{indicator}"

        return response

    def get_confidence_indicator(self, confidence):
        """Generate confidence indicator"""
        if confidence > 0.9:
            return "✓ High confidence"
        elif confidence > 0.7:
            return "△ Medium confidence"
        else:
            return "! Low confidence - verify before using"
```

### Type 2: Content Appropriateness Check

```python
class AppropriatenessCheck:
    def check_response_appropriateness(self, response, user_context):
        """Verify response is appropriate for user"""
        checks = {
            "age_appropriate": self.is_age_appropriate(response, user_context),
            "content_appropriate": self.is_content_appropriate(response),
            "factually_accurate": self.is_factually_accurate(response),
            "harmful_content": not self.contains_harmful_content(response),
            "bias_check": not self.exhibits_bias(response),
        }

        if not all(checks.values()):
            return False, checks

        return True, checks

    def is_age_appropriate(self, response, user_context):
        """Check if content is appropriate for user age"""
        if not user_context.get("age"):
            return True

        user_age = user_context["age"]
        harmful_keywords = {
            "violence": ["murder", "kill", "harm"],
            "sex": ["sexual", "nude", "pornographic"],
            "drugs": ["heroin", "cocaine"],
        }

        response_text = response.lower()

        if user_age < 13:
            # Very restrictive for children
            return not any(
                keyword in response_text
                for keywords in harmful_keywords.values()
                for keyword in keywords
            )

        if user_age < 18:
            # Somewhat restrictive for teens
            if "sex" in harmful_keywords:
                return not any(
                    k in response_text for k in harmful_keywords["sex"]
                )

        return True

    def is_factually_accurate(self, response):
        """Check if response is factually accurate"""
        # Use fact-checking API
        from fact_check import FactChecker

        checker = FactChecker()
        claims = self.extract_factual_claims(response)

        for claim in claims:
            accuracy = checker.check(claim)
            if accuracy.score < 0.7:
                return False

        return True

    def exhibits_bias(self, response):
        """Check for potential bias in response"""
        from bias_detector import BiasDetector

        detector = BiasDetector()
        bias_score = detector.detect(response)

        # High bias score indicates problematic content
        return bias_score > 0.7
```

---

## Safety Policy Definition

### Example Safety Policy

```yaml
Safety Policy: Customer Service Agent

Rules:
  ────────────────────────────────────────

  Rule 1: Medical Information
  ───────────────────────────
  Condition: User asks medical question
  Action:
    ├─ Decline to provide diagnosis
    ├─ Redirect to healthcare provider
    ├─ Add "consult doctor" disclaimer
    └─ Log attempt

  Rule 2: Financial Advice
  ────────────────────────
  Condition: User asks for financial advice
  Action:
    ├─ Provide general information only
    ├─ Add "not financial advice" disclaimer
    ├─ Suggest consulting financial advisor
    └─ Log interaction

  Rule 3: Personal Identifying Information
  ──────────────────────────────────────
  Condition: User provides SSN, credit card, etc.
  Action:
    ├─ Don't repeat it back to user
    ├─ Ask to confirm via secure channel
    ├─ Never store in logs
    └─ Log attempt (without PII)

  Rule 4: Harmful Requests
  ─────────────────────
  Condition: User asks how to cause harm
  Action:
    ├─ Decline politely
    ├─ Don't explain why in detail
    ├─ Suggest alternative
    └─ Log for security review

  Rule 5: Tool Safety
  ──────────────────
  Condition: System needs to call tool
  Action:
    ├─ Verify user authorization
    ├─ Validate parameters
    ├─ Log all tool calls
    ├─ Check for side effects
    └─ Require confirmation if risky

Monitoring:
  ──────────
  ├─ Track policy violations
  ├─ Alert on patterns
  ├─ Regular review
  └─ Update as needed
```

---

## Guardrail Configuration

### Configuration File Example

```python
# guardrails_config.py

GUARDRAIL_RULES = {
    "input_validation": {
        "max_length": 5000,
        "check_toxicity": True,
        "toxicity_threshold": 0.8,
        "detect_pii": True,
        "detect_injection": True,
    },

    "decision_validation": {
        "min_confidence": 0.5,
        "check_hallucination": True,
        "validate_permissions": True,
        "check_parameters": True,
    },

    "output_filters": {
        "remove_pii": True,
        "filter_harmful": True,
        "add_disclaimers": True,
        "add_confidence": True,
    },

    "restricted_domains": {
        "medical": {
            "keywords": ["diagnose", "prescribe", "treatment"],
            "action": "decline",
        },
        "legal": {
            "keywords": ["legal advice", "lawsuit", "attorney"],
            "action": "decline",
        },
        "financial": {
            "keywords": ["investment advice", "buy stock"],
            "action": "general_info_only",
        },
    },

    "permissions": {
        "tier1_users": ["read_public_data", "search"],
        "tier2_users": ["read_private_data", "modify_own"],
        "admins": ["all"],
    },

    "logging": {
        "log_violations": True,
        "log_decisions": True,
        "log_tool_calls": True,
        "alert_on_critical": True,
    },
}

# Severity levels for different violations
VIOLATION_SEVERITY = {
    "injection_attempt": "critical",
    "pii_leak": "critical",
    "unauthorized_tool": "high",
    "low_confidence": "medium",
    "deprecated_api": "low",
}
```

---

## Monitoring Guardrails

### Guardrail Metrics

```python
class GuardrailMonitoring:
    def __init__(self):
        self.violations = []
        self.by_type = defaultdict(int)

    def log_violation(self, violation_type, severity, details):
        """Record a guardrail violation"""
        self.violations.append({
            "timestamp": datetime.now(),
            "type": violation_type,
            "severity": severity,
            "details": details,
        })
        self.by_type[violation_type] += 1

    def get_guardrail_report(self):
        """Generate guardrail effectiveness report"""
        total_violations = len(self.violations)
        critical = sum(
            1 for v in self.violations
            if v["severity"] == "critical"
        )

        return {
            "total_violations": total_violations,
            "critical_violations": critical,
            "by_type": dict(self.by_type),
            "alert_needed": critical > 5,
        }

    def is_system_safe(self):
        """Determine if system meets safety standards"""
        report = self.get_guardrail_report()

        # System is unsafe if too many critical violations
        return report["critical_violations"] < 10
```

---

## Best Practices

### Guardrail Implementation
- [ ] Start with reasonable restrictions
- [ ] Monitor effectiveness
- [ ] Adjust based on real usage
- [ ] Don't make it unusable (false positive balance)
- [ ] Document all rules
- [ ] Test guardrails thoroughly

### Safety Culture
- [ ] Team training on safety
- [ ] Regular safety reviews
- [ ] Incident documentation
- [ ] Continuous improvement
- [ ] Share learnings
- [ ] Celebrate safety improvements

---

## References

- **NeMo Guardrails:** See Chapter 7/03-NeMo-Guardrails-Safety-Framework.md
- **Safety Frameworks:** See Chapter 9/01-AI-Safety-Frameworks.md
- **Monitoring:** See Chapter 8/02-ML-Monitoring-Production.md

---

## Conclusion

Guardrails are the practical implementation of safety principles. By combining input validation, decision checking, output filtering, and continuous monitoring, organizations create multiple layers of protection that keep agent systems safe and trustworthy.

**Core Principle:** Guardrails enable safety without completely preventing useful functionality.
