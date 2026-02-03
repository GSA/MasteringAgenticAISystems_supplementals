from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplianceCheckLevel(Enum):
    """Compliance analysis depth levels."""
    FULL = "full"  # All checks (LLM reasoning + rules)
    ADVANCED = "advanced"  # LLM reasoning only
    BASIC = "basic"  # Rule-based checks only
    MINIMAL = "minimal"  # Critical checks only


class ComplianceAgent:
    """
    Compliance analysis agent with graceful degradation.

    Maintains partial functionality during LLM or service failures.
    """

    def __init__(self, llm_client, rule_engine, critical_rules: List[str]):
        self.llm_client = llm_client
        self.rule_engine = rule_engine
        self.critical_rules = critical_rules
        self.llm_available = True
        self.rule_engine_available = True

    def analyze_proposal(
        self,
        proposal_text: str,
        target_level: ComplianceCheckLevel = ComplianceCheckLevel.FULL
    ) -> Dict[str, Any]:
        """
        Analyze proposal for compliance issues with graceful degradation.

        Automatically degrades to lower capability levels on component failures.
        """
        # Determine maximum achievable level based on component availability
        achievable_level = self._determine_achievable_level(target_level)

        if achievable_level != target_level:
            logger.warning(
                f"Degrading compliance analysis from {target_level.value} "
                f"to {achievable_level.value} due to component failures"
            )

        # Execute analysis at achievable level
        if achievable_level == ComplianceCheckLevel.FULL:
            return self._full_analysis(proposal_text)
        elif achievable_level == ComplianceCheckLevel.ADVANCED:
            return self._advanced_analysis(proposal_text)
        elif achievable_level == ComplianceCheckLevel.BASIC:
            return self._basic_analysis(proposal_text)
        elif achievable_level == ComplianceCheckLevel.MINIMAL:
            return self._minimal_analysis(proposal_text)
        else:
            # Complete system failure
            return self._failure_response()

    def _determine_achievable_level(
        self,
        target_level: ComplianceCheckLevel
    ) -> ComplianceCheckLevel:
        """Determine maximum achievable analysis level given component health."""

        # Full analysis requires both LLM and rule engine
        if self.llm_available and self.rule_engine_available:
            return ComplianceCheckLevel.FULL

        # Advanced analysis requires LLM only
        if self.llm_available and not self.rule_engine_available:
            return ComplianceCheckLevel.ADVANCED

        # Basic analysis requires rule engine only
        if not self.llm_available and self.rule_engine_available:
            return ComplianceCheckLevel.BASIC

        # Minimal analysis always available (static critical checks)
        return ComplianceCheckLevel.MINIMAL

    def _full_analysis(self, proposal_text: str) -> Dict[str, Any]:
        """Full compliance analysis using all capabilities."""
        logger.info("Executing full compliance analysis")

        try:
            # LLM-based semantic compliance reasoning
            llm_issues = self._llm_semantic_analysis(proposal_text)

            # Rule-based compliance checks
            rule_issues = self.rule_engine.check_all_rules(proposal_text)

            return {
                "level": ComplianceCheckLevel.FULL.value,
                "llm_issues": llm_issues,
                "rule_issues": rule_issues,
                "total_issues": len(llm_issues) + len(rule_issues),
                "status": "complete",
                "confidence": "high",
                "degraded": False
            }
        except Exception as e:
            logger.error(f"Full analysis failed: {str(e)}")
            self.llm_available = False
            self.rule_engine_available = False
            # Retry at degraded level
            return self.analyze_proposal(proposal_text, ComplianceCheckLevel.FULL)

    def _advanced_analysis(self, proposal_text: str) -> Dict[str, Any]:
        """LLM-only analysis (rule engine unavailable)."""
        logger.warning("Executing advanced compliance analysis (rule engine degraded)")

        try:
            llm_issues = self._llm_semantic_analysis(proposal_text)

            return {
                "level": ComplianceCheckLevel.ADVANCED.value,
                "llm_issues": llm_issues,
                "rule_issues": [],
                "total_issues": len(llm_issues),
                "status": "complete",
                "confidence": "medium",
                "degraded": True,
                "warning": "Rule-based checks unavailable. Manual verification recommended."
            }
        except Exception as e:
            logger.error(f"Advanced analysis failed: {str(e)}")
            self.llm_available = False
            return self.analyze_proposal(proposal_text, ComplianceCheckLevel.FULL)

    def _basic_analysis(self, proposal_text: str) -> Dict[str, Any]:
        """Rule-based analysis only (LLM unavailable)."""
        logger.warning("Executing basic compliance analysis (LLM degraded)")

        try:
            rule_issues = self.rule_engine.check_all_rules(proposal_text)

            return {
                "level": ComplianceCheckLevel.BASIC.value,
                "llm_issues": [],
                "rule_issues": rule_issues,
                "total_issues": len(rule_issues),
                "status": "complete",
                "confidence": "medium",
                "degraded": True,
                "warning": "Semantic analysis unavailable. Complex compliance issues may be missed."
            }
        except Exception as e:
            logger.error(f"Basic analysis failed: {str(e)}")
            self.rule_engine_available = False
            return self.analyze_proposal(proposal_text, ComplianceCheckLevel.FULL)

    def _minimal_analysis(self, proposal_text: str) -> Dict[str, Any]:
        """Critical checks only (both LLM and rule engine degraded)."""
        logger.error("Executing minimal compliance analysis (severe degradation)")

        # Static critical checks that don't require external services
        critical_issues = []

        for rule in self.critical_rules:
            if rule == "required_sections":
                # Check for required section headers
                required = ["Executive Summary", "Technical Approach", "Budget"]
                for section in required:
                    if section not in proposal_text:
                        critical_issues.append({
                            "type": "missing_section",
                            "section": section,
                            "severity": "critical"
                        })

            elif rule == "forbidden_terms":
                # Check for explicitly forbidden terms
                forbidden = ["proprietary", "confidential_source", "unrealistic_timeline"]
                for term in forbidden:
                    if term in proposal_text.lower():
                        critical_issues.append({
                            "type": "forbidden_term",
                            "term": term,
                            "severity": "critical"
                        })

        return {
            "level": ComplianceCheckLevel.MINIMAL.value,
            "llm_issues": [],
            "rule_issues": [],
            "critical_issues": critical_issues,
            "total_issues": len(critical_issues),
            "status": "degraded",
            "confidence": "low",
            "degraded": True,
            "warning": "SEVERE DEGRADATION: Only critical checks available. Manual review required."
        }

    def _failure_response(self) -> Dict[str, Any]:
        """Complete system failure response."""
        logger.critical("All compliance analysis capabilities failed")

        return {
            "level": "failed",
            "llm_issues": [],
            "rule_issues": [],
            "critical_issues": [],
            "total_issues": 0,
            "status": "failed",
            "confidence": "none",
            "degraded": True,
            "error": "All compliance systems unavailable. Manual review required immediately."
        }

    def _llm_semantic_analysis(self, proposal_text: str) -> List[Dict[str, Any]]:
        """Perform LLM-based semantic compliance analysis."""
        prompt = f"""Analyze this proposal for compliance issues:
        - Check for unrealistic promises
        - Identify potential conflicts of interest
        - Flag vague or ambiguous commitments
        - Verify technical feasibility claims

        Proposal:
        {proposal_text[:4000]}  # Truncate to fit context window

        Return issues as JSON array with: type, description, severity, location
        """

        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1
        )

        # Parse LLM response (simplified for example)
        import json
        try:
            issues = json.loads(response.choices[0].message.content)
            return issues
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return []


# Usage example
def process_proposal_with_degradation(
    proposal_text: str,
    agent: ComplianceAgent
) -> None:
    """Process proposal with graceful degradation handling."""

    result = agent.analyze_proposal(
        proposal_text,
        target_level=ComplianceCheckLevel.FULL
    )

    # Display results with degradation awareness
    print(f"Analysis Level: {result['level']}")
    print(f"Total Issues Found: {result['total_issues']}")
    print(f"Confidence: {result['confidence']}")

    if result['degraded']:
        print(f"\n⚠️  WARNING: {result.get('warning', 'System degraded')}")
        print("Consider manual review for comprehensive compliance verification.")

    if result['status'] == 'failed':
        print(f"\n❌ ERROR: {result['error']}")
        print("Manual compliance review required immediately.")
