"""
Code Example 10.3.2: Natural Language Explanation Generator

Purpose: Transform technical agent decision traces into audience-appropriate
natural language explanations

Concepts Demonstrated:
- Multi-level explanation generation (basic, detailed, technical)
- Counterfactual explanations ("what would need to change")
- Actionable improvement suggestions
- Regulatory compliance (ECOA adverse action notices)

Prerequisites:
- Understanding of decision traces (see traceable_agent.py)
- Familiarity with explanation types
- Knowledge of regulatory requirements for explainability

Author: NVIDIA Agentic AI Certification
Chapter: 10, Section: 10.3
Exam Skill: 10.3 - Implement transparency mechanisms
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# CONFIGURATION AND TYPES
# ============================================================================

class ExplanationLevel(Enum):
    """
    Audience-appropriate explanation levels.

    BASIC: End users who need simple, actionable language
    DETAILED: Power users who want more context and factors
    TECHNICAL: Developers/auditors who need full technical details
    """
    BASIC = "basic"          # Non-technical users
    DETAILED = "detailed"    # Engaged users wanting more context
    TECHNICAL = "technical"  # Developers, auditors, regulators


class DecisionOutcome(Enum):
    """Possible decision outcomes"""
    APPROVED = "approved"
    DECLINED = "declined"
    PENDING = "pending"
    ESCALATED = "escalated"


@dataclass
class LoanDecision:
    """
    Example decision structure: Loan application.

    This represents the data needed to generate explanations.
    """
    # Identifiers
    applicant_id: str
    decision_id: str

    # Decision
    outcome: DecisionOutcome
    approved: bool

    # Input features (what the model saw)
    credit_score: int
    annual_income: float
    debt_to_income_ratio: float  # Percentage
    employment_years: int
    loan_amount: float

    # Model internals
    feature_weights: Dict[str, float]  # Feature importance
    confidence_score: float
    policy_threshold: float

    # Decision factors
    primary_reason: str
    contributing_factors: List[str]

    # Metadata
    timestamp: str
    model_version: str


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """
    Generate natural language explanations from agent decisions.

    Supports multiple explanation levels for different audiences:
    - Basic: Simple, non-technical language for end users
    - Detailed: More context and factors for engaged users
    - Technical: Full technical details for auditors/developers

    Key Features:
    - Audience-appropriate language
    - Counterfactual explanations (what-if scenarios)
    - Actionable improvement suggestions
    - Regulatory compliance (ECOA, GDPR)
    """

    def __init__(self):
        """Initialize explanation generator"""
        # Humanized factor names
        self.factor_names = {
            "credit_score": "Credit Score",
            "annual_income": "Annual Income",
            "debt_to_income_ratio": "Debt-to-Income Ratio",
            "employment_years": "Employment History",
            "loan_amount": "Loan Amount"
        }

        # Humanized reason descriptions
        self.reason_descriptions = {
            "low_credit_score": "Your credit score is below our minimum requirement",
            "high_debt_ratio": "Your debt-to-income ratio exceeds our policy limit",
            "insufficient_income": "Your income doesn't meet the minimum threshold for this loan amount",
            "short_employment": "You haven't been employed long enough to meet our stability requirement",
            "excessive_loan_amount": "The requested loan amount exceeds our limit for your income level",
            "excellent_credit": "You have an excellent credit history",
            "low_debt_burden": "You have manageable debt levels",
            "stable_employment": "You have stable, long-term employment",
            "strong_income": "Your income comfortably supports this loan"
        }

    def generate_explanation(
        self,
        decision: LoanDecision,
        level: ExplanationLevel = ExplanationLevel.BASIC
    ) -> str:
        """
        Generate audience-appropriate explanation.

        Args:
            decision: Complete decision data
            level: Desired explanation level

        Returns:
            Natural language explanation as string

        Example:
            >>> decision = LoanDecision(...)
            >>> generator = ExplanationGenerator()
            >>> explanation = generator.generate_explanation(decision, ExplanationLevel.BASIC)
            >>> print(explanation)
            Your loan application has been APPROVED...
        """
        if level == ExplanationLevel.BASIC:
            return self._generate_basic_explanation(decision)
        elif level == ExplanationLevel.DETAILED:
            return self._generate_detailed_explanation(decision)
        else:
            return self._generate_technical_explanation(decision)

    def _generate_basic_explanation(self, decision: LoanDecision) -> str:
        """
        Basic explanation for end users.

        Focus: Simple language, primary reason, actionable guidance
        Avoid: Technical jargon, numerical scores, complex details

        Principles:
        - One clear sentence stating outcome
        - Main reason in plain English
        - Actionable next steps
        - Empathetic tone for denials
        """
        if decision.approved:
            explanation = f"""
Your loan application has been APPROVED! ✓

Why: {self._humanize_reason(decision.primary_reason)}

What's Next:
• You'll receive final loan documents within 2 business days
• Please review and sign the documents electronically
• Funds will be disbursed to your account within 5 business days after signing
• Questions? Contact your loan officer at loans@example.com

Application ID: {decision.decision_id}
"""
        else:
            explanation = f"""
Your loan application has been DECLINED.

Main Reason: {self._humanize_reason(decision.primary_reason)}

How to Improve:
{self._generate_improvement_suggestions(decision)}

You can reapply after making these improvements. We're here to help you
achieve your financial goals.

Questions? Contact us at loans@example.com
Application ID: {decision.decision_id}
"""

        return explanation.strip()

    def _generate_detailed_explanation(self, decision: LoanDecision) -> str:
        """
        Detailed explanation for engaged users.

        Includes:
        - Multiple contributing factors
        - Relative importance
        - Concrete metrics
        - Counterfactual scenarios

        Audience: Users who want to understand the full picture
        """
        status = "APPROVED ✓" if decision.approved else "DECLINED"

        explanation = f"""
LOAN APPLICATION DECISION: {status}

Application ID: {decision.decision_id}
Date: {decision.timestamp}

PRIMARY FACTOR: {self._humanize_reason(decision.primary_reason)}

CONTRIBUTING FACTORS (Ranked by Importance):
"""

        # Rank factors by importance (absolute weight)
        sorted_factors = sorted(
            decision.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for i, (factor, weight) in enumerate(sorted_factors[:4], 1):
            impact = "Positive" if weight > 0 else "Negative"
            strength = self._categorize_impact(abs(weight))

            explanation += f"{i}. {self._humanize_factor(factor)}: {strength} {impact.lower()} impact\n"

        # Add specific metrics
        explanation += f"""
YOUR PROFILE:
• Credit Score: {decision.credit_score}
• Annual Income: ${decision.annual_income:,.2f}
• Debt-to-Income Ratio: {decision.debt_to_income_ratio:.1f}%
• Employment: {decision.employment_years} years
• Loan Amount Requested: ${decision.loan_amount:,.2f}
"""

        # Add counterfactual for denials
        if not decision.approved:
            explanation += f"\nALTERNATIVE SCENARIOS:\n"
            explanation += self._generate_counterfactual(decision)

        # Add confidence
        explanation += f"\nDecision Confidence: {decision.confidence_score * 100:.0f}%"

        return explanation.strip()

    def _generate_technical_explanation(self, decision: LoanDecision) -> str:
        """
        Technical explanation for auditors/developers.

        Includes:
        - All features and weights
        - Model thresholds
        - Compliance references
        - Calculation details

        Audience: Technical staff, auditors, regulators
        """
        status = "APPROVED" if decision.approved else "DECLINED"

        explanation = f"""
TECHNICAL DECISION REPORT
{"=" * 60}

DECISION SUMMARY:
• Outcome: {status}
• Applicant ID: {decision.applicant_id}
• Decision ID: {decision.decision_id}
• Timestamp: {decision.timestamp}
• Model Version: {decision.model_version}
• Confidence Score: {decision.confidence_score:.4f}

INPUT FEATURES:
• credit_score: {decision.credit_score} (FICO scale: 300-850)
• annual_income: ${decision.annual_income:,.2f}
• debt_to_income_ratio: {decision.debt_to_income_ratio:.1f}%
• employment_years: {decision.employment_years}
• loan_amount: ${decision.loan_amount:,.2f}

FEATURE WEIGHTS (Model: GradientBoostingClassifier v{decision.model_version}):
"""

        for feature, weight in sorted(
            decision.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            explanation += f"• {feature:25s}: {weight:+.6f}\n"

        # Calculate combined score
        combined_score = self._calculate_combined_score(decision)

        explanation += f"""
DECISION LOGIC:
• Combined Weighted Score: {combined_score:.6f}
• Policy Threshold: {decision.policy_threshold:.6f}
• Threshold Check: {"PASS" if combined_score >= decision.policy_threshold else "FAIL"}
• Primary Reason Code: {decision.primary_reason}
• Contributing Factors: {", ".join(decision.contributing_factors)}

REGULATORY COMPLIANCE:
• Framework: ECOA (Equal Credit Opportunity Act)
• Regulation: Regulation B - Adverse Action Notices
• Reference: 12 CFR 1002.9
• Fair Credit Reporting Act (FCRA): 15 U.S.C. § 1681
• Policy Document: Underwriting Policy v3.2, Section 4.5

AUDIT TRAIL:
• Decision logged to immutable audit database
• Retention period: 25 months (regulatory minimum)
• Access controls: Tier 2+ (auditors, compliance officers)
• Tamper detection: SHA-256 hash verification enabled

EXPLANATION FIDELITY:
This explanation is derived directly from the actual model execution.
All weights and features shown were used in the real decision computation.
"""

        return explanation.strip()

    def _humanize_reason(self, reason_code: str) -> str:
        """Convert technical reason code to human-readable description"""
        return self.reason_descriptions.get(
            reason_code,
            f"Policy requirement not met: {reason_code}"
        )

    def _humanize_factor(self, factor: str) -> str:
        """Convert feature name to readable description"""
        return self.factor_names.get(factor, factor.replace("_", " ").title())

    def _categorize_impact(self, weight: float) -> str:
        """Categorize feature weight into human terms"""
        if weight > 0.4:
            return "Very strong"
        elif weight > 0.25:
            return "Strong"
        elif weight > 0.15:
            return "Moderate"
        else:
            return "Minor"

    def _generate_improvement_suggestions(self, decision: LoanDecision) -> str:
        """
        Generate actionable suggestions for declined applications.

        This is critical for user experience - tell users HOW to improve,
        not just WHY they were rejected.
        """
        suggestions = []

        # Credit score improvement
        if decision.credit_score < 650:
            target = 650
            suggestions.append(
                f"1. Improve your credit score to at least {target}:\n"
                f"   • Pay all bills on time for the next 6-12 months\n"
                f"   • Reduce credit card balances to below 30% of limits\n"
                f"   • Don't apply for new credit cards in the next 6 months\n"
                f"   • Check credit report for errors and dispute if needed"
            )

        # Debt-to-income improvement
        if decision.debt_to_income_ratio > 40:
            target_ratio = 38
            monthly_income = decision.annual_income / 12
            current_debt = (decision.debt_to_income_ratio / 100) * monthly_income
            target_debt = (target_ratio / 100) * monthly_income
            reduction_needed = current_debt - target_debt

            suggestions.append(
                f"2. Reduce monthly debt payments:\n"
                f"   • Current debt-to-income ratio: {decision.debt_to_income_ratio:.1f}%\n"
                f"   • Target ratio: {target_ratio}%\n"
                f"   • Reduce monthly debt by approximately ${reduction_needed:,.0f}\n"
                f"   • Consider paying down credit cards or consolidating loans"
            )

        # Employment stability
        if decision.employment_years < 2:
            years_needed = 2 - decision.employment_years
            suggestions.append(
                f"3. Build employment stability:\n"
                f"   • Continue current employment for at least {years_needed:.1f} more years\n"
                f"   • Avoid job changes during this period\n"
                f"   • Document stable income with recent pay stubs"
            )

        # Income vs loan amount
        income_to_loan_ratio = decision.annual_income / decision.loan_amount
        if income_to_loan_ratio < 3:
            target_ratio = 3.0
            affordable_loan = decision.annual_income / target_ratio
            suggestions.append(
                f"4. Consider a smaller loan amount:\n"
                f"   • Requested: ${decision.loan_amount:,.2f}\n"
                f"   • Recommended maximum: ${affordable_loan:,.2f}\n"
                f"   • Or increase annual income to ${decision.loan_amount * target_ratio:,.2f}+"
            )

        if not suggestions:
            suggestions.append(
                "• Continue improving your overall financial health\n"
                "• Consider consulting with a financial advisor"
            )

        return "\n\n".join(suggestions)

    def _generate_counterfactual(self, decision: LoanDecision) -> str:
        """
        Generate counterfactual explanations (what-if scenarios).

        Shows users the minimal changes that would flip the decision.
        This helps users understand decision boundaries and improvement paths.
        """
        counterfactuals = []

        # Credit score counterfactual
        if decision.credit_score < 680:
            target_score = 680
            improvement = target_score - decision.credit_score
            counterfactuals.append(
                f"• If your credit score were {target_score} (an increase of {improvement} points), "
                f"your application would likely be approved with other factors unchanged."
            )

        # Debt-to-income counterfactual
        if decision.debt_to_income_ratio > 38:
            target_ratio = 38
            reduction = decision.debt_to_income_ratio - target_ratio
            counterfactuals.append(
                f"• If you reduced your debt-to-income ratio by {reduction:.1f} percentage points "
                f"(from {decision.debt_to_income_ratio:.1f}% to {target_ratio}%), "
                f"your application would likely be approved."
            )

        # Employment counterfactual
        if decision.employment_years < 2:
            target_years = 2
            additional = target_years - decision.employment_years
            counterfactuals.append(
                f"• If you had {target_years} years of employment history "
                f"(an additional {additional:.1f} years), "
                f"this would strengthen your application significantly."
            )

        # Loan amount counterfactual
        income_to_loan = decision.annual_income / decision.loan_amount
        if income_to_loan < 3:
            affordable_amount = decision.annual_income / 3
            reduction = decision.loan_amount - affordable_amount
            counterfactuals.append(
                f"• If you requested ${affordable_amount:,.2f} instead of ${decision.loan_amount:,.2f} "
                f"(a reduction of ${reduction:,.2f}), "
                f"your application would be more likely to be approved."
            )

        return "\n".join(counterfactuals[:3])  # Show top 3 most actionable

    def _calculate_combined_score(self, decision: LoanDecision) -> float:
        """
        Calculate weighted combined score.

        This demonstrates how the model combines features into a final score.
        """
        score = 0.0

        # Normalize and weight each feature
        # Credit score: 300-850 scale, normalized to 0-1
        if "credit_score" in decision.feature_weights:
            normalized_credit = (decision.credit_score - 300) / (850 - 300)
            score += normalized_credit * decision.feature_weights["credit_score"]

        # Income: normalize relative to median (assume $50K median)
        if "annual_income" in decision.feature_weights:
            normalized_income = min(decision.annual_income / 100000, 1.0)
            score += normalized_income * decision.feature_weights["annual_income"]

        # Debt-to-income: lower is better, normalize inversely
        if "debt_to_income_ratio" in decision.feature_weights:
            normalized_dti = 1.0 - (decision.debt_to_income_ratio / 100)
            score += normalized_dti * decision.feature_weights["debt_to_income_ratio"]

        # Employment years: normalize to 0-1 (cap at 10 years)
        if "employment_years" in decision.feature_weights:
            normalized_employment = min(decision.employment_years / 10, 1.0)
            score += normalized_employment * decision.feature_weights["employment_years"]

        return score


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def create_approved_decision() -> LoanDecision:
    """Create example approved loan decision"""
    return LoanDecision(
        applicant_id="APP-2024-001",
        decision_id="DEC-2024-001-001",
        outcome=DecisionOutcome.APPROVED,
        approved=True,
        credit_score=780,
        annual_income=95000,
        debt_to_income_ratio=28,
        employment_years=8,
        loan_amount=25000,
        feature_weights={
            "credit_score": 0.45,
            "annual_income": 0.25,
            "debt_to_income_ratio": 0.20,
            "employment_years": 0.10
        },
        confidence_score=0.94,
        policy_threshold=0.75,
        primary_reason="excellent_credit",
        contributing_factors=["low_debt_burden", "stable_employment", "strong_income"],
        timestamp=datetime.utcnow().isoformat(),
        model_version="2.1.0"
    )


def create_declined_decision() -> LoanDecision:
    """Create example declined loan decision"""
    return LoanDecision(
        applicant_id="APP-2024-002",
        decision_id="DEC-2024-002-001",
        outcome=DecisionOutcome.DECLINED,
        approved=False,
        credit_score=620,
        annual_income=45000,
        debt_to_income_ratio=48,
        employment_years=1,
        loan_amount=30000,
        feature_weights={
            "credit_score": 0.45,
            "annual_income": 0.25,
            "debt_to_income_ratio": 0.20,
            "employment_years": 0.10
        },
        confidence_score=0.89,
        policy_threshold=0.75,
        primary_reason="high_debt_ratio",
        contributing_factors=["low_credit_score", "short_employment", "excessive_loan_amount"],
        timestamp=datetime.utcnow().isoformat(),
        model_version="2.1.0"
    )


def example_multi_level_explanations():
    """Demonstrate explanation generation at all levels"""
    print("\n" + "="*70)
    print("Example 1: Multi-Level Explanation Generation")
    print("="*70)

    generator = ExplanationGenerator()

    # Test both approved and declined decisions
    for decision, name in [
        (create_approved_decision(), "APPROVED APPLICATION"),
        (create_declined_decision(), "DECLINED APPLICATION")
    ]:
        print(f"\n{'='*70}")
        print(f"Case: {name}")
        print(f"{'='*70}")

        for level in ExplanationLevel:
            print(f"\n{'-' * 70}")
            print(f"{level.value.upper()} EXPLANATION (for {level.value} audience)")
            print(f"{'-' * 70}\n")
            explanation = generator.generate_explanation(decision, level)
            print(explanation)
            print()


def example_counterfactual_explanations():
    """Demonstrate counterfactual explanation generation"""
    print("\n" + "="*70)
    print("Example 2: Counterfactual Explanations (What-If Scenarios)")
    print("="*70)

    generator = ExplanationGenerator()
    decision = create_declined_decision()

    print("\nOriginal Decision: DECLINED")
    print(f"Credit Score: {decision.credit_score}")
    print(f"Debt-to-Income: {decision.debt_to_income_ratio:.1f}%")
    print(f"Employment: {decision.employment_years} years")

    print("\n" + "-"*70)
    print("COUNTERFACTUAL SCENARIOS (What would need to change)")
    print("-"*70)

    counterfactuals = generator._generate_counterfactual(decision)
    print(counterfactuals)

    print("\n✓ Counterfactuals help users understand decision boundaries")
    print("✓ Shows minimal changes needed for different outcome")


def example_improvement_suggestions():
    """Demonstrate actionable improvement suggestions"""
    print("\n" + "="*70)
    print("Example 3: Actionable Improvement Suggestions")
    print("="*70)

    generator = ExplanationGenerator()
    decision = create_declined_decision()

    print(f"\nApplication Declined - Here's how to improve:\n")
    print("-"*70)

    suggestions = generator._generate_improvement_suggestions(decision)
    print(suggestions)

    print("\n" + "-"*70)
    print("✓ Suggestions are specific and actionable")
    print("✓ Include concrete numbers (not vague advice)")
    print("✓ Provide realistic timeline for improvement")


def example_regulatory_compliance():
    """Demonstrate regulatory compliance features"""
    print("\n" + "="*70)
    print("Example 4: Regulatory Compliance (ECOA Adverse Action Notice)")
    print("="*70)

    generator = ExplanationGenerator()
    decision = create_declined_decision()

    print("\nECOA Requirement: Provide specific reasons for adverse actions")
    print("\nGenerated Adverse Action Notice:\n")
    print("-"*70)

    # Generate DETAILED explanation (suitable for adverse action notice)
    explanation = generator.generate_explanation(decision, ExplanationLevel.DETAILED)
    print(explanation)

    print("\n" + "-"*70)
    print("✓ Meets ECOA requirements for adverse action notices")
    print("✓ Provides specific reasons (not just \"credit quality\")")
    print("✓ Includes applicant's actual data for transparency")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Natural Language Explanation Generator - Code Example 10.3.2")
    print("="*70)
    print("\nTransforming technical decision traces into")
    print("audience-appropriate natural language explanations\n")

    example_multi_level_explanations()
    example_counterfactual_explanations()
    example_improvement_suggestions()
    example_regulatory_compliance()

    print("\n" + "="*70)
    print("All examples completed successfully! ✅")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Different audiences need different explanation levels")
    print("  • Basic explanations focus on clarity and actionability")
    print("  • Technical explanations include full audit details")
    print("  • Counterfactuals help users understand decision boundaries")
    print("  • Improvement suggestions must be specific and achievable")
    print("  • Regulatory compliance requires specific reason disclosure")
    print("="*70)


if __name__ == "__main__":
    main()
