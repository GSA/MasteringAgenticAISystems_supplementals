class RiskGate:
    """Escalate high-risk decisions regardless of confidence"""

    RISK_THRESHOLDS = {
        "financial": 10000,      # Over $10k needs approval
        "data_deletion": "any",  # Any deletion needs approval
        "external_api": "any",   # External calls need approval
        "customer_facing": 500   # Messages to 500+ customers need approval
    }

    def assess_decision_risk(self, decision):
        """Evaluate risk level and determine control requirements

        Returns:
            dict with risk score and control recommendations
        """
        risk_score = 0
        risk_factors = []

        if hasattr(decision, 'financial_amount'):
            if decision.financial_amount > self.RISK_THRESHOLDS["financial"]:
                risk_score += 5
                risk_factors.append(
                    f"High financial impact: ${decision.financial_amount:,.0f}"
                )

        if hasattr(decision, 'involves_deletion') and decision.involves_deletion:
            risk_score += 5
            risk_factors.append("Irreversible data deletion")

        if hasattr(decision, 'calls_external_api') and decision.calls_external_api:
            risk_score += 3
            risk_factors.append("External API interaction")

        if hasattr(decision, 'customer_reach'):
            if decision.customer_reach > self.RISK_THRESHOLDS["customer_facing"]:
                risk_score += 4
                risk_factors.append(
                    f"Broad customer impact: {decision.customer_reach} recipients"
                )

        return {
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "requires_approval": risk_score >= 5,
            "requires_monitoring": risk_score >= 8,
            "ui_action": self._determine_ui_action(risk_score)
        }

    def _determine_ui_action(self, risk_score):
        """Map risk score to UI pattern"""
        if risk_score >= 8:
            return "show_monitoring_dashboard"
        elif risk_score >= 5:
            return "show_approval_with_risk_details"
        elif risk_score >= 3:
            return "show_notification_with_risk_summary"
        else:
            return "execute_silently"
