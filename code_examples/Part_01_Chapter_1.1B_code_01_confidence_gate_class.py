class ConfidenceGate:
    """Route agent decisions to appropriate control patterns based on confidence"""

    def __init__(self, auto_threshold=0.85, review_threshold=0.65):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

    def should_escalate(self, decision):
        """Determine if human review is needed and what type

        Returns:
            dict with escalation decision and recommended UI pattern
        """
        confidence = decision.confidence_score

        if confidence >= self.auto_threshold:
            # High confidence: notification pattern
            return {
                "escalate": False,
                "pattern": "notification",
                "reason": f"High confidence: {confidence:.1%}",
                "ui_action": "execute_and_notify"
            }
        elif confidence >= self.review_threshold:
            # Medium confidence: approval pattern
            return {
                "escalate": True,
                "pattern": "approval",
                "reason": f"Moderate confidence: {confidence:.1%}",
                "ui_action": "show_approval_dialog"
            }
        else:
            # Low confidence: require review with full context
            return {
                "escalate": True,
                "pattern": "approval_with_alternatives",
                "reason": f"Low confidence: {confidence:.1%}",
                "ui_action": "show_detailed_review"
            }
