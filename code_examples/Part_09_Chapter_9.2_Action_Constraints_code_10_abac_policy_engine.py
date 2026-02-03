class ABACPolicyEngine:
    """Attribute-based access control with context-aware decisions"""

    def is_authorized(
        self,
        agent: dict,
        resource: dict,
        action: str,
        context: dict
    ) -> bool:
        """Evaluate access based on multiple contextual attributes"""
        # Check device trust level
        if context.get('device_trust_level', 0) < 0.8:
            return False  # Block from untrusted devices

        # Check geographic location
        if context.get('country') in ['CN', 'RU', 'KP']:
            return False  # Block from restricted regions

        # Check time-of-day restrictions
        hour = context.get('hour')
        if action in ['delete', 'modify'] and (hour < 9 or hour > 17):
            return False  # Block destructive actions outside business hours

        # Check resource sensitivity
        if resource.get('sensitivity') == 'high':
            if agent.get('trust_score', 0) < 0.9:
                return False  # High-sensitivity resources require high trust

        # Verify action aligns with stated user intent
        if not self._aligns_with_user_intent(action, context):
            return False

        return True  # All checks passed

    def _aligns_with_user_intent(self, action: str, context: dict) -> bool:
        """Verify action aligns with user's stated intent"""
        user_intent = context.get('user_intent', '')
        # Sophisticated intent analysis would be implemented here
        # For example, deleting files shouldn't happen when intent is "analyze report"
        return True  # Simplified for illustration

# Usage: Authorization considers multiple contextual factors
policy = ABACPolicyEngine()
authorized = policy.is_authorized(
    agent={'id': 'agent_123', 'trust_score': 0.9},
    resource={'type': 'database', 'sensitivity': 'high'},
    action='modify',
    context={
        'device_trust_level': 0.95,
        'country': 'US',
        'hour': 14,  # 2 PM
        'user_intent': 'update customer address'
    }
)