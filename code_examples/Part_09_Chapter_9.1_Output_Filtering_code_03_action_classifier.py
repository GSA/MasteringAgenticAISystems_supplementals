# Example: Layer 3 Action Classification
from enum import Enum
from typing import Dict, Any, Tuple

class ActionRisk(Enum):
    AUTONOMOUS = "autonomous"
    APPROVAL_REQUIRED = "approval_required"
    PROHIBITED = "prohibited"

class ActionController:
    """Layer 3: Control what actions agents can take"""

    def __init__(self):
        self.action_policies: Dict[str, ActionRisk] = {
            'read_data': ActionRisk.AUTONOMOUS,
            'search': ActionRisk.AUTONOMOUS,
            'send_email': ActionRisk.APPROVAL_REQUIRED,
            'modify_database': ActionRisk.APPROVAL_REQUIRED,
            'delete_data': ActionRisk.PROHIBITED,
            'financial_transaction': ActionRisk.PROHIBITED
        }

    def check_action(self, action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if action is allowed.
        Returns (is_allowed, reason)
        """
        risk_level = self.action_policies.get(action, ActionRisk.PROHIBITED)

        if risk_level == ActionRisk.PROHIBITED:
            return False, f"Action '{action}' is prohibited"

        if risk_level == ActionRisk.APPROVAL_REQUIRED:
            if not context.get('human_approved', False):
                return False, f"Action '{action}' requires human approval"

        return True, ""

# Usage
controller = ActionController()
action = "financial_transaction"
context = {'user_id': '123', 'human_approved': False}
is_allowed, reason = controller.check_action(action, context)
if not is_allowed:
    print(f"Action blocked: {reason}")
