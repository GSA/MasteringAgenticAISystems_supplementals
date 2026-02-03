class ApprovalWorkflow:
    """Require human approval for high-risk actions"""

    def __init__(self):
        self.pending_approvals = {}

    async def request_approval(
        self,
        action: str,
        details: dict,
        approver_id: str
    ) -> bool:
        """Request human approval and wait for response"""
        approval_id = self._generate_id()

        # Create approval request with context
        self.pending_approvals[approval_id] = {
            'action': action,
            'details': details,
            'approver_id': approver_id,
            'status': 'pending',
            'requested_at': datetime.utcnow()
        }

        # Notify approver through appropriate channel
        await self._notify_approver(approver_id, approval_id, action, details)

        # Wait for approval with timeout (default 1 hour)
        approved = await self._wait_for_approval(
            approval_id,
            timeout_seconds=3600
        )

        return approved

# Usage in critical operations
workflow = ApprovalWorkflow()
approved = await workflow.request_approval(
    action='delete_production_database',
    details={'database': 'customer_data', 'reason': 'cleanup'},
    approver_id='senior_dba_123'
)

if not approved:
    raise PermissionError("Action not approved by required authority")