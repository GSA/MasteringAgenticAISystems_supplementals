class HITLWorkflow:
    """
    Human-in-the-loop approval orchestration with risk-based routing,
    SLA tracking, and escalation protocols.
    """

    def __init__(self, approval_db):
        self.approval_db = approval_db  # Persistent storage for requests

        # SLA windows by risk level (hours until escalation)
        self.sla_hours = {
            'low': 24,        # Low-risk: 1 day response window
            'medium': 4,      # Medium-risk: 4 hour response
            'high': 1,        # High-risk: 1 hour response
            'critical': 0.5   # Critical: 30 minute response
        }

        # Approval authority by risk level
        self.approver_mapping = {
            'low': 'team_lead',
            'medium': 'manager',
            'high': 'director',
            'critical': 'vp_engineering'
        }

    async def request_approval(
        self,
        agent_id: str,
        action: str,
        details: Dict[str, Any],
        risk_level: str
    ) -> ApprovalRequest:
        """
        Create approval request with appropriate routing and SLA.

        This is the primary API for agents needing approval. The agent
        provides decision context; the workflow handles routing, notification,
        and timeout scheduling.
        """
        # Determine approver based on risk level
        approver_id = self._assign_approver(risk_level)

        # Calculate timeout based on SLA
        sla_hours = self.sla_hours.get(risk_level, 4)
        timeout_at = datetime.utcnow() + timedelta(hours=sla_hours)

        # Create request
        request = ApprovalRequest(
            request_id=self._generate_id(),
            agent_id=agent_id,
            action=action,
            details=details,
            risk_level=risk_level,
            approver_id=approver_id,
            requested_at=datetime.utcnow(),
            timeout_at=timeout_at
        )

        # Persist to database
        await self.approval_db.save(request)

        # Notify assigned approver (email, Slack, SMS based on urgency)
        await self._notify_approver(request)

        # Schedule escalation check at SLA deadline
        await self._schedule_escalation(request)

        return request

    def _assign_approver(self, risk_level: str) -> str:
        """
        Route request to appropriate approver based on risk level.

        Production implementations might query organizational directory
        to find on-call personnel, load-balance across available approvers,
        or escalate to backup if primary approver is unavailable.
        """
        return self.approver_mapping.get(risk_level, 'manager')