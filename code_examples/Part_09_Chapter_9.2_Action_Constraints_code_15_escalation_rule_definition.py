class EscalationRule:
    """
    Automatic escalation when approval requests exceed SLA without response.

    Prevents approval bottlenecks from paralyzing agent operations while
    maintaining human oversight by routing to increasingly senior approvers.
    """

    def __init__(self, approval_db, notifier):
        self.approval_db = approval_db
        self.notifier = notifier

        # Organizational hierarchy for escalation routing
        self.escalation_chain = {
            'team_lead': 'manager',
            'manager': 'director',
            'director': 'vp_engineering',
            'vp_engineering': 'cto',
            'cto': 'auto_reject'  # Final fallback
        }

    async def check_and_escalate(self, request: ApprovalRequest) -> bool:
        """
        Check if request breached SLA and escalate if needed.

        This is called periodically by background job that scans for
        pending requests approaching their timeout_at deadline.

        Returns True if escalated, False if still within SLA window.
        """
        # Check if SLA deadline has passed
        if not self._is_sla_breached(request):
            return False

        # Get next approver in escalation chain
        next_approver = self.escalation_chain.get(request.approver_id)

        if not next_approver:
            raise ValueError(
                f"No escalation path for approver {request.approver_id}"
            )

        # Handle final fallback
        if next_approver == 'auto_reject':
            await self._auto_reject_on_timeout(request)
            return True

        # Escalate to next level
        await self._escalate_to(request, next_approver)
        return True

    def _is_sla_breached(self, request: ApprovalRequest) -> bool:
        """
        Determine if request exceeded SLA window without approval.

        Uses timeout_at deadline calculated during request creation based
        on risk level and organizational SLA policies.
        """
        if request.status != ApprovalStatus.PENDING:
            return False  # Already resolved, no escalation needed

        return datetime.utcnow() >= request.timeout_at

    async def _escalate_to(
        self,
        request: ApprovalRequest,
        next_approver: str
    ) -> None:
        """
        Route request to next approver in escalation chain.

        Updates request record, notifies new approver with context about
        why escalation occurred, and extends timeout to give new approver
        their own SLA window.
        """
        # Record escalation in request details for audit trail
        escalation_history = request.details.get('escalations', [])
        escalation_history.append({
            'from': request.approver_id,
            'to': next_approver,
            'escalated_at': datetime.utcnow().isoformat(),
            'reason': 'SLA breach - no response from original approver'
        })
        request.details['escalations'] = escalation_history

        # Update approver and extend timeout
        original_approver = request.approver_id
        request.approver_id = next_approver

        # Give new approver fresh SLA window (typically shorter than original)
        escalation_sla_hours = 1  # Escalated requests get tighter SLA
        request.timeout_at = datetime.utcnow() + timedelta(
            hours=escalation_sla_hours
        )

        await self.approval_db.update(request)

        # Notify new approver with escalation context
        await self.notifier.send_escalation_notice(
            approver_id=next_approver,
            request=request,
            original_approver=original_approver,
            escalation_reason='SLA breach'
        )

    async def _auto_reject_on_timeout(
        self,
        request: ApprovalRequest
    ) -> None:
        """
        Automatically reject request that exhausted escalation chain.

        Conservative fallback: if even highest-level approvers don't respond,
        reject rather than auto-approve. Appropriate for high-risk actions
        where unauthorized execution creates liability.

        Some organizations configure auto-approval for low-risk actions to
        prioritize velocity. This requires explicit policy decision based on
        action reversibility and consequence severity.
        """
        request.status = ApprovalStatus.EXPIRED
        request.rejection_reason = (
            "Automatic rejection: approval request exceeded all escalation "
            "levels without response. Original approver: "
            f"{request.details['escalations'][0]['from']}"
        )

        await self.approval_db.update(request)

        # Alert operations team about escalation failure
        await self.notifier.send_alert(
            severity='high',
            message=(
                f"Approval request {request.request_id} auto-rejected after "
                f"exhausting escalation chain. This indicates approval process "
                f"breakdown requiring investigation."
            ),
            request_details=request.details
        )