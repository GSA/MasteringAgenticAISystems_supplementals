def approve_request(
    self,
    request_id: str,
    approver_id: str,
    notes: Optional[str] = None
) -> None:
    """
    Record approval decision and unblock waiting agent.

    This would typically be called by approval dashboard API when
    approver clicks "Approve" button after reviewing request context.
    """
    request = self.approval_db.get(request_id)
    if not request:
        raise ValueError(f"Request {request_id} not found")

    if request.status != ApprovalStatus.PENDING:
        raise ValueError(
            f"Request {request_id} already {request.status.value}"
        )

    # Update request with approval
    request.status = ApprovalStatus.APPROVED
    request.approved_at = datetime.utcnow()
    request.approved_by = approver_id
    if notes:
        request.details['approval_notes'] = notes

    self.approval_db.update(request)

    # Trigger notification that waiting agent can proceed
    self._notify_agent_approval_granted(request)

def reject_request(
    self,
    request_id: str,
    approver_id: str,
    reason: str
) -> None:
    """
    Record rejection decision with mandatory reason.

    Rejection reason is critical for audit trail and agent learning.
    If agents repeatedly request approvals that get rejected for
    similar reasons, this suggests agent policy needs refinement.
    """
    request = self.approval_db.get(request_id)
    if not request:
        raise ValueError(f"Request {request_id} not found")

    if request.status != ApprovalStatus.PENDING:
        raise ValueError(
            f"Request {request_id} already {request.status.value}"
        )

    # Update request with rejection
    request.status = ApprovalStatus.REJECTED
    request.approved_at = datetime.utcnow()
    request.approved_by = approver_id
    request.rejection_reason = reason

    self.approval_db.update(request)

    # Notify waiting agent of rejection
    self._notify_agent_approval_rejected(request)