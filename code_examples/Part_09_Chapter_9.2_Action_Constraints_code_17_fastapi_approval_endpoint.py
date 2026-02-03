from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Approval Dashboard API")

class ApprovalDecision(BaseModel):
    """Request body for approval/rejection submission"""
    request_id: str
    decision: str  # "approve" or "reject"
    approver_id: str
    reason: Optional[str] = None  # Required for rejection, optional for approval

class ApprovalSummary(BaseModel):
    """Lightweight request summary for queue display"""
    request_id: str
    agent_id: str
    action: str
    risk_level: str
    requested_at: str
    time_remaining_minutes: int

@app.get("/approvals/pending", response_model=List[ApprovalSummary])
async def get_pending_approvals(approver_id: str):
    """
    Get queue of pending approval requests assigned to specified approver.

    Returns lightweight summaries for queue display. Approver clicks on
    specific request to see full details via /approvals/{request_id} endpoint.
    """
    workflow = HITLWorkflow(approval_db)

    # Query all pending requests for this approver
    pending = await approval_db.get_pending_for_approver(approver_id)

    # Transform to summary format
    summaries = []
    for request in pending:
        time_remaining = (
            request.timeout_at - datetime.utcnow()
        ).total_seconds() / 60

        summaries.append(ApprovalSummary(
            request_id=request.request_id,
            agent_id=request.agent_id,
            action=request.action,
            risk_level=request.risk_level,
            requested_at=request.requested_at.isoformat(),
            time_remaining_minutes=int(time_remaining)
        ))

    # Sort by urgency: critical first, then by time remaining
    summaries.sort(
        key=lambda x: (
            0 if x.risk_level == 'critical' else 1,
            x.time_remaining_minutes
        )
    )

    return summaries

@app.get("/approvals/{request_id}")
async def get_approval_details(request_id: str):
    """
    Get full request context for detailed review.

    Includes agent reasoning, data analyzed, policy implications,
    consequence estimates—everything approver needs to make informed decision.
    """
    request = await approval_db.get(request_id)

    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    # Return complete request with all context
    return {
        'request_id': request.request_id,
        'agent_id': request.agent_id,
        'action': request.action,
        'risk_level': request.risk_level,
        'requested_at': request.requested_at.isoformat(),
        'timeout_at': request.timeout_at.isoformat(),
        'details': request.details,  # Full context from agent
        'status': request.status.value,
        'escalation_history': request.details.get('escalations', [])
    }

@app.post("/approvals/decide")
async def decide_approval(decision: ApprovalDecision):
    """
    Submit approval or rejection decision.

    Records decision in approval database, unblocks waiting agent,
    and creates audit trail entry.
    """
    workflow = HITLWorkflow(approval_db)

    if decision.decision == "approve":
        workflow.approve_request(
            decision.request_id,
            decision.approver_id,
            notes=decision.reason
        )
        return {"status": "approved", "request_id": decision.request_id}

    elif decision.decision == "reject":
        if not decision.reason:
            raise HTTPException(
                status_code=400,
                detail="Rejection reason required"
            )

        workflow.reject_request(
            decision.request_id,
            decision.approver_id,
            decision.reason
        )
        return {"status": "rejected", "request_id": decision.request_id}

    else:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'approve' or 'reject'"
        )