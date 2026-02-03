from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class ApprovalStatus(Enum):
    """Approval request lifecycle states"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ApprovalRequest:
    """
    Serializable approval request capturing complete decision context.
    Persisted to database to survive agent process restarts.
    """
    request_id: str              # Unique identifier for idempotency
    agent_id: str                # Which agent created this request
    action: str                  # What operation needs approval
    details: Dict[str, Any]      # Full context: parameters, reasoning, data
    risk_level: str              # Low/medium/high/critical for routing
    approver_id: str             # Who must approve (determined by risk)
    requested_at: datetime       # When request was created
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None  # Actual approver (may differ from assigned)
    rejection_reason: Optional[str] = None
    timeout_at: datetime = None  # When request expires if unapproved

    def is_expired(self) -> bool:
        """Check if request exceeded SLA without approval"""
        return datetime.utcnow() > self.timeout_at if self.timeout_at else False