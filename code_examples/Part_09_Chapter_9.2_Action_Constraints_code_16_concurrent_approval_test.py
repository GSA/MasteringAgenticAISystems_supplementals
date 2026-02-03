import asyncio

class EscalationScheduler:
    """
    Background service scanning for pending approval requests that need
    escalation due to SLA breach.

    Runs as separate process from agents and approval API to ensure
    escalations occur even if agent processes are down.
    """

    def __init__(
        self,
        approval_db,
        escalation_rule: EscalationRule,
        check_interval_seconds: int = 60
    ):
        self.approval_db = approval_db
        self.escalation_rule = escalation_rule
        self.check_interval = check_interval_seconds
        self.running = False

    async def start(self):
        """Start continuous escalation monitoring"""
        self.running = True

        while self.running:
            try:
                await self._check_pending_requests()
            except Exception as e:
                # Log error but continue running
                logger.error(f"Escalation check failed: {e}")

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _check_pending_requests(self):
        """
        Scan all pending approval requests and escalate those exceeding SLA.

        Queries database for requests in PENDING status with timeout_at
        in the past, indicating they've exceeded SLA without approval.
        """
        # Query pending requests approaching or past timeout
        pending_requests = await self.approval_db.get_pending_near_timeout(
            threshold_minutes=5  # Check 5 minutes before timeout
        )

        for request in pending_requests:
            try:
                escalated = await self.escalation_rule.check_and_escalate(
                    request
                )
                if escalated:
                    logger.info(
                        f"Escalated request {request.request_id} from "
                        f"{request.approver_id} after SLA breach"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to escalate request {request.request_id}: {e}"
                )