async def wait_for_approval(
    self,
    request_id: str,
    timeout_seconds: int = 3600
) -> bool:
    """
    Wait for approval with timeout, handling three outcomes:
    - Approval granted: return True
    - Approval rejected: return False
    - Timeout exceeded: return False (treat as implicit rejection)

    This is the blocking call that pauses agent execution. In production,
    prefer event-driven patterns (message queues) over polling for efficiency.
    """
    request = await self.approval_db.get(request_id)
    if not request:
        raise ValueError(f"Request {request_id} not found")

    start_time = datetime.utcnow()
    poll_interval = 1  # Start with 1 second polls

    while True:
        # Reload request from database (approver may have updated it)
        request = await self.approval_db.get(request_id)

        # Check for terminal states
        if request.status == ApprovalStatus.APPROVED:
            return True
        elif request.status == ApprovalStatus.REJECTED:
            return False
        elif request.status == ApprovalStatus.EXPIRED:
            return False

        # Check if timeout exceeded
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if elapsed > timeout_seconds:
            # Mark request as expired
            request.status = ApprovalStatus.EXPIRED
            await self.approval_db.update(request)
            return False

        # Exponential backoff: gradually increase poll interval
        await asyncio.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 30)  # Cap at 30 seconds