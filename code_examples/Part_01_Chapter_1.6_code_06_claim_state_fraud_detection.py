class ClaimState(TypedDict):
    claim_id: str
    documents: dict  # Preserved after extraction
    fraud_score: Optional[float]  # None until fraud detection succeeds
    policy_valid: Optional[bool]  # None until validation succeeds
    damage_amount: Optional[float]
    payout: Optional[float]
    current_step: str  # Tracks workflow progress
    retry_count: int  # Prevents infinite retry loops

def fraud_detection_with_retry(state: ClaimState) -> ClaimState:
    """Fraud detection with automatic retry on transient failures."""
    try:
        fraud_score = call_fraud_api(state["documents"])
        return {
            **state,
            "fraud_score": fraud_score,
            "current_step": "policy_validation",
            "retry_count": 0  # Reset on success
        }
    except API5xxError as e:
        if state["retry_count"] >= 3:
            raise MaxRetriesError("Fraud API failed after 3 attempts")

        logger.warning(f"Fraud API failure, retry {state['retry_count'] + 1}/3")
        return {
            **state,
            "retry_count": state["retry_count"] + 1
            # current_step remains "fraud_detection" for retry routing
        }
