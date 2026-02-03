from typing import List
from datetime import datetime

class AgentController:
    """Human intervention interface for monitored agents"""

    def __init__(self, agent):
        self.agent = agent
        self.circuit_breaker_threshold = 5
        self.intervention_log: List[dict] = []
        self.circuit_open = False

    def pause_agent(self, reason: str, authorized_by: str) -> dict:
        """Emergency pause requiring human authorization"""
        intervention = {
            "action": "pause",
            "reason": reason,
            "authorized_by": authorized_by,
            "timestamp": datetime.now().isoformat()
        }
        self.intervention_log.append(intervention)
        self.circuit_open = True

        print(f"[INTERVENTION] Agent paused by {authorized_by}: {reason}")
        return {"status": "paused", "intervention": intervention}

    def resume_agent(self, authorized_by: str) -> dict:
        """Resume after human review"""
        intervention = {
            "action": "resume",
            "authorized_by": authorized_by,
            "timestamp": datetime.now().isoformat()
        }
        self.intervention_log.append(intervention)
        self.circuit_open = False

        print(f"[INTERVENTION] Agent resumed by {authorized_by}")
        return {"status": "active", "intervention": intervention}

    def adjust_confidence_threshold(self, new_threshold: float,
                                   reason: str) -> dict:
        """Adjust escalation threshold (low-impact intervention)"""
        old_threshold = getattr(self.agent, 'escalation_threshold', 0.7)
        self.agent.escalation_threshold = new_threshold

        intervention = {
            "action": "threshold_adjustment",
            "old_value": old_threshold,
            "new_value": new_threshold,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "auto_approved": True  # Low-impact change
        }
        self.intervention_log.append(intervention)

        print(f"[CONFIG] Escalation threshold adjusted: " \
              f"{old_threshold:.2f} → {new_threshold:.2f}")
        return {"status": "adjusted", "intervention": intervention}

    def check_circuit_breaker(self, recent_errors: int) -> bool:
        """Automatic circuit breaker on accumulated errors"""
        if recent_errors >= self.circuit_breaker_threshold:
            self.pause_agent(
                reason=f"Circuit breaker: {recent_errors} errors " \
                       f"in monitoring window",
                authorized_by="system_automatic"
            )
            return True
        return False
