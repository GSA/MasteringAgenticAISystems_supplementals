from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import logging

@dataclass
class AgentTelemetry:
    """Structured telemetry for agent operations"""
    request_id: str
    timestamp: datetime
    decision_type: str  # "answered", "escalated", "error"
    confidence_score: float  # 0.0-1.0
    latency_ms: int
    tools_invoked: List[str]
    outcome: str  # "success", "failure", "escalated"
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "decision_type": self.decision_type,
            "confidence": self.confidence_score,
            "latency": self.latency_ms,
            "tools": self.tools_invoked,
            "outcome": self.outcome,
            "error": self.error_message
        }

class MonitoredAgent:
    """Agent with comprehensive telemetry"""

    def __init__(self):
        self.logger = logging.getLogger("agent.telemetry")
        self.request_count = 0

    def process_request(self, user_query: str) -> dict:
        """Process user request with telemetry"""
        start_time = datetime.now()
        request_id = f"req_{self.request_count}"
        self.request_count += 1

        try:
            # Agent decision-making logic
            tools_used = []

            # Retrieve relevant docs
            tools_used.append("doc_retrieval")
            docs = self._retrieve_docs(user_query)

            # Generate response
            tools_used.append("response_generation")
            response, confidence = self._generate_response(docs)

            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds() * 1000

            # Determine outcome
            if confidence < 0.7:
                decision = "escalated"
                outcome = "escalated"
            else:
                decision = "answered"
                outcome = "success"

            # Emit telemetry
            telemetry = AgentTelemetry(
                request_id=request_id,
                timestamp=start_time,
                decision_type=decision,
                confidence_score=confidence,
                latency_ms=int(latency),
                tools_invoked=tools_used,
                outcome=outcome
            )
            self.logger.info(telemetry.to_dict())

            return {
                "response": response,
                "confidence": confidence,
                "escalated": decision == "escalated"
            }

        except Exception as e:
            # Emit error telemetry
            latency = (datetime.now() - start_time).total_seconds() * 1000
            telemetry = AgentTelemetry(
                request_id=request_id,
                timestamp=start_time,
                decision_type="error",
                confidence_score=0.0,
                latency_ms=int(latency),
                tools_invoked=tools_used,
                outcome="failure",
                error_message=str(e)
            )
            self.logger.error(telemetry.to_dict())
            raise
