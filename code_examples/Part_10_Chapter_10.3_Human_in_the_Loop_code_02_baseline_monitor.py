from collections import deque
from typing import Deque, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PerformanceBaseline:
    """Statistical baseline for agent performance"""
    mean_confidence: float
    std_confidence: float
    mean_latency: float
    std_latency: float
    escalation_rate: float
    error_rate: float
    window_size: int
    last_updated: datetime

class BaselineMonitor:
    """Maintains rolling performance baselines"""

    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.telemetry_buffer: Deque = deque()
        self.baseline: Optional[PerformanceBaseline] = None

    def update(self, telemetry):
        """Add telemetry and update baseline"""
        self.telemetry_buffer.append(telemetry)

        # Remove old telemetry outside window
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        while self.telemetry_buffer and \
              self.telemetry_buffer[0].timestamp < cutoff:
            self.telemetry_buffer.popleft()

        # Recalculate baseline
        self._recalculate_baseline()

    def _recalculate_baseline(self):
        """Calculate statistical baseline from buffered data"""
        if len(self.telemetry_buffer) < 100:
            return  # Insufficient data

        # Extract metrics
        confidences = [t.confidence_score
                      for t in self.telemetry_buffer
                      if t.outcome == "success"]
        latencies = [t.latency_ms for t in self.telemetry_buffer]
        total = len(self.telemetry_buffer)
        escalated = sum(1 for t in self.telemetry_buffer
                       if t.decision_type == "escalated")
        errors = sum(1 for t in self.telemetry_buffer
                    if t.outcome == "failure")

        self.baseline = PerformanceBaseline(
            mean_confidence=np.mean(confidences),
            std_confidence=np.std(confidences),
            mean_latency=np.mean(latencies),
            std_latency=np.std(latencies),
            escalation_rate=escalated / total if total > 0 else 0,
            error_rate=errors / total if total > 0 else 0,
            window_size=len(self.telemetry_buffer),
            last_updated=datetime.now()
        )

    def detect_anomaly(self, telemetry) -> dict:
        """Check if telemetry represents anomalous behavior"""
        if not self.baseline:
            return {"is_anomaly": False, "reason": "baseline_unavailable"}

        anomalies = []

        # Confidence anomaly (2 standard deviations)
        if telemetry.outcome == "success":
            z_confidence = abs(telemetry.confidence_score -
                             self.baseline.mean_confidence) / \
                          max(self.baseline.std_confidence, 0.01)
            if z_confidence > 2.0:
                anomalies.append(f"confidence_{z_confidence:.2f}_sigma")

        # Latency anomaly (3 standard deviations for tail events)
        z_latency = abs(telemetry.latency_ms -
                       self.baseline.mean_latency) / \
                   max(self.baseline.std_latency, 1.0)
        if z_latency > 3.0:
            anomalies.append(f"latency_{z_latency:.2f}_sigma")

        return {
            "is_anomaly": len(anomalies) > 0,
            "anomalies": anomalies,
            "baseline": self.baseline
        }
