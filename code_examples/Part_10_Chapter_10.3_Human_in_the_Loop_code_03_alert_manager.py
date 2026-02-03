from enum import Enum
from typing import List, Deque
from collections import deque
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass
from datetime import datetime

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Structured alert"""
    severity: AlertSeverity
    title: str
    description: str
    telemetry: object
    anomalies: List[str]
    timestamp: datetime

class AlertManager:
    """Multi-level alert routing and correlation"""

    def __init__(self):
        self.alert_buffer: Deque[Alert] = deque(maxlen=1000)
        self.alert_counts = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.CRITICAL: 0
        }

    def process_anomaly(self, telemetry, anomaly_result: dict):
        """Generate appropriate alert from anomaly"""
        if not anomaly_result["is_anomaly"]:
            return

        # Determine severity
        severity = self._classify_severity(telemetry,
                                          anomaly_result["anomalies"])

        # Create alert
        alert = Alert(
            severity=severity,
            title=f"{severity.value.upper()}: Agent anomaly detected",
            description=self._format_description(telemetry,
                                                anomaly_result),
            telemetry=telemetry,
            anomalies=anomaly_result["anomalies"],
            timestamp=datetime.now()
        )

        # Store and route
        self.alert_buffer.append(alert)
        self.alert_counts[severity] += 1
        self._route_alert(alert)

    def _classify_severity(self, telemetry,
                          anomalies: List[str]) -> AlertSeverity:
        """Classify alert severity"""
        # Critical: errors or extreme anomalies
        if telemetry.outcome == "failure":
            return AlertSeverity.CRITICAL
        if any("3.0_sigma" in a or "4.0_sigma" in a
               for a in anomalies):
            return AlertSeverity.CRITICAL

        # Warning: moderate anomalies
        if len(anomalies) >= 2:
            return AlertSeverity.WARNING

        # Info: single mild anomaly
        return AlertSeverity.INFO

    def _format_description(self, telemetry,
                           anomaly_result: dict) -> str:
        """Format human-readable alert description"""
        baseline = anomaly_result.get("baseline")
        parts = [
            f"Request {telemetry.request_id} at " \
            f"{telemetry.timestamp.isoformat()}",
            f"Decision: {telemetry.decision_type}",
            f"Confidence: {telemetry.confidence_score:.3f} " \
            f"(baseline: {baseline.mean_confidence:.3f} ± " \
            f"{baseline.std_confidence:.3f})" if baseline else "",
            f"Latency: {telemetry.latency_ms}ms " \
            f"(baseline: {baseline.mean_latency:.0f} ± " \
            f"{baseline.std_latency:.0f}ms)" if baseline else "",
            f"Anomalies detected: {', '.join(anomaly_result['anomalies'])}"
        ]
        return "\n".join(p for p in parts if p)

    def _route_alert(self, alert: Alert):
        """Route alert based on severity"""
        if alert.severity == AlertSeverity.CRITICAL:
            self._send_immediate_notification(alert)
        elif alert.severity == AlertSeverity.WARNING:
            self._update_dashboard(alert)
        else:  # INFO
            self._log_for_analysis(alert)

    def _send_immediate_notification(self, alert: Alert):
        """Send email/SMS for critical alerts"""
        print(f"[CRITICAL ALERT] {alert.title}")
        print(alert.description)
        # In production: send via email/PagerDuty/Slack

    def _update_dashboard(self, alert: Alert):
        """Update monitoring dashboard"""
        print(f"[WARNING] {alert.title}")
        # In production: push to dashboard UI

    def _log_for_analysis(self, alert: Alert):
        """Log informational alerts"""
        # Logged for pattern analysis but no immediate action
        pass
