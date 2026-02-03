# Burn Rate Alerting Configuration
burn_rate_alerts = {
    "critical": {
        "threshold": 10.0,              # 10x burn rate = 3 days to budget exhaustion
        "window": "5 minutes",          # Sustained over 5 minutes (not transient spike)
        "action": "Page on-call engineer immediately",
        "escalation": "Escalate to VP Engineering if not ack'd in 15 min"
    },
    "warning": {
        "threshold": 2.0,               # 2x burn rate = 15 days to budget exhaustion
        "window": "1 hour",             # Sustained over 1 hour
        "action": "Create P1 ticket for investigation",
        "sla": "Response within 24 hours"
    },
    "info": {
        "threshold": 1.5,               # 1.5x burn rate = 20 days to budget exhaustion
        "window": "6 hours",            # Sustained over 6 hours
        "action": "Log for trend analysis",
        "review": "Weekly metrics review"
    }
}
