from datetime import datetime, timedelta

class TemporalPermission:
    """Time-bounded permissions that automatically expire"""

    def __init__(self, permission: str, duration_hours: int = 1):
        self.permission = permission
        self.granted_at = datetime.utcnow()
        self.expires_at = self.granted_at + timedelta(hours=duration_hours)

    def is_valid(self) -> bool:
        """Check if permission is still valid"""
        return datetime.utcnow() < self.expires_at

# Usage: Grant 24-hour access for quarterly audit
audit_permission = TemporalPermission('read:financial_records', duration_hours=24)