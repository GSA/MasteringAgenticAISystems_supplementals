class JITAccessManager:
    """Just-in-time access provisioning with automatic revocation"""

    def get_credentials(self, agent_id: str, task: str) -> Credentials:
        """Request fresh credentials for specific task"""
        # Analyze task to determine required permissions
        required_permissions = self._analyze_task(task)

        # Request temporary credentials from identity provider
        credentials = self.identity_provider.issue_token(
            principal=agent_id,
            scopes=required_permissions,
            duration_minutes=15  # Short-lived by design
        )

        # Schedule automatic revocation
        self._schedule_revocation(credentials, after_minutes=15)

        return credentials

    def revoke_credentials(self, credentials: Credentials):
        """Immediately revoke credentials"""
        self.identity_provider.revoke_token(credentials.token)

# Usage: Credentials exist only during task execution
access_manager = JITAccessManager()

# Agent invoked for specific task
credentials = access_manager.get_credentials('agent_123', 'send_email')
# Use credentials for the task...
# Credentials automatically expire after 15 minutes