class APIGateway:
    """Infrastructure-level policy enforcement for all agent requests"""

    def __init__(self):
        self.policies = PolicyEngine()
        self.rate_limiter = RateLimiter()
        self.audit_log = AuditLogger()

    def enforce(self, request: APIRequest) -> APIResponse:
        """Enforce all policies before allowing request to proceed"""
        # Step 1: Check authorization
        if not self.policies.is_authorized(request):
            self.audit_log.log_violation(request, "Unauthorized")
            return APIResponse(status=403, body="Forbidden")

        # Step 2: Check rate limits
        if self.rate_limiter.is_exceeded(request.user_id):
            self.audit_log.log_violation(request, "Rate limit exceeded")
            return APIResponse(status=429, body="Too many requests")

        # Step 3: Validate parameters
        if not self.policies.validate_parameters(request):
            self.audit_log.log_violation(request, "Invalid parameters")
            return APIResponse(status=400, body="Bad request")

        # Step 4: Execute request if all checks pass
        response = self.execute_request(request)

        # Step 5: Log successful execution
        self.audit_log.log_success(request, response)

        return response