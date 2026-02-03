# Intake Agent processes incoming ticket
class IntakeAgent:
    def process_ticket(self, customer_message: str, metadata: dict) -> TicketContext:
        """Validates and enriches incoming customer ticket."""
        # Validate message content
        if not self._is_valid_message(customer_message):
            raise InvalidTicketError("Message validation failed")

        # Enrich with account context (via MCP to CRM)
        account_data = self.mcp_client.query_resource(
            "crm://accounts/{account_id}".format(**metadata)
        )

        # Create ticket context for downstream agents
        return TicketContext(
            ticket_id=self._generate_id(),
            message=customer_message,
            account=account_data,
            timestamp=datetime.utcnow(),
            conversation_history=self._load_history(metadata['account_id'])
        )