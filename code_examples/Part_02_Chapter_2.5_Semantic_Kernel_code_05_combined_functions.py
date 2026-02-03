class CustomerServicePlugin:
    """Combines semantic and native functions for comprehensive customer service"""

    @kernel_function(description="Get customer order history from database")
    def get_order_history(self, customer_id: str) -> list:
        """Native: Reliable data retrieval"""
        return self.db.query(
            "SELECT * FROM orders WHERE customer_id = ? ORDER BY date DESC",
            [customer_id]
        )

    @kernel_function(description="Get current account status including balance and standing")
    def get_account_status(self, customer_id: str) -> dict:
        """Native: Deterministic status check"""
        return {
            "balance": self.crm.get_balance(customer_id),
            "standing": self.crm.get_account_standing(customer_id),
            "payment_method": self.crm.get_payment_method(customer_id)
        }

    @kernel_function(description="Generate personalized response to customer inquiry using profile and context")
    async def generate_customer_response(
        self,
        customer_inquiry: str,
        customer_profile: dict,
        order_history: list,
        account_status: dict
    ) -> str:
        """Semantic: Natural language generation with context"""
        prompt = f"""
You are a helpful customer service representative. Generate a personalized response to this customer.

Customer Inquiry: {customer_inquiry}

Customer Context:
- Name: {customer_profile.get('name')}
- Account Standing: {account_status.get('standing')}
- Total Orders: {len(order_history)}
- Recent Orders: {order_history[:3]}
- Current Balance: ${account_status.get('balance')}

Provide a helpful, empathetic response addressing their inquiry with specific details from their account.
"""
        response = await kernel.invoke_prompt(prompt)
        return response
