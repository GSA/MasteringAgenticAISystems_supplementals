    @kernel_function(
        description="Retrieve comprehensive customer profile including contact information, "
                    "account standing, subscription status, and metadata. Use when you need "
                    "complete customer details to answer questions or update records.",
        name="get_customer_profile"
    )
    async def get_customer_profile(self, customer_id: str) -> dict:
        """
        Native function: Retrieve customer profile from CRM API.

        Returns:
            {
                "customer_id": str,
                "name": str,
                "email": str,
                "phone": str,
                "account_standing": str,  # "good", "warning", "suspended"
                "subscription_tier": str,  # "free", "pro", "enterprise"
                "total_lifetime_value": float,
                "support_tickets_count": int,
                "last_contact_date": str
            }
        """
        response = await self.http_client.get(
            f"{self.crm_base_url}/customers/{customer_id}"
        )
        response.raise_for_status()
        return response.json()

    @kernel_function(
        description="Update customer contact information (email, phone, or address). "
                    "Use when customer requests contact detail changes or you need to "
                    "correct outdated information. Returns updated profile.",
        name="update_customer_contact"
    )
    async def update_customer_contact(
        self,
        customer_id: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        address: Optional[str] = None
    ) -> dict:
        """
        Native function: Update customer contact details in CRM.

        Returns: Updated customer profile with new contact information
        """
        update_data = {}
        if email:
            update_data["email"] = email
        if phone:
            update_data["phone"] = phone
        if address:
            update_data["address"] = address

        response = await self.http_client.patch(
            f"{self.crm_base_url}/customers/{customer_id}",
            json=update_data
        )
        response.raise_for_status()
        return response.json()
