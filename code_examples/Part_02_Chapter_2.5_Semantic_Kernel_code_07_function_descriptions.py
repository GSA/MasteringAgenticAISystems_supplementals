# Poor function description: Ambiguous purpose
@kernel_function(description="Get data")  # Which data? When should this be called?
def get_data(self, id: str) -> dict:
    pass

# Good function description: Clear purpose and appropriate usage
@kernel_function(
    description="Retrieve complete customer profile including name, contact information, "
                "account standing, and purchase history. Use when user asks about customer "
                "details, account status, or order history for a specific customer ID."
)
def get_customer_profile(self, customer_id: str) -> dict:
    pass
