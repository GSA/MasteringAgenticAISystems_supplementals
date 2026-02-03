from semantic_kernel.functions import kernel_function

class CustomerServicePlugin:
    """Plugin providing customer data access and operations"""

    def __init__(self, db_client: DatabaseClient, crm_client: CRMClient):
        """Dependencies injected by kernel"""
        self.db = db_client
        self.crm = crm_client

    @kernel_function(
        description="Retrieve customer profile including contact info, preferences, and history",
        name="get_customer_profile"
    )
    def get_customer_profile(self, customer_id: str) -> dict:
        """Get comprehensive customer data from CRM"""
        profile = self.crm.get_customer(customer_id)
        order_history = self.db.query(
            "SELECT * FROM orders WHERE customer_id = ?",
            [customer_id]
        )
        return {
            "profile": profile,
            "order_history": order_history,
            "total_orders": len(order_history)
        }

    @kernel_function(
        description="Update customer contact information (email, phone, address)",
        name="update_customer_contact"
    )
    def update_customer_contact(
        self,
        customer_id: str,
        email: str = None,
        phone: str = None,
        address: str = None
    ) -> dict:
        """Update customer contact details in CRM"""
        return self.crm.update_customer(
            customer_id,
            email=email,
            phone=phone,
            address=address
        )

# Register plugin with kernel
kernel.add_plugin(
    CustomerServicePlugin(
        db_client=kernel.get_service(DatabaseClient),
        crm_client=kernel.get_service(CRMClient)
    ),
    plugin_name="customer_service"
)
