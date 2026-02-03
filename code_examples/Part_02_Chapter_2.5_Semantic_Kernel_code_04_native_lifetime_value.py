@kernel_function(
    description="Calculate customer lifetime value based on order history and subscription data",
    name="calculate_lifetime_value"
)
def calculate_lifetime_value(self, customer_id: str) -> float:
    """Native function: Pure calculation logic"""
    orders = self.db.query(
        "SELECT SUM(total) as order_value FROM orders WHERE customer_id = ?",
        [customer_id]
    )
    subscriptions = self.db.query(
        "SELECT monthly_value, months_active FROM subscriptions WHERE customer_id = ?",
        [customer_id]
    )

    order_value = orders[0]['order_value'] if orders else 0
    subscription_value = sum(
        sub['monthly_value'] * sub['months_active']
        for sub in subscriptions
    )

    return order_value + subscription_value
