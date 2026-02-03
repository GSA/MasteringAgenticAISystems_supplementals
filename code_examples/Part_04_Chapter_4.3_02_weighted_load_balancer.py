# Load balancer configuration implementing least-connections with weights
from typing import List, Dict
import heapq

class WeightedLeastConnectionsBalancer:
    def __init__(self, replicas: List[Dict]):
        """
        replicas: [
            {"endpoint": "10.0.1.10", "weight": 3, "type": "gpu"},
            {"endpoint": "10.0.2.10", "weight": 1, "type": "cpu"},
            ...
        ]
        """
        self.replicas = replicas
        self.active_connections = {r["endpoint"]: 0 for r in replicas}

    def select_replica(self) -> str:
        """Select replica with lowest weighted connection count"""
        # Calculate effective load = connections / weight
        # Lower weight means connection counts more heavily (CPU instances)
        # Higher weight means connection counts less heavily (GPU instances)
        weighted_loads = [
            (self.active_connections[r["endpoint"]] / r["weight"], r["endpoint"])
            for r in self.replicas
        ]

        # Select replica with minimum weighted load
        min_load, selected = min(weighted_loads)

        return selected

    def track_request_start(self, endpoint: str):
        """Increment connection counter when routing request"""
        self.active_connections[endpoint] += 1

    def track_request_end(self, endpoint: str):
        """Decrement connection counter when request completes"""
        self.active_connections[endpoint] = max(
            0, self.active_connections[endpoint] - 1
        )

# Example usage
balancer = WeightedLeastConnectionsBalancer([
    {"endpoint": "10.0.1.10", "weight": 3, "type": "gpu"},
    {"endpoint": "10.0.1.11", "weight": 3, "type": "gpu"},
    {"endpoint": "10.0.2.10", "weight": 1, "type": "cpu"},
    {"endpoint": "10.0.2.11", "weight": 1, "type": "cpu"},
    {"endpoint": "10.0.2.12", "weight": 1, "type": "cpu"},
])

# Simulate request distribution
for i in range(15):
    replica = balancer.select_replica()
    print(f"Request {i+1} → {replica}")
    balancer.track_request_start(replica)

    # Simulate some requests completing
    if i > 0 and i % 3 == 0:
        completed_replica = # ... track which request completed
        balancer.track_request_end(completed_replica)
