# Load balancer example implementing weighted least-connections.
from collections import deque
from typing import Deque, Dict, List


class WeightedLeastConnectionsBalancer:
    def __init__(self, replicas: List[Dict]):
        """Initialize the balancer.

        replicas: [
            {"endpoint": "10.0.1.10", "weight": 3, "type": "gpu"},
            {"endpoint": "10.0.2.10", "weight": 1, "type": "cpu"},
            ...
        ]
        """
        if not replicas:
            raise ValueError("At least one replica is required")
        if any(replica.get("weight", 0) <= 0 for replica in replicas):
            raise ValueError("Replica weights must be positive")

        self.replicas = replicas
        self.active_connections = {replica["endpoint"]: 0 for replica in replicas}

    def select_replica(self) -> str:
        """Select the replica with the lowest weighted connection count."""
        weighted_loads = [
            (
                self.active_connections[replica["endpoint"]] / replica["weight"],
                replica["endpoint"],
            )
            for replica in self.replicas
        ]
        _, selected = min(weighted_loads)
        return selected

    def track_request_start(self, endpoint: str) -> None:
        """Increment the connection counter when routing a request."""
        if endpoint not in self.active_connections:
            raise KeyError(f"Unknown replica endpoint: {endpoint}")
        self.active_connections[endpoint] += 1

    def track_request_end(self, endpoint: str) -> None:
        """Decrement the connection counter when a request completes."""
        if endpoint not in self.active_connections:
            raise KeyError(f"Unknown replica endpoint: {endpoint}")
        self.active_connections[endpoint] = max(0, self.active_connections[endpoint] - 1)


# Example usage
balancer = WeightedLeastConnectionsBalancer([
    {"endpoint": "10.0.1.10", "weight": 3, "type": "gpu"},
    {"endpoint": "10.0.1.11", "weight": 3, "type": "gpu"},
    {"endpoint": "10.0.2.10", "weight": 1, "type": "cpu"},
    {"endpoint": "10.0.2.11", "weight": 1, "type": "cpu"},
    {"endpoint": "10.0.2.12", "weight": 1, "type": "cpu"},
])

# A FIFO queue is sufficient for this deterministic simulation. A real service
# would call track_request_end() from the completion callback for that request.
in_flight: Deque[str] = deque()
for request_number in range(1, 16):
    replica = balancer.select_replica()
    print(f"Request {request_number} -> {replica}")
    balancer.track_request_start(replica)
    in_flight.append(replica)

    if request_number % 3 == 1 and len(in_flight) > 1:
        completed_replica = in_flight.popleft()
        balancer.track_request_end(completed_replica)
