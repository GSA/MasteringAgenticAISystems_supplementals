#!/usr/bin/env python3
"""
NVIDIA NIM Load Balancer
Intelligent routing across multiple NIM instances with health checking and failover
Version: 1.0
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"


@dataclass
class NIMInstance:
    """Represents a single NIM instance."""
    url: str
    model: str
    weight: float = 1.0
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    avg_latency: float = 0.0
    last_health_check: float = 0.0


class NIMLoadBalancer:
    """
    Intelligent load balancer for NVIDIA NIM instances.

    Features:
    - Multiple routing strategies
    - Health checking with automatic failover
    - Connection tracking
    - Latency-based routing
    - Task-specific model selection
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LEAST_CONNECTIONS,
        health_check_interval: int = 30
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval

        # Model instances mapping
        self.instances: Dict[str, List[NIMInstance]] = defaultdict(list)

        # Round-robin counters
        self.rr_counters: Dict[str, int] = defaultdict(int)

        # Performance tracking
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)

        # Health check task
        self.health_check_task: Optional[asyncio.Task] = None

    def add_instance(self, model: str, url: str, weight: float = 1.0):
        """Add a NIM instance to the pool."""
        instance = NIMInstance(url=url, model=model, weight=weight)
        self.instances[model].append(instance)
        logger.info(f"Added NIM instance: {model} at {url} (weight: {weight})")

    def remove_instance(self, model: str, url: str):
        """Remove a NIM instance from the pool."""
        self.instances[model] = [
            inst for inst in self.instances[model] if inst.url != url
        ]
        logger.info(f"Removed NIM instance: {model} at {url}")

    async def start(self):
        """Start the load balancer and health checks."""
        logger.info("Starting NIM Load Balancer...")
        self.health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """Stop the load balancer."""
        logger.info("Stopping NIM Load Balancer...")
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        """Continuously check health of all instances."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_instances(self):
        """Check health of all NIM instances."""
        tasks = []
        for model, instances in self.instances.items():
            for instance in instances:
                tasks.append(self._check_instance_health(instance))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_instance_health(self, instance: NIMInstance):
        """Check health of a single instance."""
        health_url = f"{instance.url}/v1/health"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        was_unhealthy = not instance.healthy
                        instance.healthy = data.get("status") == "healthy"
                        instance.last_health_check = time.time()

                        if was_unhealthy and instance.healthy:
                            logger.info(f"Instance {instance.url} recovered")
                    else:
                        instance.healthy = False
                        logger.warning(f"Instance {instance.url} unhealthy: HTTP {response.status}")
        except Exception as e:
            instance.healthy = False
            logger.error(f"Health check failed for {instance.url}: {e}")

    def _select_model_by_task(self, prompt: str) -> str:
        """Select appropriate model based on task type."""
        prompt_lower = prompt.lower()

        # Code generation
        if any(keyword in prompt_lower for keyword in ["code", "function", "program", "script"]):
            return "mistral-7b" if "mistral-7b" in self.instances else "llama2-7b"

        # Embedding generation
        if any(keyword in prompt_lower for keyword in ["embed", "vector", "similarity"]):
            return "embed-qa"

        # Default to general purpose
        return "llama2-7b"

    def _select_instance(self, model: str) -> Optional[NIMInstance]:
        """Select best instance using configured strategy."""
        instances = [inst for inst in self.instances.get(model, []) if inst.healthy]

        if not instances:
            logger.error(f"No healthy instances available for model {model}")
            return None

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self.rr_counters[model] % len(instances)
            self.rr_counters[model] += 1
            return instances[idx]

        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.active_connections)

        elif self.strategy == RoutingStrategy.WEIGHTED:
            # Weighted random selection
            import random
            total_weight = sum(inst.weight for inst in instances)
            r = random.uniform(0, total_weight)

            cumulative = 0
            for inst in instances:
                cumulative += inst.weight
                if r <= cumulative:
                    return inst
            return instances[-1]

        elif self.strategy == RoutingStrategy.LATENCY_BASED:
            # Select instance with lowest average latency
            return min(instances, key=lambda x: x.avg_latency or float('inf'))

        return instances[0]

    async def forward_request(
        self,
        model: str,
        messages: List[Dict],
        **kwargs
    ) -> Dict:
        """
        Forward inference request to appropriate NIM instance.

        Args:
            model: Target model name
            messages: Chat messages
            **kwargs: Additional OpenAI API parameters

        Returns:
            Response from NIM instance
        """
        # Select instance
        instance = self._select_instance(model)
        if not instance:
            raise RuntimeError(f"No available instances for model {model}")

        # Track connection
        instance.active_connections += 1
        start_time = time.time()

        try:
            # Prepare request
            payload = {
                "model": model,
                "messages": messages,
                **kwargs
            }

            # Forward to NIM
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{instance.url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            # Update metrics
            latency = time.time() - start_time
            instance.total_requests += 1
            instance.avg_latency = (
                (instance.avg_latency * (instance.total_requests - 1) + latency)
                / instance.total_requests
            )
            self.request_counts[model] += 1

            logger.info(
                f"Request to {instance.url} completed in {latency:.2f}s "
                f"(avg: {instance.avg_latency:.2f}s)"
            )

            return result

        except Exception as e:
            self.error_counts[model] += 1
            logger.error(f"Request to {instance.url} failed: {e}")

            # Mark instance as unhealthy after repeated failures
            if self.error_counts[model] > 3:
                instance.healthy = False
                logger.warning(f"Marking instance {instance.url} as unhealthy")

            raise

        finally:
            instance.active_connections -= 1

    async def route_request(
        self,
        prompt: str,
        auto_select_model: bool = True,
        **kwargs
    ) -> Dict:
        """
        Route request with automatic model selection.

        Args:
            prompt: User prompt
            auto_select_model: Automatically select model based on task
            **kwargs: Additional API parameters

        Returns:
            Response from NIM instance
        """
        # Select model
        if auto_select_model:
            model = self._select_model_by_task(prompt)
        else:
            model = kwargs.pop("model", "llama2-7b")

        # Format messages
        messages = [{"role": "user", "content": prompt}]

        # Forward request
        return await self.forward_request(model, messages, **kwargs)

    def get_stats(self) -> Dict:
        """Get load balancer statistics."""
        stats = {
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "by_model": {},
            "instances": {}
        }

        for model, count in self.request_counts.items():
            error_rate = self.error_counts[model] / count if count > 0 else 0
            stats["by_model"][model] = {
                "requests": count,
                "errors": self.error_counts[model],
                "error_rate": error_rate
            }

        for model, instances in self.instances.items():
            stats["instances"][model] = [
                {
                    "url": inst.url,
                    "healthy": inst.healthy,
                    "active_connections": inst.active_connections,
                    "total_requests": inst.total_requests,
                    "avg_latency": round(inst.avg_latency, 3)
                }
                for inst in instances
            ]

        return stats


# Example usage
async def main():
    """Example usage of NIM Load Balancer."""

    # Initialize load balancer
    lb = NIMLoadBalancer(strategy=RoutingStrategy.LEAST_CONNECTIONS)

    # Add NIM instances
    lb.add_instance("llama2-7b", "http://nim-llama2-1:8000", weight=1.0)
    lb.add_instance("llama2-7b", "http://nim-llama2-2:8000", weight=1.0)
    lb.add_instance("mistral-7b", "http://nim-mistral-1:8000", weight=1.0)
    lb.add_instance("embed-qa", "http://nim-embed-1:8000", weight=1.0)

    # Start load balancer
    await lb.start()

    try:
        # Example requests
        prompts = [
            "Explain quantum computing",
            "Write a Python function to calculate Fibonacci",
            "What are the benefits of renewable energy?",
        ]

        for prompt in prompts:
            response = await lb.route_request(
                prompt=prompt,
                auto_select_model=True,
                max_tokens=100
            )
            print(f"Prompt: {prompt}")
            print(f"Response: {response['choices'][0]['message']['content'][:100]}...")
            print()

        # Print statistics
        stats = lb.get_stats()
        print(f"Load Balancer Statistics:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Total Errors: {stats['total_errors']}")
        print(f"\nBy Model:")
        for model, data in stats['by_model'].items():
            print(f"  {model}: {data['requests']} requests, {data['error_rate']:.1%} error rate")

    finally:
        await lb.stop()


if __name__ == "__main__":
    asyncio.run(main())
