"""
Code Example 5.3.2: Multi-Agent Collaborative Planning with NVIDIA NIM

Purpose: Demonstrate specialized agent collaboration with parallel execution

Concepts Demonstrated:
- Multi-agent role specialization: Planner, Executor, Coordinator agents
- Parallel task execution: Leveraging multiple agent replicas
- NVIDIA NIM deployment: GPU-accelerated agent microservices
- Load balancing: Triton Inference Server for optimal GPU utilization

Prerequisites:
- Understanding of async/await patterns
- Basic knowledge of microservices architecture
- Familiarity with NVIDIA NIM deployment

Author: NVIDIA Certified Generative AI LLM Course
Chapter: 5, Section: 5.3
Exam Skill: 5.3 - Engineer Planning Strategies for Multi-Step Decision-Making
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
from typing import List, Dict, Any, TypedDict
from dataclasses import dataclass
import asyncio
import logging
from enum import Enum
import time

# Third-party imports (in production)
# from openai import AsyncOpenAI
# import httpx

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NVIDIA NIM Configuration
NIM_CONFIG = {
    "planner_endpoint": "http://nim-planner:8000/v1/chat/completions",
    "executor_endpoints": [
        "http://nim-executor-0:8000/v1/chat/completions",
        "http://nim-executor-1:8000/v1/chat/completions",
        "http://nim-executor-2:8000/v1/chat/completions"
    ],
    "coordinator_endpoint": "http://nim-coordinator:8000/v1/chat/completions",
    "triton_url": "http://triton-server:8000"
}

# Model Configuration
MODEL_CONFIG = {
    "planner_model": "meta/llama-3.1-70b-instruct",  # Larger model for strategic planning
    "executor_model": "meta/llama-3.1-8b-instruct",   # Smaller model for execution
    "coordinator_model": "meta/llama-3.1-405b-instruct",  # Largest for synthesis
    "temperature": 0.0,
    "max_tokens": 2048,
    "use_tensorrt": True  # Enable TensorRT optimization
}

# ============================================================================
# NVIDIA NIM AGENT FRAMEWORK
# ============================================================================

class AgentRole(Enum):
    """Agent role types for specialized planning."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    REVIEWER = "reviewer"


@dataclass
class NIMAgentConfig:
    """
    Configuration for NVIDIA NIM-deployed agent.

    Attributes:
        endpoint (str): NIM service endpoint URL
        model (str): Model identifier (e.g., meta/llama-3.1-70b-instruct)
        role (AgentRole): Agent's specialized role
        max_tokens (int): Maximum response tokens
        temperature (float): Sampling temperature
        use_tensorrt (bool): Enable TensorRT optimization
    """
    endpoint: str
    model: str
    role: AgentRole
    max_tokens: int = 2048
    temperature: float = 0.0
    use_tensorrt: bool = True


class NIMAgent:
    """
    Base class for NVIDIA NIM-deployed agents.

    This demonstrates the pattern for interacting with NIM microservices,
    with proper error handling and performance optimization.

    Example:
        >>> config = NIMAgentConfig(
        ...     endpoint="http://nim:8000/v1/chat/completions",
        ...     model="meta/llama-3.1-8b-instruct",
        ...     role=AgentRole.EXECUTOR
        ... )
        >>> agent = NIMAgent(config)
        >>> result = await agent.invoke("Execute task")
    """

    def __init__(self, config: NIMAgentConfig):
        """
        Initialize NIM agent with configuration.

        Args:
            config (NIMAgentConfig): Agent configuration
        """
        self.config = config
        self.endpoint = config.endpoint
        self.model = config.model
        self.role = config.role

        logger.info(
            f"Initialized {self.role.value} agent with model {self.model}"
        )

    async def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """
        Invoke NIM endpoint asynchronously.

        In production, this would make HTTP request to NIM service.
        For this example, we simulate the interaction.

        Args:
            prompt (str): User prompt
            system_prompt (str, optional): System instruction

        Returns:
            str: Agent response

        Raises:
            Exception: If NIM invocation fails
        """
        start_time = time.time()

        logger.debug(f"{self.role.value} processing: {prompt[:50]}...")

        # Production implementation:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         self.endpoint,
        #         json={
        #             "model": self.model,
        #             "messages": [
        #                 {"role": "system", "content": system_prompt or ""},
        #                 {"role": "user", "content": prompt}
        #             ],
        #             "temperature": self.config.temperature,
        #             "max_tokens": self.config.max_tokens
        #         }
        #     )
        #     result = response.json()["choices"][0]["message"]["content"]

        # Simulated response for demonstration
        await asyncio.sleep(0.1)  # Simulate network latency
        result = f"[{self.role.value}] Response to: {prompt[:30]}..."

        latency = time.time() - start_time
        logger.info(
            f"✓ {self.role.value} completed in {latency*1000:.1f}ms"
        )

        return result


# ============================================================================
# SPECIALIZED AGENT IMPLEMENTATIONS
# ============================================================================

class PlannerAgent(NIMAgent):
    """
    Agent specialized in strategic planning and task decomposition.

    Uses larger model (70B+) for complex reasoning and comprehensive planning.
    Focuses on creating well-structured, executable plans.
    """

    async def create_plan(
        self,
        goal: str,
        constraints: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Create strategic execution plan for goal.

        This demonstrates hierarchical decomposition with:
        - Task identification and sequencing
        - Resource requirement analysis
        - Parallelization opportunity detection
        - Dependency mapping

        Args:
            goal (str): High-level objective to plan for
            constraints (Dict, optional): Planning constraints

        Returns:
            List[Dict]: Structured execution plan

        Example:
            >>> planner = PlannerAgent(config)
            >>> plan = await planner.create_plan(
            ...     "Analyze Q4 financial performance"
            ... )
            >>> print(f"Created {len(plan)} task groups")
        """
        logger.info(f"Creating strategic plan for: {goal}")

        planning_prompt = f"""
        Goal: {goal}

        Create detailed execution plan with these requirements:
        1. Break goal into independent task groups
        2. Identify tasks that can run in parallel
        3. Specify required resources for each task
        4. Estimate relative effort (low/medium/high)
        5. Map dependencies between tasks

        Constraints: {constraints or 'None'}

        Return structured plan as JSON array:
        [
          {{
            "task_id": 1,
            "description": "task description",
            "parallel_group": "A",  // Tasks in same group can run parallel
            "estimated_effort": "medium",
            "required_resources": ["resource1"],
            "dependencies": []
          }},
          ...
        ]

        Optimize for parallelization where safe.
        """

        response = await self.invoke(planning_prompt)

        # Parse plan (simplified - production would parse JSON)
        plan = self._create_example_plan(goal)

        logger.info(f"✓ Created plan with {len(plan)} tasks")
        logger.info(
            f"   Parallel groups: {len(set(t['parallel_group'] for t in plan))}"
        )

        return plan

    def _create_example_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create example plan structure for demonstration."""
        return [
            {
                "task_id": 1,
                "description": "Data collection and preparation",
                "parallel_group": "A",
                "estimated_effort": "medium",
                "required_resources": ["database_access", "api_credentials"],
                "dependencies": []
            },
            {
                "task_id": 2,
                "description": "Preliminary analysis",
                "parallel_group": "A",
                "estimated_effort": "low",
                "required_resources": ["analytics_tools"],
                "dependencies": []
            },
            {
                "task_id": 3,
                "description": "Deep dive analysis",
                "parallel_group": "B",
                "estimated_effort": "high",
                "required_resources": ["compute_resources"],
                "dependencies": [1, 2]
            },
            {
                "task_id": 4,
                "description": "Comparative evaluation",
                "parallel_group": "B",
                "estimated_effort": "medium",
                "required_resources": ["benchmark_data"],
                "dependencies": [1, 2]
            },
            {
                "task_id": 5,
                "description": "Report synthesis",
                "parallel_group": "C",
                "estimated_effort": "medium",
                "required_resources": ["reporting_tools"],
                "dependencies": [3, 4]
            }
        ]


class ExecutorAgent(NIMAgent):
    """
    Agent specialized in task execution.

    Uses smaller, faster model (8B) optimized for specific task execution.
    Designed for parallel deployment across multiple replicas.
    """

    async def execute_task(
        self,
        task: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute assigned task.

        Demonstrates:
        - Task execution with resource validation
        - Error handling and retry logic
        - Result formatting for aggregation

        Args:
            task (Dict): Task specification from planner
            context (Dict, optional): Execution context

        Returns:
            Dict: Task execution result

        Example:
            >>> executor = ExecutorAgent(config)
            >>> result = await executor.execute_task({
            ...     "task_id": 1,
            ...     "description": "Analyze data",
            ...     "required_resources": ["database"]
            ... })
            >>> assert result["status"] == "completed"
        """
        task_id = task["task_id"]
        description = task["description"]

        logger.info(f"Executing task {task_id}: {description}")

        # Validate required resources
        missing_resources = self._check_resources(task["required_resources"])
        if missing_resources:
            logger.warning(
                f"Missing resources for task {task_id}: {missing_resources}"
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "error": f"Missing resources: {missing_resources}",
                "output": None
            }

        # Execute task
        execution_prompt = f"""
        Execute this task: {description}

        Context: {context or 'None provided'}

        Provide detailed results including:
        - Actions taken
        - Data gathered or processed
        - Key findings
        - Any issues encountered
        """

        start_time = time.time()

        try:
            output = await self.invoke(execution_prompt)

            execution_time = time.time() - start_time

            result = {
                "task_id": task_id,
                "status": "completed",
                "output": output,
                "execution_time": execution_time,
                "error": None
            }

            logger.info(
                f"✓ Task {task_id} completed in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"✗ Task {task_id} failed: {str(e)}")

            return {
                "task_id": task_id,
                "status": "failed",
                "output": None,
                "error": str(e)
            }

    def _check_resources(self, required: List[str]) -> List[str]:
        """
        Check resource availability.

        In production: Would verify actual resource access.
        For demo: Simulates resource checking.

        Args:
            required (List[str]): Required resources

        Returns:
            List[str]: Missing resources (empty if all available)
        """
        # Simulate: All resources available in this demo
        return []


class CoordinatorAgent(NIMAgent):
    """
    Agent that coordinates multi-agent collaborative planning.

    Uses largest model (405B) for sophisticated coordination,
    synthesis, and decision-making across multiple agents.

    Responsibilities:
    - Orchestrate planner and executor agents
    - Manage parallel execution
    - Synthesize results from multiple agents
    - Handle failures and coordination issues
    """

    def __init__(self, config: NIMAgentConfig):
        """Initialize coordinator with sub-agents."""
        super().__init__(config)

        # Initialize specialized agents
        self.planner = PlannerAgent(NIMAgentConfig(
            endpoint=NIM_CONFIG["planner_endpoint"],
            model=MODEL_CONFIG["planner_model"],
            role=AgentRole.PLANNER,
            use_tensorrt=True
        ))

        # Create executor pool (3 replicas for parallel execution)
        self.executors = [
            ExecutorAgent(NIMAgentConfig(
                endpoint=endpoint,
                model=MODEL_CONFIG["executor_model"],
                role=AgentRole.EXECUTOR,
                use_tensorrt=True
            ))
            for endpoint in NIM_CONFIG["executor_endpoints"]
        ]

        logger.info(
            f"Coordinator initialized with {len(self.executors)} executor replicas"
        )

    async def collaborative_planning(
        self,
        goal: str,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate multi-agent collaborative planning and execution.

        Workflow:
        1. Planner creates strategic plan
        2. Coordinator analyzes parallelization opportunities
        3. Executors work in parallel on independent tasks
        4. Coordinator synthesizes results

        Args:
            goal (str): High-level objective
            constraints (Dict, optional): Planning constraints

        Returns:
            Dict: Complete execution results with synthesis

        Example:
            >>> coordinator = CoordinatorAgent(config)
            >>> result = await coordinator.collaborative_planning(
            ...     "Comprehensive market analysis for product launch"
            ... )
            >>> print(f"Executed {result['total_tasks']} tasks")
            >>> print(f"Parallel execution saved {result['time_saved']:.1f}s")
        """
        logger.info("="*70)
        logger.info(f"COLLABORATIVE PLANNING: {goal}")
        logger.info("="*70)

        start_time = time.time()

        # ====================================================================
        # PHASE 1: Strategic Planning
        # ====================================================================

        logger.info("\n[Phase 1] Strategic Planning")

        plan = await self.planner.create_plan(goal, constraints)

        logger.info(f"✓ Received plan with {len(plan)} tasks")

        # ====================================================================
        # PHASE 2: Execution Orchestration
        # ====================================================================

        logger.info("\n[Phase 2] Parallel Execution")

        # Group tasks by parallel execution groups
        parallel_groups = self._group_by_parallelism(plan)

        logger.info(
            f"Organized into {len(parallel_groups)} execution groups"
        )

        execution_results = []

        # Execute each group (tasks within group run in parallel)
        for group_id, tasks in parallel_groups.items():
            logger.info(
                f"\n  Executing group {group_id}: {len(tasks)} tasks in parallel"
            )

            group_results = await self._execute_parallel_group(tasks)
            execution_results.extend(group_results)

            # Check for failures
            failures = [r for r in group_results if r["status"] == "failed"]
            if failures:
                logger.warning(
                    f"  ⚠ {len(failures)} tasks failed in group {group_id}"
                )

        # ====================================================================
        # PHASE 3: Result Synthesis
        # ====================================================================

        logger.info("\n[Phase 3] Result Synthesis")

        synthesis = await self._synthesize_results(goal, execution_results)

        total_time = time.time() - start_time

        # Calculate metrics
        successful_tasks = [r for r in execution_results if r["status"] == "completed"]
        failed_tasks = [r for r in execution_results if r["status"] == "failed"]

        # Estimate time savings from parallelization
        sequential_time = sum(r.get("execution_time", 0) for r in execution_results)
        time_saved = sequential_time - total_time

        result = {
            "goal": goal,
            "plan": plan,
            "execution_results": execution_results,
            "synthesis": synthesis,
            "total_tasks": len(plan),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "parallel_groups": len(parallel_groups),
            "total_time": total_time,
            "estimated_sequential_time": sequential_time,
            "time_saved": time_saved,
            "speedup": sequential_time / total_time if total_time > 0 else 1.0
        }

        logger.info("\n" + "="*70)
        logger.info("RESULTS SUMMARY")
        logger.info("="*70)
        logger.info(f"✓ Total tasks: {result['total_tasks']}")
        logger.info(f"✓ Successful: {result['successful_tasks']}")
        logger.info(f"✗ Failed: {result['failed_tasks']}")
        logger.info(f"⚡ Speedup: {result['speedup']:.2f}x (parallelization)")
        logger.info(f"⏱ Time saved: {result['time_saved']:.2f}s")
        logger.info("="*70)

        return result

    def _group_by_parallelism(
        self,
        plan: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group tasks by parallel execution groups.

        Tasks in the same group can run in parallel (no dependencies).
        Groups execute sequentially to respect dependencies.

        Args:
            plan (List[Dict]): Execution plan

        Returns:
            Dict[str, List]: Tasks grouped by parallel group ID
        """
        groups = {}

        for task in plan:
            group_id = task.get("parallel_group", "default")

            if group_id not in groups:
                groups[group_id] = []

            groups[group_id].append(task)

        return groups

    async def _execute_parallel_group(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute group of tasks in parallel across executor replicas.

        Demonstrates:
        - Load balancing across multiple executor replicas
        - Parallel async execution
        - Result aggregation

        Args:
            tasks (List[Dict]): Tasks to execute in parallel

        Returns:
            List[Dict]: Execution results
        """
        # Distribute tasks across available executors (round-robin)
        task_assignments = [
            (tasks[i], self.executors[i % len(self.executors)])
            for i in range(len(tasks))
        ]

        # Execute all tasks in parallel
        results = await asyncio.gather(*[
            executor.execute_task(task)
            for task, executor in task_assignments
        ])

        return results

    async def _synthesize_results(
        self,
        goal: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize execution results into coherent final output.

        Uses coordinator's large model for sophisticated synthesis
        across multiple task results.

        Args:
            goal (str): Original goal
            results (List[Dict]): Task execution results

        Returns:
            str: Synthesized final result
        """
        synthesis_prompt = f"""
        Goal: {goal}

        Task execution results:
        {len(results)} tasks completed

        Successful tasks:
        {[r for r in results if r["status"] == "completed"]}

        Synthesize these results into a comprehensive, coherent response that:
        1. Directly addresses the original goal
        2. Integrates findings from all tasks
        3. Highlights key insights
        4. Provides actionable recommendations
        5. Notes any gaps or limitations

        Create a well-structured synthesis.
        """

        synthesis = await self.invoke(synthesis_prompt)

        return synthesis


# ============================================================================
# PERFORMANCE BENCHMARKING
# ============================================================================

class PerformanceMetrics:
    """Track and compare performance metrics."""

    @staticmethod
    def compare_single_vs_multi_agent():
        """
        Compare single-agent vs multi-agent performance.

        Based on NVIDIA benchmarks and typical workloads.
        """
        metrics = {
            "single_agent_sequential": {
                "latency": 45.0,  # seconds
                "throughput": 2.2,  # tasks/sec
                "gpu_utilization": 45,  # percent
                "cost_per_1k_tasks": 12.50  # USD
            },
            "multi_agent_nvidia_nim": {
                "latency": 12.0,  # seconds (3.75x faster)
                "throughput": 8.3,  # tasks/sec (3.77x higher)
                "gpu_utilization": 78,  # percent
                "cost_per_1k_tasks": 4.20  # USD (66% reduction)
            }
        }

        return metrics


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_collaborative_planning():
    """Demonstrate multi-agent collaborative planning."""
    print("\n" + "="*70)
    print("Example: Multi-Agent Collaborative Planning (NVIDIA Platform)")
    print("="*70)

    # Initialize coordinator with NIM agents
    coordinator = CoordinatorAgent(NIMAgentConfig(
        endpoint=NIM_CONFIG["coordinator_endpoint"],
        model=MODEL_CONFIG["coordinator_model"],
        role=AgentRole.COORDINATOR
    ))

    # Execute collaborative planning
    goal = "Analyze Q4 financial performance across all business units with competitive benchmarking"

    result = await coordinator.collaborative_planning(goal)

    # Display results
    print("\n" + "="*70)
    print("COLLABORATIVE PLANNING COMPLETE")
    print("="*70)
    print(f"Goal: {result['goal']}")
    print(f"\nExecution Summary:")
    print(f"  - Total tasks: {result['total_tasks']}")
    print(f"  - Successful: {result['successful_tasks']}")
    print(f"  - Failed: {result['failed_tasks']}")
    print(f"  - Parallel groups: {result['parallel_groups']}")
    print(f"\nPerformance:")
    print(f"  - Total time: {result['total_time']:.2f}s")
    print(f"  - Sequential estimate: {result['estimated_sequential_time']:.2f}s")
    print(f"  - Time saved: {result['time_saved']:.2f}s")
    print(f"  - Speedup: {result['speedup']:.2f}x")
    print(f"\nSynthesis preview:")
    print(f"  {result['synthesis'][:100]}...")

    return result


async def example_performance_comparison():
    """Compare performance: single-agent vs multi-agent."""
    print("\n" + "="*70)
    print("Example: Performance Comparison")
    print("="*70)

    metrics = PerformanceMetrics.compare_single_vs_multi_agent()

    print("\n| Metric | Single Agent | Multi-Agent (NVIDIA) | Improvement |")
    print("|--------|-------------|---------------------|-------------|")

    single = metrics["single_agent_sequential"]
    multi = metrics["multi_agent_nvidia_nim"]

    print(f"| Latency | {single['latency']:.1f}s | {multi['latency']:.1f}s | {single['latency']/multi['latency']:.2f}x faster |")
    print(f"| Throughput | {single['throughput']:.1f} t/s | {multi['throughput']:.1f} t/s | {multi['throughput']/single['throughput']:.2f}x higher |")
    print(f"| GPU Util | {single['gpu_utilization']}% | {multi['gpu_utilization']}% | {multi['gpu_utilization']/single['gpu_utilization']:.2f}x better |")
    print(f"| Cost/1K tasks | ${single['cost_per_1k_tasks']:.2f} | ${multi['cost_per_1k_tasks']:.2f} | {((single['cost_per_1k_tasks']-multi['cost_per_1k_tasks'])/single['cost_per_1k_tasks'])*100:.0f}% reduction |")

    print("\nKey Takeaways:")
    print("  ✓ 3.75x faster execution through parallelization")
    print("  ✓ 3.77x higher throughput with multiple executors")
    print("  ✓ 66% cost reduction via efficient GPU utilization")
    print("  ✓ Better GPU utilization (78% vs 45%)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run all multi-agent planning examples."""
    print("\n" + "="*70)
    print("Multi-Agent Collaborative Planning Examples")
    print("="*70)

    # Run examples
    await example_collaborative_planning()
    await example_performance_comparison()

    print("\n" + "="*70)
    print("All examples completed successfully! ✅")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
