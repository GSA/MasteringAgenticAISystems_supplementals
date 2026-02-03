class DependencyGraph:
    def __init__(self):
        self.dependencies = {}  # agent_name -> list of dependencies
        self.reverse_deps = {}  # agent_name -> list of dependents

    def add_dependency(self, agent: str, depends_on: list[str]):
        """Declare that agent depends on depends_on agents completing."""

        # Check for cycle creation before adding dependency
        if self.would_create_cycle(agent, depends_on):
            raise ValueError(
                f"Circular dependency detected: {agent} -> {depends_on}\n"
                f"This would create a deadlock. Existing dependencies:\n"
                f"{self.format_dependency_chain(agent, depends_on)}"
            )

        self.dependencies[agent] = depends_on

        # Build reverse index for efficient queries
        for dep in depends_on:
            self.reverse_deps.setdefault(dep, []).append(agent)

    def would_create_cycle(self, agent: str, depends_on: list[str]) -> bool:
        """Use DFS to detect if adding this dependency creates a cycle."""

        visited = set()
        recursion_stack = set()

        def dfs(node: str) -> bool:
            """Returns True if cycle detected."""

            if node in recursion_stack:
                # Found node that's currently being explored - cycle!
                return True

            if node in visited:
                # Already fully explored this branch - no cycle here
                return False

            visited.add(node)
            recursion_stack.add(node)

            # Explore all dependencies of current node
            for dep in self.dependencies.get(node, []):
                if dfs(dep):
                    return True

            recursion_stack.remove(node)
            return False

        # Temporarily add the new dependencies to check for cycles
        temp_deps = self.dependencies.get(agent, [])
        self.dependencies[agent] = depends_on

        # Check if any new dependency creates a cycle back to agent
        cycle_found = any(dfs(dep) for dep in depends_on)

        # Restore original state
        self.dependencies[agent] = temp_deps

        return cycle_found

    def get_execution_order(self) -> list[list[str]]:
        """Return agents grouped into parallel execution phases."""

        # Topological sort with level assignment
        in_degree = {agent: len(deps)
                     for agent, deps in self.dependencies.items()}

        execution_phases = []
        remaining = set(self.dependencies.keys())

        while remaining:
            # Find all agents with no remaining dependencies
            ready = {agent for agent in remaining
                    if in_degree.get(agent, 0) == 0}

            if not ready:
                raise ValueError(
                    f"Deadlock detected in dependency graph: "
                    f"remaining agents {remaining} all have unresolved dependencies"
                )

            execution_phases.append(sorted(ready))

            # Remove ready agents and update dependents
            for agent in ready:
                remaining.remove(agent)
                for dependent in self.reverse_deps.get(agent, []):
                    in_degree[dependent] -= 1

        return execution_phases
