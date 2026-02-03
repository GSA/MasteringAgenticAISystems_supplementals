from nemo.agent_toolkit import AgentProfiler
from crewai import Agent, Task, Crew

# Define CrewAI agent
researcher = Agent(
    role="Research Analyst",
    goal="Find and summarize AI research",
    tools=[arxiv_search_tool, summarize_tool],
    llm=llm  # Same NIM-backed LLM
)

task = Task(
    description="Find latest AI papers and summarize",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task])

# Wrap with same profiler (framework-agnostic)
profiler = AgentProfiler(crew)
result = profiler.kickoff()
metrics = profiler.get_metrics()  # Identical metrics structure
