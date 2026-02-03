# Sequential process: Research → Write → Edit in strict order
content_crew_sequential = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

# Execute the content production workflow
result = content_crew_sequential.kickoff()
print(f"Final blog post:\n{result}")

# Hierarchical process: Manager coordinates and delegates dynamically
content_crew_hierarchical = Crew(
    agents=[editorial_manager, researcher, writer, editor],
    tasks=[
        Task(
            description="Produce a publication-ready blog post on RAG optimization",
            expected_output="Complete blog post meeting all quality standards",
            agent=editorial_manager  # Manager receives high-level goal
        )
    ],
    process=Process.hierarchical,
    manager_llm="gpt-4o",  # LLM powering manager decisions
    verbose=True
)

# Manager breaks down the goal and delegates to specialists
result = content_crew_hierarchical.kickoff()
