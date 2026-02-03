from langchain.agents import Tool, AgentExecutor, create_react_agent

tools = [
    Tool(
        name="search_kb",
        func=lambda q: vector_db.search(embed_text(q)),
        description="Search knowledge base for relevant articles"
    )
]

agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(agent=agent, tools=tools)

answer = executor.invoke({"input": user_question})
