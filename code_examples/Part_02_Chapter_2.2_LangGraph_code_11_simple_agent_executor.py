from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool

# Define knowledge base search tool
search_tool = Tool(
    name="search_knowledge_base",
    func=lambda q: vector_db.search(embed_text(q)),
    description="Search company knowledge base for relevant articles"
)

# Create agent with tools
agent = create_react_agent(llm, [search_tool], react_prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])

# Execute query
response = executor.invoke({"input": user_question})
