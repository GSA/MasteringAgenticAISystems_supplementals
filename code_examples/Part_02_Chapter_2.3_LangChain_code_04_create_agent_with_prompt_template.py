from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the prompt template
# This template structures how the agent reasons about tasks
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful research assistant. "
     "Use the web search tool to find current information when needed. "
     "Provide clear, concise answers based on search results. "
     "If you cannot find relevant information, say so directly."),

    # Placeholder for conversation history from memory
    MessagesPlaceholder(variable_name="chat_history"),

    # Current user query
    ("human", "{input}"),

    # Placeholder for agent scratchpad (thought-action-observation history)
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the OpenAI Functions agent
# This combines the LLM, tools, and prompt into the agent logic
agent = create_openai_tools_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# Wrap the agent with AgentExecutor
# The executor handles the ReAct loop, tool calling, and memory management
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    memory=memory,
    verbose=True,  # Print reasoning steps to console for debugging
    handle_parsing_errors=True  # Gracefully handle malformed tool calls
)
