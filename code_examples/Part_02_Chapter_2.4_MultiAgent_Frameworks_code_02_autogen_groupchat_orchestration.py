# Create group chat with all agents
groupchat = GroupChat(
    agents=[user_proxy, researcher, synthesizer, critic],
    messages=[],
    max_round=12,  # Prevent infinite conversation loops
    speaker_selection_method="auto"  # LLM determines optimal speaker order
)

# GroupChatManager orchestrates the conversation
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Initiate the research workflow with a user request
research_query = """Generate a literature review on 'multi-agent reinforcement
learning for robotics'. Focus on papers from the last 3 years. The review should:
1. Identify key methodologies (model-free, model-based, hybrid approaches)
2. Compare coordination strategies (centralized, decentralized, hierarchical)
3. Highlight benchmark environments and evaluation metrics
4. Summarize open challenges and future research directions

Target length: 1500-2000 words with proper citations."""

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message=research_query
)
