from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os

# Configure LLM settings shared across agents
llm_config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Researcher agent: Specializes in finding and extracting paper information
researcher = AssistantAgent(
    name="Researcher",
    system_message="""You are an expert research assistant specializing in academic
    literature search. Your responsibilities:
    - Search academic databases (arXiv, PubMed, Google Scholar) for relevant papers
    - Extract key information: titles, authors, abstracts, key findings
    - Summarize research methodologies and results
    - Identify the most impactful papers based on citations and relevance

    Provide structured output with paper details formatted consistently.""",
    llm_config=llm_config,
    human_input_mode="NEVER"  # Fully autonomous
)

# Synthesizer agent: Combines research findings into coherent narratives
synthesizer = AssistantAgent(
    name="Synthesizer",
    system_message="""You are an expert at synthesizing research literature into
    comprehensive reviews. Your responsibilities:
    - Analyze research findings from multiple papers
    - Identify common themes, methodologies, and conclusions
    - Compare and contrast different approaches
    - Highlight research gaps and future directions
    - Generate well-structured literature review sections

    Write in clear academic prose with proper citations.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Critic agent: Reviews output for quality and completeness
critic = AssistantAgent(
    name="Critic",
    system_message="""You are a critical reviewer ensuring research quality.
    Your responsibilities:
    - Verify factual accuracy of claims about papers
    - Check for logical coherence in synthesis
    - Identify missing perspectives or important papers
    - Suggest improvements for clarity and completeness
    - Validate that citations are properly formatted

    Provide specific, actionable feedback.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# User proxy: Represents the human researcher who initiates the task
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Human approves final output
    code_execution_config=False  # No code execution needed for this workflow
)
