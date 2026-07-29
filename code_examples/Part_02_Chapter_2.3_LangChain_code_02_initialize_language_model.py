from langchain_openai import ChatOpenAI

# ChatOpenAI reads OPENAI_API_KEY from the environment. Do not place keys in code.
# Initialize GPT-4 with temperature=0 for deterministic tool calling.
# Higher temperatures (0.3-0.7) work for creative tasks, but tool selection
# benefits from deterministic behavior.
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
)
