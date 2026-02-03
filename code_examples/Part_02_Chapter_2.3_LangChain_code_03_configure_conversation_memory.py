from langchain.memory import ConversationBufferMemory

# Initialize memory with return_messages=True for chat models
# This formats history as a list of message objects rather than a single string
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
