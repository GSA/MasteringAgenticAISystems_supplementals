# Turn 1: User uploads photo and asks
user_query_1 = "What's wrong with this device?"
image_1 = receive_image()

analysis_1 = vlm.generate(image_1, prompt="Identify any damage or defects")
response_1 = f"I see a cracked screen and damaged USB-C port. {resolution_suggestions}"

# Turn 2: User asks follow-up WITHOUT uploading new image
user_query_2 = "How much will that cost?"

# Agent must remember previous image analysis
# Store image_1 and analysis_1 in conversation context
context = {
    "image": image_1,
    "visual_analysis": analysis_1,
    "conversation_history": [
        {"role": "user", "content": user_query_1},
        {"role": "assistant", "content": response_1},
        {"role": "user", "content": user_query_2}
    ]
}

# Generate response using full context
response_2 = llm.generate(
    prompt=f"Context: {context}\n\nUser is asking about cost of repairs for: {analysis_1}\n\nProvide cost estimate."
)