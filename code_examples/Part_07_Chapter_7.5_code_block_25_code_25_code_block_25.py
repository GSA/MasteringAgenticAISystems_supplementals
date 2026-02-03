# User opens accessibility app and asks: "What's in front of me?"
voice_query = riva_asr.transcribe(user_audio)
# Transcribed: "What's in front of me?"

# Capture image from phone camera
image = capture_camera_frame()

# Neva VLM describes scene
scene_description = neva_vlm.generate(
    image=image,
    prompt="Describe this scene in detail, focusing on obstacles, doorways, signs, and people."
)

# Neva response: "You are standing in a hallway. Directly ahead (3 meters) is a closed door with a red EXIT sign. To your left (1 meter) is a water fountain. To your right is a wall with a bulletin board. The floor is clear with no obstacles."

# Agent enhances with contextual guidance
agent_response = f"{scene_description} I recommend walking straight ahead to reach the exit door."

# Riva TTS speaks response to user
riva_tts.synthesize_and_play(agent_response)