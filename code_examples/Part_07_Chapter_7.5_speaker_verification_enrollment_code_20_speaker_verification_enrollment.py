from riva.client import SpeakerVerificationService

sv_service = SpeakerVerificationService(uri="localhost:50051")

# User speaks enrollment phrase (15-30 seconds of speech)
enrollment_phrases = [
    "My voice is my password verify me",
    "I am enrolling in voice authentication",
    "This is my voice pattern for security"
]

# Record user speaking all enrollment phrases
enrollment_audio = []
for phrase in enrollment_phrases:
    print(f"Please say: {phrase}")
    audio = record_audio(duration=5.0)
    enrollment_audio.append(audio)

# Create voiceprint from enrollment audio
voiceprint = sv_service.enroll(
    user_id="user_12345",
    audio_samples=enrollment_audio,
    language="en-US"
)

# Store voiceprint in secure database
save_voiceprint(user_id="user_12345", voiceprint=voiceprint)