# User calls in and speaks naturally
verification_audio = record_audio(duration=5.0)

# Compare against enrolled voiceprint
verification_result = sv_service.verify(
    user_id="user_12345",
    audio_sample=verification_audio,
    threshold=0.85  # Minimum similarity score for authentication (0-1)
)

if verification_result.confidence >= 0.85:
    print(f"Authentication successful (confidence: {verification_result.confidence:.3f})")
    # Grant account access
else:
    print(f"Authentication failed (confidence: {verification_result.confidence:.3f})")
    # Deny access, request alternative authentication