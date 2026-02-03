from riva.client import TTSService, SynthesizeSpeechRequest

tts_service = TTSService(uri="localhost:50051")

# Configure voice characteristics
request = SynthesizeSpeechRequest(
    text="Your checking account balance is $3,247.18 as of today.",
    language_code="en-US",
    voice_name="English-US.Male-1",     # Pre-trained voice identity
    sample_rate_hertz=22050,
    audio_encoding="LINEAR_PCM",

    # Prosody customization
    pitch="-5st",        # Lower pitch by 5 semitones (deeper, more authoritative)
    rate="0.9",          # Speak 10% slower (clearer, more deliberate)
    volume="+3dB"        # Slightly louder (compensate for phone line attenuation)
)

# Synthesize speech
response = tts_service.synthesize(request)

# Play audio or stream to user's phone line
play_audio(response.audio)