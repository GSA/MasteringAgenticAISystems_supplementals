# Streaming TTS - audio starts playing immediately
with tts_service.synthesize_online(request) as stream:
    for audio_chunk in stream:
        play_audio_chunk(audio_chunk)  # Play immediately, don't wait for complete synthesis

# Total latency: ~100ms to first audio (vs. 500ms for full synthesis)