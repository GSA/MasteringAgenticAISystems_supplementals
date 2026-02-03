try:
    audio = tts_service.synthesize(response_text, timeout=3.0)
    play_audio(audio)
except grpc.RpcError:
    # TTS overloaded, fall back to text output
    display_text(response_text)