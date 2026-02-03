# Agent client with fallback handling
try:
    transcript = asr_service.transcribe(audio, timeout=5.0)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        # ASR service overloaded, queue timeout
        tts_service.speak("I'm having trouble processing voice right now. Could you please type your question instead?")
        transcript = get_text_input_from_user()
    elif e.code() == grpc.StatusCode.UNAVAILABLE:
        # ASR service completely down
        tts_service.speak("Voice services are temporarily unavailable. Please try again shortly or speak with a human representative.")
        transfer_to_human_agent()
    else:
        # Unexpected error
        log_error(f"ASR error: {e}")
        raise