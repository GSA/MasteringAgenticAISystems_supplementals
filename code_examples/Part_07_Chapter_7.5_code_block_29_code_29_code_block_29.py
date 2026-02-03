import asyncio

async def process_multimodal_query(audio, image):
    # Start both ASR and VLM processing simultaneously
    transcription_task = asyncio.create_task(
        asr.transcribe_async(audio)
    )

    image_analysis_task = asyncio.create_task(
        vlm.analyze_async(image, prompt="Describe this image in detail")
    )

    # Wait for both to complete
    transcription, image_description = await asyncio.gather(
        transcription_task,
        image_analysis_task
    )

    # Combine results for LLM reasoning
    combined_context = f"User query: {transcription}\nImage content: {image_description}"

    # LLM generates response
    response = await llm.generate_async(combined_context)

    # TTS synthesizes response
    audio_response = await tts.synthesize_async(response)

    return audio_response

# Parallel processing reduces latency:
# Sequential: 500ms (ASR) + 400ms (VLM) + 300ms (LLM) + 150ms (TTS) = 1350ms
# Parallel: max(500ms ASR, 400ms VLM) + 300ms (LLM) + 150ms (TTS) = 950ms
# Improvement: 30% faster