# Detect query-image mismatch
query_topic = extract_topic(transcription)  # "revenue trend"
image_content = extract_content_type(image_description)  # "product photo"

if not topics_match(query_topic, image_content):
    response = "I see you're asking about revenue trends, but the image shows a product photo. Did you mean to upload a different image, such as a chart or financial report?"

    tts.speak(response)

    # Wait for user clarification
    clarification = asr.transcribe(record_audio_until_silence())