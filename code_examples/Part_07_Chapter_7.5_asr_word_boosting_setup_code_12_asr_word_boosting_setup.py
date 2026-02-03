from riva.client import ASRService

asr_service = ASRService(uri="localhost:50051")

# Boost medical terminology
word_boosting = [
    "myocardial infarction",
    "pulmonary embolism",
    "arrhythmia",
    "bradycardia",
    "tachycardia",
    "hypertension",
    "dyspnea",
    "hemoglobin"
]

config = asr_service.streaming_config(
    language_code="en-US",
    boosted_words=word_boosting,
    boosted_score=20.0  # How much to bias toward these words (0-100)
)

# Streaming recognition with word boosting
with asr_service.streaming_response(config) as stream:
    for audio_chunk in microphone_stream():
        stream.send_audio(audio_chunk)

        transcript = stream.receive_transcript()
        if transcript.is_final:
            print(f"Final: {transcript.text}")