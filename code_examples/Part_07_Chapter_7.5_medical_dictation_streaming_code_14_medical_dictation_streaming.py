import grpc
from riva.client import ASRService, StreamingRecognitionConfig

# Connect to Riva server running on local GPU workstation
channel = grpc.insecure_channel("localhost:50051")
asr_service = ASRService(channel)

# Medical terminology word boosting (500 common medical terms)
medical_vocab = [
    # Conditions
    "myocardial infarction", "pulmonary embolism", "pneumothorax",
    "sepsis", "diabetic ketoacidosis", "acute respiratory distress syndrome",

    # Medications
    "nitroglycerin", "epinephrine", "atropine", "adenosine",
    "alteplase", "heparin", "enoxaparin",

    # Procedures
    "endotracheal intubation", "central venous catheter",
    "chest tube insertion", "cardioversion", "defibrillation",

    # Symptoms
    "dyspnea", "tachycardia", "bradycardia", "hypotension",
    "hemoptysis", "diaphoresis", "cyanosis"
]

# Configure streaming ASR with medical customization
streaming_config = StreamingRecognitionConfig(
    config=RecognitionConfig(
        language_code="en-US",
        model="conformer",              # State-of-the-art acoustic model
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,    # Per-word confidence scores
        boosted_lm_words=medical_vocab,
        boosted_lm_score=25.0,          # Strong bias toward medical terms
        audio_channel_count=1
    ),
    interim_results=True  # Return provisional transcripts for low latency
)

# Stream audio from physician's headset microphone
with asr_service.streaming_response(streaming_config) as stream:
    print("Medical dictation agent active. Begin dictating...")

    for audio_chunk in microphone_stream(device_id="headset"):
        # Send 160ms audio chunks (80ms overlap for stability)
        stream.send_audio(audio_chunk)

        # Receive transcript results
        response = stream.receive_transcript()

        if response.is_final:
            # Finalized transcript segment - commit to medical record
            transcript_text = response.alternatives[0].transcript
            confidence = response.alternatives[0].confidence

            # Flag low-confidence segments for manual review
            if confidence < 0.85:
                print(f"[LOW CONFIDENCE: {confidence:.2f}] {transcript_text}")
                # Mark for physician review before committing to EHR
            else:
                print(f"[FINAL: {confidence:.2f}] {transcript_text}")
                # Automatically commit to electronic health record
        else:
            # Provisional transcript - display for real-time feedback
            transcript_text = response.alternatives[0].transcript
            print(f"[PROVISIONAL] {transcript_text}", end="\r")