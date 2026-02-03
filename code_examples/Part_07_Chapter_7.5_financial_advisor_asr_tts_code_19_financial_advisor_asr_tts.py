from riva.client import ASRService, TTSService, SpeakerVerificationService
import grpc

# Initialize Riva services
channel = grpc.insecure_channel("localhost:50051")
asr_service = ASRService(channel)
tts_service = TTSService(channel)
speaker_verify_service = SpeakerVerificationService(channel)

# Step 1: Greet caller and request authentication
greeting = "Welcome to Acme Wealth Management. Please state your name and date of birth for verification."

tts_request = TTSService.synthesize_request(
    text=greeting,
    language_code="en-US",
    voice_name="English-US.Male-1",
    pitch="-3st",     # Slightly deeper for professional tone
    rate="0.95"       # Slightly slower for clarity
)

greeting_audio = tts_service.synthesize(tts_request)
play_audio(greeting_audio)

# Step 2: Capture caller's voice for authentication
auth_audio = record_audio(duration=5.0)  # 5 seconds of speech

# Step 3: Perform speaker verification against enrolled voiceprint
verification_result = speaker_verify_service.verify(
    audio=auth_audio,
    user_id="client_12345",  # Retrieved from phone number lookup
    threshold=0.85           # Confidence threshold for acceptance
)

if verification_result.confidence < 0.85:
    # Authentication failed
    failure_msg = "I'm unable to verify your identity. Please call back using your registered phone number or speak with a human representative."
    failure_audio = tts_service.synthesize(failure_msg)
    play_audio(failure_audio)
    terminate_call()

# Step 4: Authentication succeeded - handle portfolio inquiry
authenticated_prompt = "Authentication successful. How can I help you today?"
prompt_audio = tts_service.synthesize(authenticated_prompt)
play_audio(prompt_audio)

# Step 5: Process user query via ASR
user_query_audio = record_audio_until_silence()
query_transcript = asr_service.transcribe(user_query_audio)

# User asked: "What's my portfolio performance year-to-date?"

# Step 6: Retrieve portfolio data and generate response
portfolio_data = get_portfolio_performance(user_id="client_12345")

# Generate response with context-appropriate prosody
response_text = f"""
Your portfolio has returned {portfolio_data['ytd_return']:.2f}% year-to-date.
<emphasis level="strong">This outperforms the S&P 500 benchmark by 2.3%.</emphasis>
Your current balance is ${portfolio_data['total_value']:,.2f}.
Would you like a detailed breakdown by asset class?
"""

# Synthesize with SSML tags for emphasis on outperformance
tts_request = TTSService.synthesize_request(
    text=response_text,
    language_code="en-US",
    voice_name="English-US.Male-1",
    pitch="-3st",
    rate="0.95",
    ssml_enabled=True  # Enable SSML tags for emphasis
)

response_audio = tts_service.synthesize(tts_request)
play_audio(response_audio)