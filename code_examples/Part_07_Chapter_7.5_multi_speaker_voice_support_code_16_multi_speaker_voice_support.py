# Agent response that quotes CEO statement
response_text = """
According to the CEO's statement during the earnings call,
<voice name="English-US.Male-2">We expect 20% revenue growth in fiscal year 2025, driven by strong AI adoption across enterprise customers.</voice>
This projection impacts your investment portfolio by increasing expected returns.
"""

# Riva processes SSML tags to switch voices
# Agent voice: English-US.Female-1 (main agent persona)
# CEO voice: English-US.Male-2 (authoritative male voice for quote)
request = SynthesizeSpeechRequest(
    text=response_text,
    language_code="en-US",
    voice_name="English-US.Female-1",
    ssml_enabled=True  # Enable SSML tag processing
)

response = tts_service.synthesize(request)