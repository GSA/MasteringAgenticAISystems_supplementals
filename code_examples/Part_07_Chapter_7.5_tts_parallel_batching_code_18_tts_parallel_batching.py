response_text = """
Your checking account balance is $3,247.18 as of today.
Your savings account balance is $12,450.00.
You have no pending transactions.
"""

# Split into sentences and synthesize in parallel
sentences = response_text.split(". ")

# Synthesize first sentence immediately
first_audio = tts_service.synthesize(sentences[0])
play_audio(first_audio)

# Synthesize remaining sentences while first plays
for sentence in sentences[1:]:
    audio = tts_service.synthesize(sentence)
    play_audio(audio)

# Perceived latency: ~150ms (time to first audio)
# Total synthesis time: ~400ms (but parallelized with playback)