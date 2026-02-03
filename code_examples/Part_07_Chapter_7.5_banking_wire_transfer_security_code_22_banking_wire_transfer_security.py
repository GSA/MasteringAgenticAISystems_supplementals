# User calls in: "I need to wire $50,000 to my business checking account"

# Step 1: ASR transcribes request
transfer_request = asr_service.transcribe(user_audio)

# Detect high-risk operation (wire transfer > $10,000)
if is_high_risk_operation(transfer_request):
    # Step 2: Request knowledge-based authentication
    tts_service.speak("I'll need to verify your identity for this wire transfer. Please provide your account number and the last four digits of your Social Security number.")

    auth_info = asr_service.transcribe(record_audio_until_silence())

    if not verify_account_info(auth_info):
        tts_service.speak("The information provided doesn't match our records. Please try again or speak with a representative.")
        terminate_call()

    # Step 3: Perform voice biometric verification (transparent to user)
    # Use audio from user's request as verification sample
    verification_result = sv_service.verify(
        user_id=extract_user_id(auth_info),
        audio_sample=user_audio,
        threshold=0.88  # Higher threshold for high-risk operations
    )

    if verification_result.confidence < 0.88:
        # Voice doesn't match enrolled voiceprint - possible account compromise
        tts_service.speak("I'm unable to verify your identity via voice authentication. For your security, please visit a branch or use online banking to complete this transfer.")

        # Alert fraud department
        alert_fraud_team(
            user_id=extract_user_id(auth_info),
            reason="Voice verification failed for wire transfer request",
            confidence=verification_result.confidence
        )

        terminate_call()

    # Step 4: Both authentication factors passed - process transfer
    tts_service.speak("Authentication successful. I'm processing your wire transfer of $50,000 to your business checking account. This will complete within 1-2 business days.")

    execute_wire_transfer(request=transfer_request)