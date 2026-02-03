from riva.client import ASRService, TTSService
from neva.client import NevaVLM
import grpc

# Initialize services
asr = ASRService(uri="localhost:50051")
tts = TTSService(uri="localhost:50051")
vlm = NevaVLM(uri="localhost:8000")

# Step 1: Claimant calls in
tts.speak("Thank you for calling AutoInsure claims. I'll help you file your claim. Can you briefly describe what happened?")

# Claimant speaks: "I was in a fender bender this morning. The other driver hit my rear bumper in a parking lot."
incident_description = asr.transcribe(record_audio_until_silence())

# Step 2: Request photos
tts.speak("I'm sorry to hear that. To process your claim quickly, please text photos of the damage to 555-0123. I'll analyze them while we talk.")

# Claimant uploads 3 photos via SMS
damage_photos = receive_sms_images(phone_number="555-867-5309")

# Step 3: Neva VLM analyzes damage
damage_assessments = []

for photo in damage_photos:
    assessment = vlm.generate(
        image=photo,
        prompt="""Analyze this vehicle damage photo. Identify:
        1. Damaged components (bumper, lights, panels, etc.)
        2. Severity (minor/moderate/severe)
        3. Visible safety concerns (structural damage, leaking fluids)
        4. Estimated repair complexity (simple/moderate/complex)
        """
    )

    damage_assessments.append(assessment)

# Neva assessments:
# Photo 1: "Rear bumper damage - moderate severity. Bumper cover is dented and paint is scratched. No cracks visible. Bumper appears to retain structural integrity. Assessment: Simple repair - bumper replacement."
# Photo 2: "Taillight damage - minor severity. Right taillight lens is cracked but housing is intact. No electrical damage visible. Assessment: Simple repair - taillight replacement."
# Photo 3: "Rear panel view - no additional damage visible beyond bumper and taillight."

# Step 4: Estimate repair costs using damage assessment
damage_summary = {
    "components": ["rear_bumper", "right_taillight"],
    "severity": "moderate",
    "safety_concerns": False,
    "repair_complexity": "simple"
}

estimated_cost = calculate_repair_cost(damage_summary)
# Result: $1,850 (bumper replacement $1,400 + taillight $450)

# Step 5: Provide immediate estimate to claimant
response = f"""
I've analyzed your photos. The damage is moderate, affecting your rear bumper and right taillight.
The good news is there are no structural or safety concerns - this is straightforward cosmetic damage.

My estimated repair cost is ${estimated_cost:,}. Your policy has a $500 deductible, so your out-of-pocket would be $500.

I can pre-approve this claim immediately and set up a repair appointment at one of our partner shops. Would you like to proceed?
"""

tts.speak(response)

# Step 6: Handle claimant decision
claimant_response = asr.transcribe(record_audio_until_silence())

if "yes" in claimant_response.lower() or "proceed" in claimant_response.lower():
    # Pre-approve claim, schedule repair
    claim_id = create_claim(
        description=incident_description,
        damage_photos=damage_photos,
        damage_assessment=damage_assessments,
        estimated_cost=estimated_cost,
        status="pre_approved"
    )

    repair_appointment = schedule_repair(
        claim_id=claim_id,
        claimant_location="user_zip_code",
        preferred_date="next_available"
    )

    confirmation = f"Perfect. I've created claim {claim_id} and scheduled your repair for {repair_appointment.date} at {repair_appointment.shop_name}. You'll receive a confirmation email shortly. Is there anything else I can help with?"

    tts.speak(confirmation)