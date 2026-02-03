# User calls support: "My printer isn't working"
user_query = riva_asr.transcribe(user_audio)

# Agent requests visual information
riva_tts.speak("Can you send me a photo of your printer's control panel?")

# User uploads photo
printer_image = receive_user_image()

# Neva analyzes image for error indicators
diagnostic_result = neva_vlm.generate(
    image=printer_image,
    prompt="Identify any error messages, warning lights, or unusual indicators on this printer control panel."
)

# Neva response: "The control panel shows a flashing orange light next to the paper icon, and the display shows 'PAPER JAM TRAY 2'."

# Agent provides resolution steps
resolution_steps = get_troubleshooting_steps(issue="paper_jam_tray_2")

response = f"I see the issue. {diagnostic_result} Here's how to clear it: {resolution_steps}"

riva_tts.speak(response)