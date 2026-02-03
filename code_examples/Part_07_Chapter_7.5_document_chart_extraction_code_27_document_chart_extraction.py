# Client uploads quarterly earnings report PDF
document = load_pdf("Q3_2024_Earnings.pdf")

# Extract pages with charts
chart_pages = extract_pages_with_images(document)

# Client asks via voice: "How did Azure revenue perform?"
query = riva_asr.transcribe(client_audio)

# Process chart with Neva
for page in chart_pages:
    chart_image = page.extract_image()

    chart_analysis = neva_vlm.generate(
        image=chart_image,
        prompt="Extract revenue data from this chart, focusing on Azure/cloud services."
    )

    # Neva response: "This bar chart shows Azure revenue: Q3 2023: $24.3B, Q3 2024: $31.4B. This represents 29% year-over-year growth."

# Combine chart data with textual analysis from RAG
text_context = rag_retrieval(query="Azure revenue Q3 2024", document=document)

# Agent synthesizes complete response
response = f"Based on the earnings report, {chart_analysis} The report notes that this growth was driven by AI services adoption and enterprise migration to cloud. {text_context}"

riva_tts.speak(response)