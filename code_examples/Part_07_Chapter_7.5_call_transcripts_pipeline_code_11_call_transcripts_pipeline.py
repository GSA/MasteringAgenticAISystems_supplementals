# Load call transcripts (format: {id, text, call_duration, agent_id, timestamp})
transcripts = DocumentDataset.read_json("call_transcripts.jsonl")

# Stage 1: Remove incomplete calls (too short to be useful)
transcripts = transcripts.filter(
    WordCountFilter(min_words=100)  # Complete conversations are typically 200+ words
)

# Stage 2: Remove calls with excessive ASR errors (detected via unusual token patterns)
transcripts = transcripts.filter(
    PerplexityFilter(max_perplexity=2000)  # Higher threshold than web text (spoken language is less formal)
)

# Stage 3: Classify by call type (support vs. sales vs. other)
call_classifier = DomainClassifier(
    model="bert-base-uncased",
    domains=["technical_support", "billing_inquiry", "sales", "survey", "other"],
    quality_threshold=0.8
)

transcripts = call_classifier.classify(transcripts)

# Keep only support-related calls
transcripts = transcripts.filter(
    lambda doc: doc["domain"] in ["technical_support", "billing_inquiry"]
    and doc["domain_confidence"] > 0.8
)

# Stage 4: Redact sensitive information
pii_redactor = PIIRedaction(
    language="en",
    entities=[
        "ACCOUNT_NUMBER",    # Custom entity for account IDs
        "CREDIT_CARD",
        "PHONE_NUMBER",
        "EMAIL",
        "SSN",
        "PERSON"             # Customer and agent names
    ],
    replacement_strategy="placeholder"
)

transcripts = pii_redactor(transcripts)

# Stage 5: Remove near-duplicate conversations (many calls follow identical scripts)
fuzzy_dedup = FuzzyDuplicates(
    similarity_threshold=0.90  # Higher threshold than web text (some repetition is expected)
)

transcripts = fuzzy_dedup(transcripts)

# Save curated transcripts
transcripts.to_json("curated_transcripts.jsonl")