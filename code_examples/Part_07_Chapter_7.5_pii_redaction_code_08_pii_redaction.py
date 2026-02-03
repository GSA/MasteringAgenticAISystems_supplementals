from nemo_curator.modules import PIIRedaction

# Redact personally identifiable information
pii_redactor = PIIRedaction(
    language="en",
    entities=["EMAIL", "PHONE_NUMBER", "SSN", "CREDIT_CARD", "PERSON", "LOCATION"],
    replacement_strategy="placeholder"  # Replace with [EMAIL], [PHONE], etc.
)

dataset = pii_redactor(dataset)