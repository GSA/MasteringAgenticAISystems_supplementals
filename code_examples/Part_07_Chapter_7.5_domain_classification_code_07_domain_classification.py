# Classify documents by domain and quality
classifier = DomainClassifier(
    model="bert-base-uncased",
    domains=[
        "financial_analysis",   # Detailed market/company analysis
        "earnings_reports",     # Official quarterly/annual reports
        "regulatory_filings",   # SEC filings (10-K, 10-Q, 8-K)
        "financial_news",       # News articles about markets/companies
        "general_finance"       # Generic financial content
    ],
    quality_threshold=0.7  # Keep documents with >70% confidence in target domains
)

dataset = classifier.classify(dataset)

# Filter to high-value domains
dataset = dataset.filter(
    lambda doc: doc["domain"] in [
        "financial_analysis",
        "earnings_reports",
        "regulatory_filings"
    ] and doc["domain_confidence"] > 0.7
)

print(f"After domain filtering: {len(dataset)} documents")
# Expected: ~800 million documents (focused on high-quality financial content)