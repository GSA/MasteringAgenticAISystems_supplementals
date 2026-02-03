from nemo_curator import DocumentDataset
from nemo_curator.filters import (
    WordCountFilter,
    PerplexityFilter,
    LanguageIdentificationFilter,
    DocumentQualityFilter
)
from nemo_curator.modules import ExactDuplicates, FuzzyDuplicates
from nemo_curator.classifiers import DomainClassifier

# Load dataset with lazy loading (streams from disk, doesn't load all into RAM)
dataset = DocumentDataset.read_json(
    "raw_financial_data.jsonl",
    backend="dask"  # Distributed processing framework
)

print(f"Initial dataset: {len(dataset)} documents")