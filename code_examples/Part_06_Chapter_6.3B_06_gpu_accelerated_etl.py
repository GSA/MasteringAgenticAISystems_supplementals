from nemo_curator import ScatterReduce
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.filters import WordCountFilter
import dask_cudf


# Load data distributed across GPUs
dataset = DocumentDataset.read_json(
    "s3://knowledge-base/*.jsonl",
    backend="cudf"  # GPU-accelerated backend
)

# GPU-accelerated quality filtering
word_filter = WordCountFilter(min_words=50, max_words=10000)
filtered = word_filter(dataset)

# GPU-accelerated exact deduplication
dedup = ExactDuplicates(hash_method="xxhash")
unique_docs = dedup(filtered)

# Fuzzy near-duplicate removal
fuzzy_dedup = ScatterReduce(
    minhash_lsh,
    jaccard_threshold=0.8,
    num_hashes=128
)
final_docs = fuzzy_dedup(unique_docs)
