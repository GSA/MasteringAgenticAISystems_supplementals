from functools import lru_cache

@lru_cache(maxsize=128)
def cached_arxiv_search(query: str) -> dict:
    """
    Cached ArXiv search with 128 query capacity.
    Cache hit: 0.5ms latency, $0 cost
    Cache miss: 1234ms latency, $0.0023 cost
    """
    return expensive_arxiv_search(query)
