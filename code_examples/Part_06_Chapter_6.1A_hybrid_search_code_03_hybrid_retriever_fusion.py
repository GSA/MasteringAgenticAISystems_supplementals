# Hybrid retrieval combining vector search and BM25
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vector_db, corpus, alpha=0.5):
        self.vector_db = vector_db  # Dense retrieval
        self.bm25 = BM25Okapi([doc.split() for doc in corpus])  # Sparse
        self.alpha = alpha  # Fusion weight

    def retrieve(self, query, k=10):
        # Dense retrieval
        dense_results = self.vector_db.search(query, k=k*2)

        # Sparse retrieval (BM25)
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        sparse_indices = np.argsort(bm25_scores)[-k*2:][::-1]

        # Reciprocal Rank Fusion (RRF)
        scores = {}
        for rank, doc in enumerate(dense_results):
            scores[doc.id] = scores.get(doc.id, 0) + self.alpha / (rank + 60)

        for rank, idx in enumerate(sparse_indices):
            doc_id = self.corpus[idx].id
            scores[doc_id] = scores.get(doc_id, 0) + (1-self.alpha) / (rank + 60)

        # Return top-k by fused score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
