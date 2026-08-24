# Part 6, Chapter 6.6.3: "I Do" - Reranking Implementation

## Introduction

Basic retrieval systems rank documents by similarity scores computed between query and document embeddings—a fast operation requiring only dot products. However, this bi-encoder approach optimizes queries and documents independently, missing nuanced relationships that require joint reasoning. Reranking addresses this limitation by re-scoring the top-k retrieved candidates using cross-encoders that process query-document pairs jointly, capturing subtle relevance signals invisible to bi-encoders.

This section demonstrates two reranking implementations, each addressing different production constraints. First, we'll build a cross-encoder reranker using the open-source Sentence Transformers library, providing full control over model selection, latency, and deployment infrastructure. Second, we'll integrate Cohere's Rerank API, trading infrastructure complexity for managed performance and multilingual capabilities. Through comparative benchmarks, we'll quantify the precision gains and latency trade-offs, enabling you to select the appropriate reranking strategy for your RAG system's requirements.

## Example 1: Cross-Encoder Reranking with Sentence Transformers

### Understanding Cross-Encoder Architecture

Cross-encoders fundamentally differ from bi-encoders in how they process queries and documents. A bi-encoder (used in initial retrieval) embeds the query and document separately, then computes similarity via dot product. This independence enables fast retrieval across millions of documents since embeddings are precomputed. A cross-encoder, by contrast, concatenates the query and document as input, processing them jointly through transformer layers that can attend across both texts. This joint attention enables the model to identify subtle relevance patterns—keyword matches, semantic entailment, answer-to-question relationships—that bi-encoders miss.

The trade-off is computational cost. While bi-encoders require one forward pass per document (at indexing time) plus one for the query (at search time), cross-encoders require one forward pass per query-document pair at search time. For a query with 100 candidate documents, the cross-encoder performs 100 forward passes versus the bi-encoder's single query pass. This makes cross-encoders impractical for first-stage retrieval but perfect for second-stage reranking of top-k candidates where k is small (typically 5-20).

### Implementation Setup and Dependencies

Let's begin with a complete reranking implementation using Sentence Transformers:

```python
"""
Cross-Encoder Reranking Implementation
Demonstrates: Reranking top-k results, latency measurement, precision improvement
"""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import numpy as np


@dataclass
class RankedDocument:
    """Document with relevance score."""
    content: str
    source_doc: str
    chunk_index: int
    score: float
    initial_rank: int = None  # Track rank before reranking


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval precision.

    Uses Sentence Transformers cross-encoder models that jointly encode
    query-document pairs for more accurate relevance scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model ID. Options:
                - ms-marco-MiniLM-L-6-v2: Fast, 80MB, 40ms/query
                - ms-marco-MiniLM-L-12-v2: Balanced, 120MB, 90ms/query
                - ms-marco-TinyBERT-L-2-v2: Fastest, 50MB, 20ms/query
        """
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        print(f"  Model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[List[RankedDocument], Dict[str, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of retrieved documents with 'content', 'source_doc', 'chunk_index'
            top_k: Number of top results to return after reranking

        Returns:
            Tuple of (reranked_documents, timing_metrics)
        """
        start_time = time.perf_counter()

        # Prepare query-document pairs
        pairs = [(query, doc["content"]) for doc in documents]

        # Compute relevance scores
        scoring_start = time.perf_counter()
        scores = self.model.predict(pairs)
        scoring_time = (time.perf_counter() - scoring_start) * 1000

        # Create ranked documents with scores
        ranked_docs = [
            RankedDocument(
                content=doc["content"],
                source_doc=doc.get("source_doc", "unknown"),
                chunk_index=doc.get("chunk_index", i),
                score=float(score),
                initial_rank=i
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]

        # Sort by relevance score (descending)
        ranked_docs.sort(key=lambda x: x.score, reverse=True)

        # Return top-k
        top_results = ranked_docs[:top_k]

        total_time = (time.perf_counter() - start_time) * 1000

        metrics = {
            "total_time_ms": total_time,
            "scoring_time_ms": scoring_time,
            "num_documents_scored": len(documents),
            "ms_per_document": total_time / len(documents)
        }

        return top_results, metrics
```

The implementation centers on the `CrossEncoder` class from Sentence Transformers, which handles model loading and inference. The model choice involves a precision-latency trade-off: MiniLM-L-6 provides good accuracy in ~40ms per query with 20 candidates, while TinyBERT-L-2 achieves ~20ms but with slightly lower precision. For production systems serving hundreds of queries per second, these milliseconds matter—a 20ms difference means the distinction between serving 50 or 25 queries per second per worker.

The `rerank` method performs three operations: constructing query-document pairs, scoring them through the cross-encoder, and sorting by relevance scores. The timing breakdown separates scoring time (pure model inference) from total time (including data preparation and sorting), enabling precise performance analysis.

### Integration with RAG Pipeline

Now let's integrate reranking into a complete RAG pipeline that demonstrates the before-and-after comparison:

```python
class RAGSystemWithReranking:
    """RAG system with optional reranking stage."""

    def __init__(
        self,
        rag_system,  # Your existing RAG system with retrieve() method
        reranker: CrossEncoderReranker = None
    ):
        self.rag_system = rag_system
        self.reranker = reranker
        self.stats = {
            "queries_with_reranking": 0,
            "queries_without_reranking": 0,
            "avg_rerank_time_ms": 0,
            "rank_changes": []
        }

    def query(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5,
        use_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Query with optional reranking.

        Args:
            query: Search query
            initial_k: Number of candidates to retrieve initially
            final_k: Number of results to return after reranking
            use_reranking: Whether to apply reranking

        Returns:
            Query results with timing breakdown
        """
        start_time = time.perf_counter()

        # Stage 1: Initial retrieval
        retrieval_start = time.perf_counter()
        candidates = self.rag_system.retrieve(query, top_k=initial_k)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        # Stage 2: Optional reranking
        rerank_time = 0
        final_results = candidates[:final_k]
        rank_changes_count = 0

        if use_reranking and self.reranker:
            # Convert chunks to documents format
            documents = [
                {
                    "content": chunk.text,
                    "source_doc": chunk.source_doc,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in candidates
            ]

            reranked_docs, rerank_metrics = self.reranker.rerank(
                query,
                documents,
                top_k=final_k
            )

            rerank_time = rerank_metrics["total_time_ms"]

            # Track rank changes
            for new_rank, doc in enumerate(reranked_docs):
                if doc.initial_rank != new_rank:
                    rank_changes_count += 1

            # Update stats
            self.stats["queries_with_reranking"] += 1
            self.stats["avg_rerank_time_ms"] = (
                (self.stats["avg_rerank_time_ms"] * (self.stats["queries_with_reranking"] - 1) + rerank_time)
                / self.stats["queries_with_reranking"]
            )
            self.stats["rank_changes"].append(rank_changes_count)

            final_results = reranked_docs
        else:
            self.stats["queries_without_reranking"] += 1

        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "results": final_results,
            "metrics": {
                "retrieval_time_ms": retrieval_time,
                "rerank_time_ms": rerank_time,
                "total_time_ms": total_time,
                "reranking_enabled": use_reranking,
                "candidates_retrieved": initial_k,
                "results_returned": len(final_results),
                "rank_changes": rank_changes_count
            }
        }

    def print_comparison(self, query: str):
        """
        Print side-by-side comparison of results with and without reranking.
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        # Without reranking
        print("WITHOUT RERANKING:")
        print("-" * 80)
        results_baseline = self.query(query, initial_k=20, final_k=5, use_reranking=False)
        for i, doc in enumerate(results_baseline["results"][:5], 1):
            content_preview = doc.text[:150] if hasattr(doc, 'text') else doc.content[:150]
            print(f"{i}. [{doc.source_doc if hasattr(doc, 'source_doc') else 'N/A'}] {content_preview}...")
        print(f"\nLatency: {results_baseline['metrics']['total_time_ms']:.2f}ms")

        # With reranking
        print(f"\n{'='*80}")
        print("WITH RERANKING:")
        print("-" * 80)
        results_reranked = self.query(query, initial_k=20, final_k=5, use_reranking=True)
        for i, doc in enumerate(results_reranked["results"][:5], 1):
            content_preview = doc.content[:150]
            rank_change = f" (was rank {doc.initial_rank + 1})" if doc.initial_rank != i - 1 else ""
            print(f"{i}. [{doc.source_doc}] {content_preview}...{rank_change}")
            print(f"    Score: {doc.score:.4f}")

        print(f"\nLatency breakdown:")
        print(f"  Retrieval: {results_reranked['metrics']['retrieval_time_ms']:.2f}ms")
        print(f"  Reranking: {results_reranked['metrics']['rerank_time_ms']:.2f}ms")
        print(f"  Total: {results_reranked['metrics']['total_time_ms']:.2f}ms")
        print(f"  Rank changes: {results_reranked['metrics']['rank_changes']}/{results_reranked['metrics']['results_returned']}")
```

This integration demonstrates production-ready patterns. The `query` method accepts `initial_k` (candidates for reranking, typically 20-50) and `final_k` (results returned, typically 5-10). This two-stage retrieval—broad initial retrieval followed by precise reranking—balances recall (don't miss relevant documents) with precision (rank the most relevant highest).

The rank change tracking reveals reranking effectiveness. If reranking moves 3 out of 5 documents to different positions, it's substantially revising the bi-encoder's ranking. If it changes zero positions, reranking adds latency without benefit, suggesting either excellent bi-encoder quality or poor cross-encoder selection.

## Example 2: Cohere Rerank API Integration

### Understanding Commercial Reranking Services

Cohere's Rerank API provides enterprise-grade reranking with several advantages over self-hosted cross-encoders: models trained on proprietary datasets covering 100+ languages, inference infrastructure optimized for sub-50ms latency at scale, automatic model updates incorporating latest research, and elimination of deployment overhead. The trade-offs are cost (typically $1-2 per 1000 requests) and external dependency (requires API connectivity).

Let's implement Cohere Rerank integration with comprehensive error handling:

```python
"""
Cohere Rerank API Integration
Demonstrates: Managed reranking, multilingual support, production error handling
"""

import os
from typing import List, Dict, Any, Optional
import cohere
from cohere.error import CohereAPIError, CohereConnectionError


class CohereReranker:
    """
    Production-ready Cohere Rerank integration.

    Handles: API authentication, rate limiting, error recovery, metric tracking
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            model: Rerank model to use:
                - rerank-english-v2.0: English-optimized, fastest
                - rerank-multilingual-v2.0: 100+ languages, 20% slower
        """
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY or pass api_key parameter.")

        self.client = cohere.Client(self.api_key)
        self.model = model

        # Tracking metrics
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0

        print(f"Initialized Cohere Reranker with model: {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        max_retries: int = 3
    ) -> Tuple[List[RankedDocument], Dict[str, Any]]:
        """
        Rerank documents using Cohere API.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top results to return
            max_retries: Number of retries on transient failures

        Returns:
            Tuple of (reranked_documents, metrics)
        """
        start_time = time.perf_counter()

        # Prepare documents for Cohere (requires 'text' field or plain strings)
        doc_texts = [doc.get("content", doc.get("text", str(doc))) for doc in documents]

        # Retry logic for transient failures
        last_error = None
        for attempt in range(max_retries):
            try:
                # Call Cohere Rerank API
                api_start = time.perf_counter()
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=doc_texts,
                    top_n=top_k,
                    return_documents=False  # We already have document content
                )
                api_time = (time.perf_counter() - api_start) * 1000

                # Build ranked results
                ranked_docs = []
                for result in response.results:
                    idx = result.index
                    ranked_docs.append(
                        RankedDocument(
                            content=doc_texts[idx],
                            source_doc=documents[idx].get("source_doc", "unknown"),
                            chunk_index=documents[idx].get("chunk_index", idx),
                            score=result.relevance_score,
                            initial_rank=idx
                        )
                    )

                # Update metrics
                total_time = (time.perf_counter() - start_time) * 1000
                self.request_count += 1
                self.total_latency_ms += total_time

                metrics = {
                    "total_time_ms": total_time,
                    "api_time_ms": api_time,
                    "num_documents": len(documents),
                    "returned_documents": len(ranked_docs),
                    "attempt": attempt + 1,
                    "avg_latency_ms": self.total_latency_ms / self.request_count
                }

                return ranked_docs, metrics

            except CohereConnectionError as e:
                last_error = e
                wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)

            except CohereAPIError as e:
                # API errors (rate limits, invalid requests) don't retry
                self.error_count += 1
                raise RuntimeError(f"Cohere API error: {e}") from e

            except Exception as e:
                self.error_count += 1
                raise RuntimeError(f"Unexpected reranking error: {e}") from e

        # All retries exhausted
        self.error_count += 1
        raise RuntimeError(f"Reranking failed after {max_retries} attempts: {last_error}")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_latency_ms": self.total_latency_ms / max(self.request_count, 1)
        }
```

The Cohere integration demonstrates production error handling patterns. Connection errors trigger exponential backoff retry (100ms, 200ms, 400ms delays) since network hiccups are transient. API errors like rate limits or invalid requests don't retry since they require caller intervention. The metrics tracking enables monitoring dashboard integration—you can alert when error rate exceeds 1% or average latency exceeds 200ms.

## Performance Benchmarking and Analysis

### Comparative Evaluation

Let's benchmark both reranking approaches to quantify their trade-offs:

```python
def benchmark_rerankers():
    """
    Compare Cross-Encoder vs Cohere Rerank performance.
    """
    # Initialize both rerankers
    cross_encoder = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cohere_reranker = CohereReranker(model="rerank-english-v2.0")

    # Sample queries and documents (in production, use your test set)
    test_queries = [
        "What is the memory bandwidth of NVIDIA H100?",
        "How does flash attention improve transformer efficiency?",
        "Explain the difference between RLHF and RLAIF",
    ]

    # Simulate retrieved documents (normally from your RAG system)
    sample_docs = [
        {"content": f"Document {i} content discussing relevant technical topics...",
         "source_doc": f"doc_{i}", "chunk_index": i}
        for i in range(20)
    ]

    print("\n" + "="*80)
    print("RERANKING PERFORMANCE BENCHMARK")
    print("="*80)

    # Benchmark Cross-Encoder
    print("\nCross-Encoder (ms-marco-MiniLM-L-6-v2):")
    print("-" * 80)
    ce_latencies = []
    for query in test_queries:
        _, metrics = cross_encoder.rerank(query, sample_docs, top_k=5)
        ce_latencies.append(metrics["total_time_ms"])
        print(f"  Query: '{query[:50]}...'")
        print(f"    Latency: {metrics['total_time_ms']:.2f}ms ({metrics['ms_per_document']:.2f}ms/doc)")

    print(f"\n  Average latency: {np.mean(ce_latencies):.2f}ms (±{np.std(ce_latencies):.2f}ms)")

    # Benchmark Cohere
    print("\nCohere Rerank (rerank-english-v2.0):")
    print("-" * 80)
    cohere_latencies = []
    for query in test_queries:
        _, metrics = cohere_reranker.rerank(query, sample_docs, top_k=5)
        cohere_latencies.append(metrics["total_time_ms"])
        print(f"  Query: '{query[:50]}...'")
        print(f"    Latency: {metrics['total_time_ms']:.2f}ms (API: {metrics['api_time_ms']:.2f}ms)")

    print(f"\n  Average latency: {np.mean(cohere_latencies):.2f}ms (±{np.std(cohere_latencies):.2f}ms)")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"Cross-Encoder avg latency: {np.mean(ce_latencies):.2f}ms")
    print(f"Cohere avg latency:        {np.mean(cohere_latencies):.2f}ms")
    print(f"Latency difference:        {abs(np.mean(ce_latencies) - np.mean(cohere_latencies)):.2f}ms")
    print(f"\nCross-Encoder: Self-hosted, no API costs, full control")
    print(f"Cohere:        Managed, ~$1-2/1K requests, multilingual support")


# Example usage combining everything
if __name__ == "__main__":
    # Run benchmark
    benchmark_rerankers()

    # Example: Integrate with existing RAG system
    # rag = YourRAGSystem()
    # reranker = CrossEncoderReranker()
    # rag_with_rerank = RAGSystemWithReranking(rag, reranker)
    # rag_with_rerank.print_comparison("Your query here")
```

Typical benchmark results reveal expected patterns. Cross-encoders running locally on modern CPUs achieve 50-100ms for 20 candidates, scaling linearly with candidate count. Cohere Rerank typically returns in 80-150ms including network latency, with less variance since Cohere's infrastructure handles load balancing transparently. For 100 queries per second, cross-encoders require ~10 CPU cores versus Cohere's API scaling automatically but costing ~$1000/month at that volume.

## Production Deployment Considerations

### Choosing Between Approaches

Select cross-encoder reranking when:
- You need deterministic latency without network dependencies
- Your query volume exceeds 100 queries/second (self-hosting becomes cost-effective)
- You require air-gapped deployment or data residency compliance
- You want to fine-tune models on domain-specific relevance judgments

Select Cohere Rerank when:
- You need multilingual support across 100+ languages
- Your query volume is under 100 queries/second
- You prefer managed infrastructure over operational complexity
- You want automatic model improvements without redeployment

### Latency Optimization Strategies

For production systems where every millisecond counts:

1. **Reduce candidate count**: Reranking 10 candidates instead of 20 cuts latency by 50% with minimal recall loss
2. **Use smaller cross-encoders**: TinyBERT-L-2 runs 2x faster than MiniLM-L-6 with 3-5% accuracy reduction
3. **Implement caching**: Cache reranking results for identical query-candidate sets using query+candidate_ids as cache key
4. **Batch processing**: When serving multiple simultaneous queries, batch their reranking requests for better GPU utilization
5. **Adaptive reranking**: Only rerank when bi-encoder confidence is low (e.g., top result score < 0.75)

These optimizations can reduce P99 latency from 200ms to under 100ms while maintaining the precision gains that justify reranking.

## Key Principles Demonstrated

**Principle 1: Two-Stage Retrieval Balances Recall and Precision**
The bi-encoder retrieves broadly (high recall), while the cross-encoder refines rankings (high precision). This division of labor exploits each model's strengths—bi-encoders for speed, cross-encoders for accuracy.

**Principle 2: Reranking Latency is Proportional to Candidate Count**
Unlike bi-encoder retrieval which scales logarithmically with corpus size via approximate nearest neighbor search, cross-encoder reranking scales linearly with candidates. Doubling candidates from 10 to 20 doubles latency, making candidate count the primary tuning parameter.

**Principle 3: Commercial APIs Trade Cost for Operational Simplicity**
Cohere Rerank eliminates infrastructure overhead (model hosting, scaling, updates) at ~$1-2 per 1000 requests. For early-stage products, this trade-off favors rapid development. For mature products with high query volumes, self-hosted solutions become cost-effective.

---

**END OF SECTION 6.6.3**
