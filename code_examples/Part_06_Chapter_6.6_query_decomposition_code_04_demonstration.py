"""
Query Decomposition Implementation - Complete Pipeline Demonstration

End-to-end decomposed RAG system with timing analysis.
"""

import time
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SubQuery:
    """Represents a decomposed sub-query with its retrieved context."""
    query_text: str
    retrieved_chunks: List[Dict[str, Any]] = None
    retrieval_successful: bool = False


async def decomposed_rag_query(
    user_query: str,
    retrieve_context_func,
    decompose_query_func,
    synthesize_answer_func,
    top_k_per_subquery: int = 3,
    enable_decomposition: bool = True
) -> Dict[str, Any]:
    """
    Complete RAG pipeline with optional query decomposition.

    Args:
        user_query: The user's question
        retrieve_context_func: Retrieval function (async)
        decompose_query_func: Query decomposition function
        synthesize_answer_func: Answer synthesis function
        top_k_per_subquery: Chunks to retrieve per sub-query
        enable_decomposition: If False, use standard single-query RAG

    Returns:
        Dict with answer and metadata
    """
    start_time = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"Processing query: '{user_query}'")
    print(f"Decomposition: {'ENABLED' if enable_decomposition else 'DISABLED'}")
    print(f"{'='*70}")

    if not enable_decomposition:
        # Standard RAG: single query, single retrieval
        chunks, retrieval_time = await retrieve_context_func(
            user_query,
            top_k=top_k_per_subquery * 2,
            alpha=0.7
        )

        print(f"✓ Standard retrieval: {len(chunks)} chunks in {retrieval_time:.2f}ms")
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "answer": "[Standard RAG answer would be generated here]",
            "method": "standard_rag",
            "total_time_ms": total_time,
            "chunks_retrieved": len(chunks)
        }

    # Decomposed RAG pipeline
    decomposition_start = time.perf_counter()
    sub_queries = await decompose_query_func(user_query)
    decomposition_time = (time.perf_counter() - decomposition_start) * 1000

    # Parallel retrieval for all sub-queries
    retrieval_start = time.perf_counter()
    sub_results = await retrieve_for_all_subqueries(
        sub_queries,
        retrieve_context_func,
        top_k_per_subquery
    )
    retrieval_time = (time.perf_counter() - retrieval_start) * 1000

    synthesis_start = time.perf_counter()
    result = await synthesize_answer_func(
        user_query,
        sub_results
    )
    synthesis_time = (time.perf_counter() - synthesis_start) * 1000

    total_time = (time.perf_counter() - start_time) * 1000

    result.update({
        "method": "decomposed_rag",
        "total_time_ms": total_time,
        "decomposition_time_ms": decomposition_time,
        "retrieval_time_ms": retrieval_time,
        "synthesis_time_ms": synthesis_time,
        "timing_breakdown": {
            "decomposition": f"{decomposition_time:.2f}ms",
            "parallel_retrieval": f"{retrieval_time:.2f}ms",
            "synthesis": f"{synthesis_time:.2f}ms"
        }
    })

    print(f"\n{'='*70}")
    print(f"✓ Decomposed RAG complete in {total_time:.2f}ms")
    print(f"  - Decomposition: {decomposition_time:.2f}ms")
    print(f"  - Retrieval (parallel): {retrieval_time:.2f}ms")
    print(f"  - Synthesis: {synthesis_time:.2f}ms")
    print(f"{'='*70}")

    return result


async def retrieve_for_all_subqueries(
    sub_queries: List[str],
    retrieve_context_func,
    top_k: int
) -> List[SubQuery]:
    """Helper to retrieve for all sub-queries in parallel."""
    import asyncio

    async def retrieve_one(sq: str) -> SubQuery:
        try:
            chunks, _ = await retrieve_context_func(sq, top_k=top_k)
            return SubQuery(
                query_text=sq,
                retrieved_chunks=chunks,
                retrieval_successful=len(chunks) > 0
            )
        except Exception:
            return SubQuery(
                query_text=sq,
                retrieved_chunks=[],
                retrieval_successful=False
            )

    return await asyncio.gather(*[retrieve_one(sq) for sq in sub_queries])
