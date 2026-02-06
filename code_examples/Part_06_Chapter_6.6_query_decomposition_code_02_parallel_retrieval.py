"""
Query Decomposition Implementation - Parallel Retrieval

Orchestrates concurrent retrieval for multiple sub-queries.
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SubQuery:
    """Represents a decomposed sub-query with its retrieved context."""
    query_text: str
    retrieved_chunks: List[Dict[str, Any]] = None
    retrieval_successful: bool = False


async def parallel_subquery_retrieval(
    sub_queries: List[str],
    retrieve_context_func,
    top_k_per_query: int = 3
) -> List[SubQuery]:
    """
    Retrieve context for all sub-queries in parallel.

    Args:
        sub_queries: List of decomposed sub-queries
        retrieve_context_func: Async retrieval function
        top_k_per_query: Chunks to retrieve per sub-query

    Returns:
        List of SubQuery objects with retrieved context
    """
    async def retrieve_for_subquery(sub_query: str) -> SubQuery:
        """Retrieve context for a single sub-query."""
        try:
            chunks, retrieval_time = await retrieve_context_func(
                sub_query,
                top_k=top_k_per_query,
                alpha=0.7
            )

            return SubQuery(
                query_text=sub_query,
                retrieved_chunks=chunks,
                retrieval_successful=len(chunks) > 0
            )

        except Exception as e:
            print(f"⚠️  Retrieval failed for sub-query '{sub_query[:50]}...': {e}")
            return SubQuery(
                query_text=sub_query,
                retrieved_chunks=[],
                retrieval_successful=False
            )

    # Create retrieval tasks for all sub-queries
    retrieval_tasks = [
        retrieve_for_subquery(sq)
        for sq in sub_queries
    ]

    # Execute all retrievals concurrently
    results = await asyncio.gather(*retrieval_tasks)

    successful_retrievals = sum(1 for r in results if r.retrieval_successful)
    print(f"\n✓ Retrieved context for {successful_retrievals}/{len(results)} sub-queries")

    return results
