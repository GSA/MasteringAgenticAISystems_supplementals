"""
Query Decomposition Implementation - Core Decomposition

Demonstrates: LLM-powered decomposition of complex queries into sub-queries.
"""

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class SubQuery:
    """Represents a decomposed sub-query with its retrieved context."""
    query_text: str
    retrieved_chunks: List[Dict[str, Any]] = None
    retrieval_successful: bool = False


async def decompose_query(original_query: str, llm_client) -> List[str]:
    """
    Decompose complex query into focused sub-queries using LLM.

    Args:
        original_query: The user's question
        llm_client: OpenAI or compatible client

    Returns:
        List of sub-query strings
    """
    decomposition_prompt = f"""Analyze this user question and identify distinct information needs.
For each distinct need, generate a focused search query that would retrieve relevant information.

User question: {original_query}

Requirements:
- Generate 1-4 sub-queries (only what's necessary)
- Each sub-query should target ONE specific piece of information
- Use concrete search terms, not abstract concepts
- If the question is simple and focused, return just the original query

Format your response as a numbered list:
1. [First sub-query]
2. [Second sub-query]
etc.
"""

    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing information needs and formulating precise search queries."
            },
            {"role": "user", "content": decomposition_prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    decomposition_text = response.choices[0].message.content

    # Parse numbered list into sub-queries
    sub_queries = []
    for line in decomposition_text.strip().split('\n'):
        line = line.strip()
        # Match patterns like "1. query text" or "1) query text"
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove number prefix and clean
            query = line.split('.', 1)[-1].split(')', 1)[-1].strip()
            if query and not query.startswith('['):
                sub_queries.append(query)

    print(f"\n📋 Decomposed into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"   {i}. {sq}")

    return sub_queries


async def retrieve_for_subquery(
    sub_query: str,
    retrieve_context_func,
    top_k: int = 3
) -> SubQuery:
    """
    Retrieve context for a single sub-query.

    Args:
        sub_query: The focused search query
        retrieve_context_func: Async function that performs actual retrieval
        top_k: Number of chunks to retrieve per sub-query

    Returns:
        SubQuery with retrieved context
    """
    try:
        chunks, retrieval_time = await retrieve_context_func(
            sub_query,
            top_k=top_k,
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
