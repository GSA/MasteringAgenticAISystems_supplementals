"""
Adaptive Retrieval Implementation - Query Classification and Routing

Routes different query types to optimal retrieval strategies.
"""

from typing import List, Dict, Any


async def classify_query_type(query: str, llm_client) -> str:
    """
    Classify query to determine optimal retrieval strategy.

    Args:
        query: The user's question
        llm_client: OpenAI or compatible client

    Returns:
        Query type: 'factual', 'comparison', 'procedural', 'opinion', or 'calculation'
    """
    classification_prompt = f"""Classify this question into ONE category:

- factual: Asking for specific facts, definitions, or data (e.g., "What is X?", "When did Y happen?")
- comparison: Comparing multiple items or options (e.g., "X vs Y", "Which is better?")
- procedural: Asking how to do something (e.g., "How do I...?", "Steps to...")
- opinion: Asking for subjective judgment or analysis
- calculation: Mathematical or logical computation

Question: {query}

Respond with ONLY the category name."""

    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at classifying question types."},
            {"role": "user", "content": classification_prompt}
        ],
        temperature=0.0,
        max_tokens=10
    )

    query_type = response.choices[0].message.content.strip().lower()
    valid_types = {"factual", "comparison", "procedural", "opinion", "calculation"}

    return query_type if query_type in valid_types else "factual"


async def routed_adaptive_rag(
    user_query: str,
    retrieve_context_func,
    generate_answer_func,
    llm_client,
    decomposed_rag_func
) -> Dict[str, Any]:
    """
    Route queries to optimal strategy based on type classification.

    Args:
        user_query: The user's question
        retrieve_context_func: Async retrieval function
        generate_answer_func: Async answer generation function
        llm_client: OpenAI or compatible client
        decomposed_rag_func: Function for decomposed RAG

    Returns:
        Dict with answer and metadata
    """
    query_type = await classify_query_type(user_query, llm_client)
    print(f"📋 Query classified as: {query_type.upper()}")

    # Define routing logic based on query type
    routing_rules = {
        "calculation": {
            "retrieve": False,
            "reason": "LLM can perform calculations from parametric knowledge"
        },
        "opinion": {
            "retrieve": False,
            "reason": "Subjective questions don't require external knowledge"
        },
        "factual": {
            "retrieve": True,
            "reason": "Factual queries benefit from authoritative sources"
        },
        "comparison": {
            "retrieve": True,
            "decompose": True,
            "reason": "Comparisons require comprehensive context for each item"
        },
        "procedural": {
            "retrieve": True,
            "top_k": 8,
            "reason": "Step-by-step procedures benefit from detailed documentation"
        }
    }

    route = routing_rules.get(query_type, {"retrieve": True})
    print(f"🎯 Routing decision: {route.get('reason', 'Default retrieval')}")

    if not route.get("retrieve", True):
        # Direct parametric generation
        answer, confidence = await generate_answer_func(user_query, confidence=True)
        return {
            "answer": answer,
            "method": "routed_parametric",
            "query_type": query_type,
            "retrieved": False,
            "routing_reason": route.get("reason")
        }

    # Retrieve with type-specific configuration
    top_k = route.get("top_k", 5)
    should_decompose = route.get("decompose", False)

    if should_decompose:
        # Use decomposed retrieval for comparisons
        return await decomposed_rag_func(
            user_query,
            retrieve_context_func,
            top_k_per_subquery=3,
            enable_decomposition=True
        )
    else:
        # Standard retrieval
        chunks, _ = await retrieve_context_func(user_query, top_k=top_k)
        answer, tokens_used, _ = await generate_answer_func(user_query, chunks)

        return {
            "answer": answer,
            "method": "routed_rag",
            "query_type": query_type,
            "retrieved": True,
            "chunks_count": len(chunks),
            "tokens_used": tokens_used,
            "routing_reason": route.get("reason")
        }
