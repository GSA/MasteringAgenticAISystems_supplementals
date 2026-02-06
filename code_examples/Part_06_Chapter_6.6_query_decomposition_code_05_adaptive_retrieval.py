"""
Adaptive Retrieval Implementation - Confidence-Based Decisions

Determines whether to retrieve or use parametric knowledge based on confidence.
"""

import re
from typing import Optional, Tuple, List, Dict, Any


async def generate_with_confidence(
    query: str,
    llm_client,
    include_confidence_prompt: bool = True
) -> Tuple[str, float]:
    """
    Generate answer and estimate confidence.

    Args:
        query: The user's question
        llm_client: OpenAI or compatible client
        include_confidence_prompt: Whether to ask for confidence rating

    Returns:
        (answer_text, confidence_score) where confidence is 0-1
    """
    base_prompt = f"Answer this question concisely: {query}"

    if include_confidence_prompt:
        prompt = f"""{base_prompt}

After your answer, on a new line, rate your confidence that your answer is correct:
- HIGH: You are certain the answer is correct based on well-established facts
- MEDIUM: You believe the answer is likely correct but there may be nuances
- LOW: You are uncertain or the question requires specific context you lack

Format: 'Confidence: HIGH/MEDIUM/LOW'"""
    else:
        prompt = base_prompt

    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer accurately and honestly assess your confidence."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )

    full_response = response.choices[0].message.content

    # Parse confidence if included in prompt
    confidence_score = 0.5  # Default: uncertain
    answer_text = full_response

    if include_confidence_prompt:
        # Look for "Confidence: HIGH/MEDIUM/LOW" pattern
        confidence_match = re.search(
            r'Confidence:\s*(HIGH|MEDIUM|LOW)',
            full_response,
            re.IGNORECASE
        )

        if confidence_match:
            confidence_level = confidence_match.group(1).upper()
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
            confidence_score = confidence_map.get(confidence_level, 0.5)

            # Remove confidence statement from answer
            answer_text = full_response[:confidence_match.start()].strip()

    return answer_text, confidence_score


async def adaptive_rag_query(
    user_query: str,
    retrieve_context_func,
    generate_answer_func,
    llm_client,
    confidence_threshold: float = 0.7,
    always_retrieve_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Adaptive RAG: retrieve only when confidence is low or query matches patterns.

    Args:
        user_query: The user's question
        retrieve_context_func: Retrieval function (async)
        generate_answer_func: Answer generation function (async)
        llm_client: OpenAI or compatible client
        confidence_threshold: If confidence < threshold, retrieve
        always_retrieve_patterns: Regex patterns that force retrieval

    Returns:
        Dict with answer and metadata
    """
    import time

    start_time = time.perf_counter()

    # Check if query matches patterns that always require retrieval
    force_retrieval = False
    if always_retrieve_patterns:
        for pattern in always_retrieve_patterns:
            if re.search(pattern, user_query, re.IGNORECASE):
                print(f"🎯 Query matches pattern '{pattern}' - forcing retrieval")
                force_retrieval = True
                break

    if not force_retrieval:
        # Step 1: Generate initial answer with confidence estimation
        print(f"\n📊 Estimating confidence for query: '{user_query[:60]}...'")
        initial_answer, confidence = await generate_with_confidence(
            user_query,
            llm_client,
            include_confidence_prompt=True
        )

        print(f"   Initial answer: {initial_answer[:80]}...")
        print(f"   Confidence score: {confidence:.2f}")

        # Step 2: Decide whether to retrieve
        if confidence >= confidence_threshold:
            print(f"✓ Confidence {confidence:.2f} >= {confidence_threshold} - using direct answer")
            total_time = (time.perf_counter() - start_time) * 1000

            return {
                "answer": initial_answer,
                "method": "direct_parametric",
                "confidence": confidence,
                "retrieved": False,
                "total_time_ms": total_time,
                "decision": f"High confidence ({confidence:.2f}) - no retrieval needed"
            }

    # Step 3: Confidence too low or forced - retrieve and generate
    print(f"🔍 {'Forced retrieval' if force_retrieval else f'Confidence {confidence:.2f} < {confidence_threshold}'} - retrieving context")

    retrieval_start = time.perf_counter()
    chunks, retrieval_time = await retrieve_context_func(
        user_query,
        top_k=5,
        alpha=0.7
    )

    if not chunks:
        print("⚠️  No context retrieved - falling back to parametric answer")
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "answer": initial_answer if not force_retrieval else "Unable to retrieve relevant context.",
            "method": "fallback_parametric",
            "confidence": confidence if not force_retrieval else 0.0,
            "retrieved": True,
            "retrieval_failed": True,
            "total_time_ms": total_time
        }

    # Step 4: Generate grounded answer with retrieved context
    print(f"✓ Retrieved {len(chunks)} chunks - generating grounded answer")

    generation_start = time.perf_counter()
    rag_answer, tokens_used, generation_time = await generate_answer_func(
        user_query,
        chunks
    )

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "answer": rag_answer,
        "method": "adaptive_rag_retrieved",
        "confidence": confidence if not force_retrieval else None,
        "retrieved": True,
        "chunks_count": len(chunks),
        "total_time_ms": total_time,
        "retrieval_time_ms": retrieval_time,
        "generation_time_ms": generation_time,
        "tokens_used": tokens_used,
        "decision": f"{'Forced retrieval' if force_retrieval else f'Low confidence ({confidence:.2f})'} - retrieved and grounded"
    }
