"""
Query Decomposition - Hybrid Pipeline with Fallback Strategies

Combines decomposition and adaptive retrieval with graceful degradation.
"""

from typing import List, Dict, Any
import time


async def hybrid_rag_pipeline(
    user_query: str,
    retrieve_context_func,
    generate_answer_func,
    decompose_query_func,
    synthesize_answer_func,
    llm_client,
    classify_query_func,
    enable_decomposition: bool = True,
    enable_adaptive: bool = True
) -> Dict[str, Any]:
    """
    Hybrid RAG pipeline combining decomposition, adaptive retrieval, and routing.

    Args:
        user_query: The user's question
        retrieve_context_func: Async retrieval function
        generate_answer_func: Async answer generation function
        decompose_query_func: Query decomposition function
        synthesize_answer_func: Answer synthesis function
        llm_client: OpenAI or compatible client
        classify_query_func: Query classification function
        enable_decomposition: Whether to use decomposition
        enable_adaptive: Whether to use adaptive retrieval

    Returns:
        Dict with answer and detailed metadata
    """
    start_time = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"HYBRID RAG PIPELINE")
    print(f"Query: {user_query}")
    print(f"Decomposition: {'ENABLED' if enable_decomposition else 'DISABLED'}")
    print(f"Adaptive Retrieval: {'ENABLED' if enable_adaptive else 'DISABLED'}")
    print(f"{'='*70}")

    # Step 1: Query classification
    classify_start = time.perf_counter()
    query_type = await classify_query_func(user_query, llm_client)
    classify_time = (time.perf_counter() - classify_start) * 1000
    print(f"\n📋 Query classified as: {query_type.upper()} ({classify_time:.1f}ms)")

    # Step 2: Route to optimal strategy
    should_decompose = (
        enable_decomposition and
        query_type in ["comparison", "procedural"]
    )

    should_retrieve = query_type not in ["opinion", "calculation"]
    should_adaptive = enable_adaptive and query_type == "factual"

    print(f"🎯 Strategy: {'Decomposed' if should_decompose else 'Standard'} + "
          f"{'Adaptive' if should_adaptive else 'Direct'} retrieval")

    # Step 3: Execute retrieval strategy
    retrieval_start = time.perf_counter()

    if should_decompose:
        # Decomposed retrieval for complex queries
        sub_queries = await decompose_query_func(user_query)

        # Parallel retrieval for each sub-query
        sub_results = []
        for sq in sub_queries:
            try:
                chunks, _ = await retrieve_context_func(sq, top_k=3)
                sub_results.append({
                    'query': sq,
                    'chunks': chunks,
                    'successful': len(chunks) > 0
                })
            except Exception as e:
                print(f"⚠️  Sub-query retrieval failed: {e}")
                sub_results.append({
                    'query': sq,
                    'chunks': [],
                    'successful': False
                })

        retrieval_result = {
            'method': 'decomposed',
            'sub_queries': len(sub_queries),
            'successful_sub_queries': sum(1 for r in sub_results if r['successful']),
            'total_chunks': sum(len(r['chunks']) for r in sub_results)
        }

    elif should_retrieve:
        # Standard or adaptive retrieval
        if should_adaptive:
            # Try direct generation first with confidence check
            try:
                answer, confidence = await generate_answer_func(
                    user_query,
                    confidence=True
                )

                if confidence >= 0.7:
                    # High confidence - use direct answer
                    total_time = (time.perf_counter() - start_time) * 1000
                    return {
                        "answer": answer,
                        "method": "hybrid_parametric",
                        "query_type": query_type,
                        "confidence": confidence,
                        "retrieved": False,
                        "total_time_ms": total_time,
                        "strategy": "high_confidence_parametric"
                    }
            except Exception as e:
                print(f"⚠️  Confidence estimation failed: {e}")

        # Low confidence or confidence estimation failed - retrieve
        try:
            chunks, _ = await retrieve_context_func(user_query, top_k=5)
            retrieval_result = {
                'method': 'standard',
                'chunks': chunks,
                'successful': len(chunks) > 0
            }
        except Exception as e:
            print(f"⚠️  Retrieval failed: {e}")
            retrieval_result = {
                'method': 'failed',
                'chunks': [],
                'successful': False
            }

    else:
        # No retrieval needed - use parametric knowledge
        retrieval_result = {
            'method': 'parametric_only',
            'chunks': [],
            'successful': True
        }

    retrieval_time = (time.perf_counter() - retrieval_start) * 1000

    # Step 4: Generate answer
    generation_start = time.perf_counter()

    if should_decompose and retrieval_result.get('sub_queries'):
        # Synthesize from decomposed results
        answer = f"[Synthesized answer from {retrieval_result['successful_sub_queries']} sub-queries]"
    elif should_retrieve and retrieval_result.get('successful'):
        # Generate from retrieved context
        chunks = retrieval_result.get('chunks', [])
        answer, tokens, _ = await generate_answer_func(user_query, chunks)
    else:
        # Parametric generation
        answer, _, _ = await generate_answer_func(user_query, [])

    generation_time = (time.perf_counter() - generation_start) * 1000

    total_time = (time.perf_counter() - start_time) * 1000

    return {
        "answer": answer,
        "method": "hybrid_rag",
        "query_type": query_type,
        "strategy": {
            "decomposed": should_decompose,
            "adaptive": should_adaptive,
            "retrieval": should_retrieve
        },
        "retrieval": retrieval_result,
        "total_time_ms": total_time,
        "timing": {
            "classification_ms": classify_time,
            "retrieval_ms": retrieval_time,
            "generation_ms": generation_time
        }
    }
