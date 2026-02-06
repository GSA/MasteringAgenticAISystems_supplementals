"""
Query Decomposition Implementation - Result Synthesis

Synthesizes comprehensive answers from multiple sub-query contexts.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class SubQuery:
    """Represents a decomposed sub-query with its retrieved context."""
    query_text: str
    retrieved_chunks: List[Dict[str, Any]] = None
    retrieval_successful: bool = False


async def synthesize_decomposed_answer(
    original_query: str,
    sub_query_results: List[SubQuery],
    generate_answer_func,
    llm_client
) -> Dict[str, Any]:
    """
    Synthesize answer from multiple sub-query contexts.

    Args:
        original_query: The user's original question
        sub_query_results: Retrieved contexts for each sub-query
        generate_answer_func: Async function that generates answers
        llm_client: OpenAI or compatible client

    Returns:
        Dict with synthesized answer and metadata
    """
    # Build structured context with sub-query attribution
    context_sections = []
    all_chunks = []

    for i, sub_result in enumerate(sub_query_results, 1):
        if not sub_result.retrieval_successful or not sub_result.retrieved_chunks:
            context_sections.append(
                f"\n--- Context for Sub-Question {i}: {sub_result.query_text} ---\n"
                f"[No relevant information found]"
            )
            continue

        chunk_texts = []
        for j, chunk in enumerate(sub_result.retrieved_chunks):
            chunk_ref = f"{i}.{j+1}"
            chunk_text = chunk.get('content', '')
            chunk_texts.append(f"[{chunk_ref}] {chunk_text}")
            all_chunks.append({
                'content': chunk_text,
                'sub_query': sub_result.query_text,
                'reference': chunk_ref,
                **chunk
            })

        context_sections.append(
            f"\n--- Context for Sub-Question {i}: {sub_result.query_text} ---\n" +
            "\n\n".join(chunk_texts)
        )

    full_context = "\n".join(context_sections)

    synthesis_prompt = f"""Answer the user's question using the provided context from multiple focused searches.
Each section contains context retrieved for a specific sub-question.

User's Question: {original_query}

Retrieved Context:
{full_context}

Instructions:
- Synthesize a comprehensive answer that addresses all aspects of the question
- Cite sources using the reference format [1.1], [2.3], etc.
- If certain sub-questions lack context, acknowledge what information is unavailable
- Provide a coherent narrative rather than separate answers per sub-question

Answer:"""

    # Generate synthesized answer
    response = await llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at synthesizing information from multiple sources into coherent, comprehensive answers."
            },
            {"role": "user", "content": synthesis_prompt}
        ],
        temperature=0.3,
        max_tokens=700
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sub_queries": [sq.query_text for sq in sub_query_results],
        "successful_retrievals": sum(1 for sq in sub_query_results if sq.retrieval_successful),
        "total_chunks_retrieved": len(all_chunks),
        "chunks_by_subquery": {
            sq.query_text: len(sq.retrieved_chunks) if sq.retrieved_chunks else 0
            for sq in sub_query_results
        },
        "tokens_used": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
