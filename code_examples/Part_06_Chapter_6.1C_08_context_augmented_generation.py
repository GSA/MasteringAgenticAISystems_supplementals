from typing import List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI

client = OpenAI()

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str
    source_doc: str
    chunk_index: int
    embedding: List[float] = None
    metadata: dict = None

class SimpleRAGSystem:
    """RAG system with context-augmented generation."""

    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context and generate answer.

        Steps:
        1. Retrieve top-k relevant chunks
        2. Build augmented prompt with context
        3. Generate answer with LLM
        4. Return answer with sources
        """
        # Step 1: Retrieve
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        # Step 2: Build prompt
        context = "\n\n".join([
            f"[{i+1}] {chunk.text}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        prompt = f"""Answer the question using ONLY the provided context. Include citations using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

        # Step 3: Generate answer
        print("\nGenerating answer...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite sources using [1], [2] format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                {
                    "text": chunk.text[:200] + "...",
                    "source_doc": chunk.source_doc,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in retrieved_chunks
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Placeholder for retrieve method."""
        pass
