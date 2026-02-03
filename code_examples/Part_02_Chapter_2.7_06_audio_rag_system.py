"""
Complete Audio RAG System with Embedding and Retrieval
Integrates time-indexed chunks with vector database storage
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

class AudioRAGSystem:
    """
    Complete audio RAG system with embedding and retrieval
    """

    def __init__(self, model_size: str = "base"):
        # Import WhisperAudioProcessor from previous example
        from Part_02_Chapter_2_7_05_whisper_audio_processor import WhisperAudioProcessor

        self.audio_processor = WhisperAudioProcessor(model_size=model_size)

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("Embedding model loaded")

        # In-memory vector storage (production would use Milvus/Pinecone)
        self.chunks = []
        self.embeddings = []

    def ingest_audio(self, audio_path: str, chunk_duration: int = 300):
        """
        Process and index audio file for retrieval

        Args:
            audio_path: Path to audio file
            chunk_duration: Target chunk size in seconds
        """
        # Create time-indexed chunks
        chunks = self.audio_processor.create_time_indexed_chunks(
            audio_path,
            chunk_duration=chunk_duration
        )

        # Embed each chunk
        print(f"Embedding {len(chunks)} chunks...")
        for chunk in chunks:
            embedding = self.embedder.encode(
                chunk["text"],
                convert_to_numpy=True,
                normalize_embeddings=True  # Cosine similarity optimization
            )

            self.chunks.append(chunk)
            self.embeddings.append(embedding)

        print(f"Ingestion complete: {len(self.chunks)} total chunks indexed")

    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Semantic search over audio chunks

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples sorted by relevance
        """
        if not self.chunks:
            return []

        # Embed query
        query_embedding = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compute similarities (cosine similarity via dot product of normalized vectors)
        embeddings_matrix = np.array(self.embeddings)
        similarities = embeddings_matrix @ query_embedding

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Format results
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def format_result(self, chunk: Dict, score: float) -> str:
        """Format search result for display"""
        minutes = int(chunk["start_time"] // 60)
        seconds = int(chunk["start_time"] % 60)

        result = f"""
Score: {score:.3f}
Source: {Path(chunk["audio_source"]).name}
Timestamp: {minutes:02d}:{seconds:02d} - {int(chunk["end_time"]//60):02d}:{int(chunk["end_time"]%60):02d}
Language: {chunk["metadata"]["language"]}

Transcript:
{chunk["text"][:300]}{"..." if len(chunk["text"]) > 300 else ""}
        """
        return result
