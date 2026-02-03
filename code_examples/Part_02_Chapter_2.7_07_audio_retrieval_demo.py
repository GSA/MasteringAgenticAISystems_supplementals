"""
Audio RAG System Query Demonstration
Shows end-to-end retrieval workflow with example queries
"""

from pathlib import Path

# Example usage: Index weekly engineering syncs
# (Note: AudioRAGSystem imported from Part_02_Chapter_2.7_06_audio_rag_system.py)
rag_system = None  # Initialize with AudioRAGSystem(model_size="base")

# Ingest three months of weekly meetings
meeting_files = [
    "meetings/eng_sync_2024_01_15.mp3",
    "meetings/eng_sync_2024_01_22.mp3",
    "meetings/eng_sync_2024_01_29.mp3",
]

# for meeting in meeting_files:
#     rag_system.ingest_audio(meeting, chunk_duration=300)

# Query 1: Performance optimization discussion
print("\n" + "="*60)
print("Query: What performance optimizations did we discuss for vision models?")
print("="*60)

# results = rag_system.search(
#     "vision model performance optimization inference speed",
#     top_k=2
# )

# for i, (chunk, score) in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print(rag_system.format_result(chunk, score))

# Example output:
# --- Result 1 ---
# Score: 0.847
# Source: eng_sync_2024_01_22.mp3
# Timestamp: 23:15 - 28:22
# Language: en
#
# Transcript:
# Moving on to performance requirements. We're targeting sub-second retrieval
# latency for text queries, but vision model inference is the bottleneck.
# NeVA 22B takes 800ms per image on A100 GPUs. We discussed three optimization
# approaches: first, TensorRT-LLM compilation reduces latency to 300ms. Second,
# batching multiple images per request improves throughput by 3x...

# Query 2: Architecture decisions
print("\n" + "="*60)
print("Query: What did we decide about multimodal RAG architecture?")
print("="*60)

# results = rag_system.search(
#     "multimodal RAG architecture decision grounding approach",
#     top_k=2
# )

# for i, (chunk, score) in enumerate(results, 1):
#     print(f"\n--- Result {i} ---")
#     print(rag_system.format_result(chunk, score))

# Example output:
# --- Result 1 ---
# Score: 0.821
# Source: eng_sync_2024_01_15.mp3
# Timestamp: 14:32 - 19:45
# Language: en
#
# Transcript:
# Today we're discussing the multimodal RAG implementation. We need to integrate
# vision models for chart understanding and audio processing for meeting transcripts.
# The architecture decision came down to three approaches. We chose Approach 2,
# grounding all modalities to text, because it leverages our existing text RAG
# infrastructure without requiring multimodal embedders or rerankers...
