# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_rag_system():
    """Demonstrate complete RAG system."""
    print("="*70)
    print("RAG System Demo: Technical Documentation Q&A")
    print("="*70)

    # Initialize RAG system
    rag = SimpleRAGSystem(chunk_size=200, chunk_overlap=20)

    # Sample documents (NVIDIA GPU documentation excerpts)
    documents = {
        "H100_specs": """
        The NVIDIA H100 Tensor Core GPU delivers exceptional performance for AI and HPC workloads.
        With 80GB of HBM3 memory and 3TB/s memory bandwidth, H100 provides 3x faster training
        compared to A100. The fourth-generation Tensor Cores enable mixed-precision computing
        with FP64, TF32, FP16, BF16, FP8, and INT8 precision. H100 features the new Transformer
        Engine that automatically selects optimal precision for transformer models, accelerating
        large language model training by up to 6x. The GPU includes 132 Streaming Multiprocessors
        with 16,896 CUDA cores. H100 supports NVLink 4.0 with 900GB/s bandwidth for multi-GPU scaling.
        """,

        "NIM_overview": """
        NVIDIA NIM (NVIDIA Inference Microservices) provides optimized inference containers
        for deploying AI models in production. NIMs include pre-built containers with TensorRT-LLM,
        Triton Inference Server, and model-specific optimizations. Each NIM supports standard APIs
        (OpenAI-compatible for LLMs) making integration simple. NIMs achieve up to 5x higher throughput
        compared to unoptimized deployments. Key features include automatic batching, multi-GPU support,
        and INT4/INT8/FP8 quantization. NIMs are available through NVIDIA NGC catalog and can be deployed
        on-premises, in cloud (AWS, Azure, GCP), or at the edge.
        """,

        "RAG_best_practices": """
        For production RAG systems, chunking strategy significantly impacts retrieval quality.
        Recommended chunk size is 512-1024 tokens with 10-15% overlap. Use semantic chunking
        for narrative documents and recursive chunking for structured documentation. Always
        attach metadata (source, section, date) to chunks for filtering. For embeddings,
        NVIDIA NV-Embed-v2 supports 32,768-token context windows, ideal for long documents.
        Hybrid search combining vector similarity and BM25 keyword matching improves accuracy
        by 15-25% over pure vector search. Implement reranking with cross-encoders to boost
        precision@5 by 10-30%. Monitor retrieval metrics: Recall@10 should exceed 80% for
        production systems.
        """
    }

    # Ingest documents
    for doc_id, content in documents.items():
        rag.ingest_document(content, doc_id)

    print("\n" + "="*70)
    print("Knowledge base ready! Processing queries...")
    print("="*70)

    # Test queries
    queries = [
        "What is the memory bandwidth of H100?",
        "How does NVIDIA NIM improve inference performance?",
        "What chunk size is recommended for RAG systems?"
    ]

    for query in queries:
        print("\n" + "="*70)
        result = rag.generate_answer(query, top_k=3)

        print(f"\nQUESTION: {query}")
        print(f"\nANSWER: {result['answer']}")
        print(f"\nSOURCES:")
        for i, source in enumerate(result['sources']):
            print(f"  [{i+1}] {source['source_doc']} (chunk {source['chunk_index']})")
            print(f"      Preview: {source['text']}")
        print(f"\nTOKENS USED: {result['usage']['total_tokens']} "
              f"(prompt: {result['usage']['prompt_tokens']}, "
              f"completion: {result['usage']['completion_tokens']})")


if __name__ == "__main__":
    example_rag_system()
