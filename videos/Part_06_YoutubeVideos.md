# Part 06: RAG and Data Pipeline Implementation - Video and Learning Resources

> **Catalog note:** A hyperlink means a resource is currently assigned; it does not by itself guarantee that every title, runtime, or scope claim has been independently verified. Entries without a hyperlink are unverified discovery candidates. See [README.md](README.md) and [video_resource_status.csv](video_resource_status.csv) for status and provenance.

## Table of Contents

- [Chapter 6.1 - RAG Chunking](#chapter-61---rag-chunking)
- [Chapter 6.2A - Vector Database Selection](#chapter-62a---vector-database-selection)
- [Chapter 6.2B - Production Vector Database Deployment](#chapter-62b---production-vector-database-deployment)
- [Chapter 6.3A - ETL Fundamentals](#chapter-63a---etl-fundamentals)
- [Chapter 6.3B - ETL Load and Integration](#chapter-63b---etl-load-and-integration)
- [Chapter 6.3C - ETL Practice](#chapter-63c---etl-practice)
- [Chapter 6.4 - Data Quality](#chapter-64---data-quality)
- [Chapter 6.4B - Data Quality Practice](#chapter-64b---data-quality-practice)
- [Chapter 6.5 - Production RAG](#chapter-65---production-rag)
- [Chapter 6.5B - Production RAG Practice](#chapter-65b---production-rag-practice)
- [Chapter 6.6A - Reranking Implementation](#chapter-66a---reranking-implementation)
- [Chapter 6.6 - Query Decomposition](#chapter-66---query-decomposition)

---

<a name="chapter-61---rag-chunking"></a>
## Chapter 6.1 - RAG Chunking

**Topics:** Embeddings, Vector Search, Hybrid Retrieval, GPU Acceleration, Matryoshka Representation Learning

### Learn RAG from Scratch
- [https://www.youtube.com/watch?v=sVcwVQRHIc8](https://www.youtube.com/watch?v=sVcwVQRHIc8) ~3 hours
- Covers: Comprehensive RAG with embeddings, chunking, query translation

### 5 Levels of Text Splitting for RAG
- [https://www.youtube.com/watch?v=8OJC21T2SL4](https://www.youtube.com/watch?v=8OJC21T2SL4) ~45 minutes
- Covers: Essential chunking strategies

### RAG+Langchain Python Project
- [https://www.youtube.com/watch?v=tcqEUSNCn8I](https://www.youtube.com/watch?v=tcqEUSNCn8I) Variable
- Covers: Hands-on RAG implementation

### Python RAG Tutorial with Local LLMs
- [https://www.youtube.com/watch?v=2TJxpyO3ei4](https://www.youtube.com/watch?v=2TJxpyO3ei4) Variable
- Covers: Local RAG deployment

### RAG vs. Agentic AI - IBM Technology
- [https://www.youtube.com/watch?v=fB2JQXEH_94](https://www.youtube.com/watch?v=fB2JQXEH_94)
- Covers: How retrieval-augmented generation differs from agentic systems and how agentic behavior can extend retrieval workflows

---

<a name="chapter-62a---vector-database-selection"></a>
## Chapter 6.2A - Vector Database Selection

**Topics:** Vector databases, HNSW algorithm, ANN search, database comparison

### Vector Databases Simply Explained
- [https://www.youtube.com/watch?v=dN0lsF2cvm4](https://www.youtube.com/watch?v=dN0lsF2cvm4) ~10-15 minutes
- Covers: Overview and database comparison

### Milvus: Cloud-Native Vector Database
- [https://www.youtube.com/watch?v=75G513Y9rkU](https://www.youtube.com/watch?v=75G513Y9rkU) ~30-45 minutes
- Covers: Milvus architecture and capabilities

### Recommendation System with Weaviate
- [https://www.youtube.com/watch?v=SF1ZlRjVsxw](https://www.youtube.com/watch?v=SF1ZlRjVsxw) ~20-40 minutes
- Covers: Weaviate implementation

### Billion-scale ANN Search
- [https://www.youtube.com/watch?v=SKrHs03i08Q](https://www.youtube.com/watch?v=SKrHs03i08Q) ~60+ minutes
- Covers: Academic depth on algorithms

### Hierarchical Navigable Small Worlds (HNSW) Explained
- [https://www.youtube.com/watch?v=77QH0Y2PYKg&t=289s](https://www.youtube.com/watch?v=77QH0Y2PYKg&t=289s)
- Covers: how the hierarchical navigable small worlds (HNSW) algorithm works when we want to index vector databases, and how it can speed up the process of finding the most similar vectors in a database to a given query.

---

<a name="chapter-62b---production-vector-database-deployment"></a>
## Chapter 6.2B - Production Vector Database Deployment

**Topics:** Docker Compose, Weaviate Production, HNSW, Hybrid Search, Monitoring

### Weaviate Tutorial
- [https://www.youtube.com/watch?v=SF1ZlRjVsxw](https://www.youtube.com/watch?v=SF1ZlRjVsxw) Variable
- Covers: Production Weaviate deployment

---

<a name="chapter-63a---etl-fundamentals"></a>
## Chapter 6.3A - ETL Fundamentals

**Topics:** Extract, Transform, Load, Data Pipeline Architecture

Note: Basic ETL concepts are well covered across multiple tutorial channels. See official documentation for platform-specific details.

### Data Engineering Pipeline Fundamentals
- [https://www.youtube.com/watch?v=uqRRjcsUGgk](https://www.youtube.com/watch?v=uqRRjcsUGgk) Variable
- Covers: Pipeline fundamentals

---

<a name="chapter-63b---etl-load-and-integration"></a>
## Chapter 6.3B - ETL Load and Integration

**Topics:** Vector DB Integration, Batch Insertion, Pipeline Orchestration

Note: Comprehensive tutorials available from James Briggs, freeCodeCamp, and NVIDIA on ETL patterns and vector database integration.

---

<a name="chapter-63c---etl-practice"></a>
## Chapter 6.3C - ETL Practice

**Topics:** Incremental Updates, Streaming ETL, Custom Chunking

### LangChain RAG Build and Deploy
- [https://www.youtube.com/watch?v=EhlPDL4QrWY](https://www.youtube.com/watch?v=EhlPDL4QrWY) Variable
- Covers: RAG deployment patterns

### Azure Databricks ETL
- [https://www.youtube.com/watch?v=pc8Kv-lRD8k](https://www.youtube.com/watch?v=pc8Kv-lRD8k) Variable
- Covers: Cloud ETL pipelines

### Uber Data Engineering Project
- [https://www.youtube.com/watch?v=WpQECq5Hx9g](https://www.youtube.com/watch?v=WpQECq5Hx9g) Variable
- Covers: Production data engineering

---

<a name="chapter-64---data-quality"></a>
## Chapter 6.4 - Data Quality

**Topics:** Data Quality Assessment, Validation, Testing

Note: Quality assessment frameworks well covered. See tutorials on schema validation and testing patterns from data engineering courses.

---

<a name="chapter-64b---data-quality-practice"></a>
## Chapter 6.4B - Data Quality Practice

**Topics:** Pydantic Validation, Data Testing, Monitoring

### Why You Should Use Pydantic in 2024 - Tutorial - ArjanCodes
- [https://www.youtube.com/watch?v=502XOB0u8OY](https://www.youtube.com/watch?v=502XOB0u8OY)
- Covers: Pydantic models, Python type hints, parsing and validation, constraints, and structured application data

### Data Engineering for Beginners
- [https://www.youtube.com/watch?v=PHsC_t0j1dU](https://www.youtube.com/watch?v=PHsC_t0j1dU) Variable
- Covers: Data quality practices

---

<a name="chapter-65---production-rag"></a>
## Chapter 6.5 - Production RAG

**Topics:** System Design, Scalability, Observability, Performance

### Learn RAG from Scratch
- [https://www.youtube.com/watch?v=sVcwVQRHIc8](https://www.youtube.com/watch?v=sVcwVQRHIc8) ~2.5 hours
- Covers: RAG architecture and design patterns

---

<a name="chapter-65b---production-rag-practice"></a>
## Chapter 6.5B - Production RAG Practice

**Topics:** Real-World Implementation, Evaluation, Optimization

Note: Practical RAG implementation patterns available from Pinecone Learning Hub, DeepLearning.AI, and Grafana tutorials. See official documentation for production monitoring.

---

<a name="chapter-66a---reranking-implementation"></a>
## Chapter 6.6A - Reranking Implementation

**Topics:** Cross-encoder, Two-stage Retrieval, Transformer Attention

### Attention in Transformers
- [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc) Variable
- Covers: Transformer attention for reranking

---

<a name="chapter-66---query-decomposition"></a>
## Chapter 6.6 - Query Decomposition

**Topics:** Query Decomposition, Adaptive Retrieval, Query Routing

### Break Down Complex Questions with Query Decomposition
- [https://www.youtube.com/watch?v=kR4pVvBnOII](https://www.youtube.com/watch?v=kR4pVvBnOII)
- Covers: query decomposition.

### RAG from Scratch: Part 10 (Routing)
- [https://www.youtube.com/watch?v=pfpIndq7Fi8](https://www.youtube.com/watch?v=pfpIndq7Fi8)
- Covers: Logical and semantic routing for directing a query to the appropriate data source or processing chain
