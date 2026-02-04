# Part 08: Production Reliability and Success Metrics - YouTube Video Resources

## Table of Contents

- [Chapter 8.1: Latency Fundamentals](#chapter-81---latency-fundamentals)
- [Chapter 8.2A: Error Taxonomy and SLOs](#chapter-82a---error-taxonomy-and-slos)
- [Chapter 8.2B: Circuit Breakers and NeMo Guardrails](#chapter-82b---circuit-breakers-and-nemo-guardrails)
- [Chapter 8.3: Token Economics](#chapter-83---token-economics)
- [Chapter 8.4: Success Metrics](#chapter-84---success-metrics)

---

<a name="chapter-81---latency-fundamentals"></a>

## Chapter 8.1: Latency Fundamentals

**Topics:** Distributed Tracing, OpenTelemetry, Percentile Metrics (P50, P95, P99), GPU Performance, TTFT, Caching, SLO Design, Load Testing, Tail Latency

### CNCF Conference Talk: Distributed Tracing with Jaeger and OpenTelemetry
- ~33 minutes
- Covers: Parent-child span hierarchies, bottleneck identification, distributed systems tracing

Note: Additional resources well covered by DevOps and observability tutorial channels. See recommended YouTube searches: OpenTelemetry tutorial, Jaeger distributed tracing, Percentile metrics and SLOs, GPU Performance Monitoring with NVIDIA DCGM, Time to First Token (TTFT) optimization, Caching Strategies (Redis/LRU), SLO Design and Management, Load Testing tutorials, Tail Latency Optimization, GPU Inference Optimization, and Prometheus/Grafana Monitoring.

---

<a name="chapter-82a---error-taxonomy-and-slos"></a>

## Chapter 8.2A: Error Taxonomy and SLOs

**Topics:** Error Taxonomy (Planning/Execution/Verification), SLOs, Error Budgets, Burn Rate, Distributed Tracing, Deadlocks, Topological Sort, Concurrency Control

### SLO Error Budget Burn Rate Based Alerting
- [https://www.youtube.com/watch?v=EEHKiQimv_Q](https://www.youtube.com/watch?v=EEHKiQimv_Q) ~20 minutes
- Dynatrace Tips & Tricks #15
- Covers: Error budgets, burn rate alerting, SLO management

### Index Concurrency Control
- [https://www.youtube.com/watch?v=x5tqzyf0zrk](https://www.youtube.com/watch?v=x5tqzyf0zrk) ~30 minutes
- CMU 15-445 Database Systems
- Covers: Latch-based concurrency, B+Tree latching, concurrent access patterns

### MIT 6.824 Distributed Systems
- Search "MIT 6.824 Spring 2020"
- Covers: Distributed system fundamentals and coordination

### Jenny's Lectures - Deadlock Series
- Search "Jenny's Lectures CS/IT deadlock"
- Covers: Deadlock prevention and detection

### Topological Sort Algorithms
- Tushar Roy
- Covers: Dependency ordering and DAG traversal

### Circuit Breaker Patterns
- Search "circuit breaker pattern tutorial"
- Covers: State machines and failure prevention

---

<a name="chapter-82b---circuit-breakers-and-nemo-guardrails"></a>

## Chapter 8.2B: Circuit Breakers and NeMo Guardrails

**Topics:** Circuit Breaker Patterns, Resilience4j, Distributed System Resilience, Microservices, SRE Practices, NeMo Guardrails

### Battle of the Circuit Breakers
- GOTO Conference
- Covers: Resilience4J vs Istio comparison and implementations

### freeCodeCamp - Microservices Courses
- Covers: Comprehensive circuit breaker patterns

### Martin Kleppmann's Distributed Systems Lectures
- ~7 hours
- Cambridge course
- Covers: Distributed system resilience and failure handling

### TechWorld with Nana - Prometheus/Grafana Monitoring
- Covers: DevOps patterns and observability

### Google SRE Documentation
- Covers: Error budget and SLO guidance

### Resilience4j Documentation
- Covers: Circuit breaker implementation reference

### NeMo Guardrails Tutorial
- Covers: Guardrails implementation and deployment

---

<a name="chapter-83---token-economics"></a>

## Chapter 8.3: Token Economics

**Topics:** Tokenization, API Usage and Costs, RAG Optimization, System-Level Optimization, Monitoring, Prompt Caching, Model Routing

### Neural Networks: Zero to Hero (Tokenization)
- Andrej Karpathy
- Covers: Tokenizers, BPE algorithm, GPT tokenizer implementation

### Attention in Transformers
- 3Blue1Brown
- Covers: Visual explanations of token processing

### Transformers Tech Behind LLMs
- 3Blue1Brown
- Covers: Token embeddings and transformer architecture

### ChatGPT Prompt Engineering
- DeepLearning.AI
- Covers: LLM API usage, tokens, and cost management

### OpenAI API Five Projects
- freeCodeCamp/Ania Kubow
- ~5 hours
- Covers: Practical token consumption patterns

### RAG-Based LLM App
- Tech with Tim
- Covers: RAG optimization for cost reduction

### LLM Observability Series
- AssemblyAI
- Covers: Cost tracking and monitoring

### ML System Design
- ByteByteGo
- Covers: Cost optimization for AI systems

### Fine-Tune LLMs in 2025
- Philipp Schmid
- Covers: Training cost optimization and PEFT methods

---

<a name="chapter-84---success-metrics"></a>

## Chapter 8.4: Success Metrics

**Topics:** Balanced Scorecard, Task Completion Rate, CSAT (Customer Satisfaction Score), NPS (Net Promoter Score), CES (Customer Effort Score), Deflection Rate, AHT, Metric Correlation, Multi-Dimensional Optimization

Note: High-quality resources available from Harvard Business Review (Balanced Scorecard), Qualtrics XM Institute (NPS), and Bain & Company (Net Promoter System). See recommended YouTube searches: Net Promoter Score calculation tutorial, CSAT customer satisfaction measurement, Balanced scorecard KPI framework, Customer Effort Score CES guide, Contact center metrics deflection rate, Customer satisfaction dashboard, NPS survey design, CSAT vs NPS comparison, Multi-dimensional metrics framework, and Success metric correlation analysis.

---
