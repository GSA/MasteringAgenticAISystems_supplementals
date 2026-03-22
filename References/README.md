# References and Source Materials

This directory contains the complete reference library for **Mastering Agentic AI Systems**, organized by textbook chapter and aligned with the NVIDIA Certified Professional: Agentic AI (NCP-AAI) certification exam domains.

## Contents

### Reference Index Files

Three index files catalog every reference with descriptions, file types, and chapter mappings:

| Index File | Coverage | Entries |
|------------|----------|---------|
| `reference_file_index_ch1-3.md` | Chapters 1 through 3 | 185 references |
| `reference_file_index_ch4-7.md` | Chapters 4 through 7 | 67 references |
| `reference_file_index_ch8-10.md` | Chapters 8 through 10 | 150 references |

Start with these index files to locate specific references by topic, chapter, or source type.

### NVIDIA Curriculum Materials

Three NVIDIA Deep Learning Institute (DLI) curriculum PDFs are included at the top level:

- **Curriculum - Adding New Knowledge to LLMs.pdf** -- Fine-tuning and knowledge injection techniques
- **Curriculum - Building Agentic AI Applications with LLMs.pdf** -- Agent development patterns and frameworks
- **Curriculum - Deploying RAG Pipelines for Production at Scale.pdf** -- Production RAG architecture and deployment

### Chapter Reference Directories

Each chapter directory contains the primary source materials used during research and writing. The directories follow the textbook's ten-part structure, which mirrors the ten NCP-AAI exam domains.

| Directory | Files | Exam Domain (Weight) | Key Sources |
|-----------|------:|----------------------|-------------|
| Chapter 1 - Agent Architecture and Design | 96 | Domain 1 (15%) | NeMo Agent Toolkit docs, memory system research, multi-agent communication protocols |
| Chapter 2 - Agent Development | 141 | Domain 2 (15%) | LangChain, LangGraph, AutoGen, CrewAI, Semantic Kernel framework documentation |
| Chapter 3 - Evaluation and Tuning | 14 | Domain 3 (13%) | Benchmark suites, AgentBench, WebArena, trace analysis methodologies |
| Chapter 4 - Deployment and Scaling | 16 | Domain 4 (5%) | Kubernetes guides, TensorRT-LLM documentation, scaling pattern references |
| Chapter 5 - Cognition, Planning, and Memory | 14 | Domain 5 (10%) | CoT/ToT/MCTS research papers, memory architecture surveys, planning algorithms |
| Chapter 6 - Knowledge Integration and Data Handling | 2 | Domain 6 (10%) | RAG architecture references, vector database documentation |
| Chapter 7 - NVIDIA Platform Implementation | 35 | Domain 7 (7%) | NIM deployment guides, Triton Inference Server, Riva ASR/TTS, NeMo Curator |
| Chapter 8 - Run, Monitor, and Maintain | 16 | Domain 8 (7%) | LangSmith integration, Prometheus/Grafana monitoring, profiling methodologies |
| Chapter 9 - Safety, Ethics, and Compliance | 260 | Domain 9 (5%) | NeMo Guardrails full documentation, IEEE standards, GDPR/EU AI Act, NIST AI RMF |
| Chapter 10 - Human-AI Interaction and Oversight | 13 | Domain 10 (5%) | RLHF methodology, HITL/HOTL patterns, conversational UI research |

**Total: 614 files** across all directories.

### Notable Subdirectories

Several chapters contain structured documentation sets from major NVIDIA projects:

- **Chapter 1 / Nemo_Agent_Toolkit/** -- Complete NeMo Agent Intelligence Toolkit documentation including quick-start guides, workflow references, MCP integration, tutorials, and API reference
- **Chapter 4 / TensorRT/** -- TensorRT-LLM optimization documentation
- **Chapter 9 / Nemo_Guardrail/** -- Full NeMo Guardrails documentation set (102 files) covering Colang DSL, rail types, security configurations, and integration patterns

## File Format Distribution

| Format | Count | Description |
|--------|------:|-------------|
| Markdown (.md) | 326 | Framework documentation, NVIDIA guides, integration references |
| PDF (.pdf) | 75 | Academic papers, whitepapers, regulatory documents, curriculum materials |
| Text (.txt) | 7 | URL caches, plain-text notes |
| Other | 206 | Configuration files, templates, static assets (primarily within NeMo toolkit and Guardrails documentation sets) |

## Source Categories

1. **NVIDIA Official Documentation** -- NeMo Framework, TensorRT-LLM, Triton Inference Server, NIM, Guardrails, Riva, Curator, Agent Intelligence Toolkit
2. **Agent Framework Documentation** -- LangChain, LangGraph, AutoGen, CrewAI, Semantic Kernel, LlamaIndex
3. **Academic and Research Papers** -- Agent planning surveys, memory architecture research, reasoning strategy papers, RLHF methodology
4. **Regulatory and Standards Documents** -- IEEE Ethically Aligned Design, FDA AI/ML guidance, GDPR compliance frameworks, EU AI Act, NIST AI Risk Management Framework
5. **Industry Whitepapers and Technical Blogs** -- NVIDIA technical blogs, enterprise AI architecture guides, optimization case studies

## How to Use This Directory

**Finding references for a specific chapter:** Open the corresponding index file (`reference_file_index_ch1-3.md`, `ch4-7.md`, or `ch8-10.md`) and search for the chapter number. Each entry includes a description and file path.

**Finding references by topic:** Use the index files' descriptions to search by keyword (e.g., "guardrails," "quantization," "RLHF").

**Browsing by exam domain:** The chapter directories map directly to exam domains. For exam preparation, focus on directories with higher exam weight (Chapters 1-3 cover 43% of the exam).

## License

Reference materials are included for educational and research purposes in support of the textbook. Individual documents retain their original licenses and copyright. NVIDIA documentation is subject to NVIDIA's documentation license. Academic papers are referenced under fair use for educational commentary.
