# Part 03: Deploying Agentic AI - Video and Learning Resources

> **Catalog note:** A hyperlink means a resource is currently assigned; it does not by itself guarantee that every title, runtime, or scope claim has been independently verified. Entries without a hyperlink are unverified discovery candidates. See [README.md](README.md) and [video_resource_status.csv](video_resource_status.csv) for status and provenance.

## Table of Contents

- [Chapter 3.1A - Hallucination Detection](#chapter-31a---hallucination-detection)
- [Chapter 3.1B - Grounding in External Knowledge](#chapter-31b---grounding-in-external-knowledge)
- [Chapter 3.1C - Multi-Modal Evaluation](#chapter-31c---multi-modal-evaluation)
- [Chapter 3.2 - Context Relevance](#chapter-32---context-relevance)
- [Chapter 3.3 - Harmfulness & Safety Assessment](#chapter-33---harmfulness--safety-assessment)
- [Chapter 3.4 - Behavioral Consistency](#chapter-34---behavioral-consistency)
- [Chapter 3.5 - Prompt Optimization, Few-Shot Learning, and Fine-Tuning](#chapter-35---prompt-optimization-few-shot-learning-and-fine-tuning)
- [Chapter 3.6 - Agent Benchmarking Frameworks](#chapter-36---agent-benchmarking-frameworks)
- [Chapter 3.7 - Tool Auditing](#chapter-37---tool-auditing)
- [Chapter 3.8 - Action Accuracy](#chapter-38---action-accuracy)
- [Chapter 3.9 - Reasoning Quality](#chapter-39---reasoning-quality)
- [Chapter 3.10 - Efficiency Metrics](#chapter-310---efficiency-metrics)

---

<a name="chapter-31a---hallucination-detection"></a>
## Chapter 3.1A - Hallucination Detection

**Topics:** Factual consistency, response grounding, factual overlap, knowledge verification, semantic similarity metrics

### Detecting Hallucinations in Large Language Models Using Semantic Entropy
- [https://www.youtube.com/watch?v=15I5rna-gag](https://www.youtube.com/watch?v=15I5rna-gag) ~30 minutes
- Covers: Semantic entropy as a hallucination detection technique, factual consistency verification without ground truth, theoretical foundations of semantic uncertainty, practical applications in detecting confabulations

### Do Androids Know They're Only Dreaming of Electric Sheep?
- [https://www.youtube.com/watch?v=YkLRGl8wZTM](https://www.youtube.com/watch?v=YkLRGl8wZTM) ~59 minutes
- Covers: USC ISI presentation by Sky Wang on uncertainty, model awareness, and hallucination-related questions in language models

### Introduction to RAG (Retrieval-Augmented Generation) | LlamaIndex
- [https://www.youtube.com/watch?v=A4U5CwcXr0I](https://www.youtube.com/watch?v=A4U5CwcXr0I) ~15 minutes
- Covers: RAG fundamentals for grounding LLM outputs, retrieval mechanisms to reduce hallucinations, grounding strategies using external knowledge, integration with LLM workflows

---

<a name="chapter-31b---grounding-in-external-knowledge"></a>
## Chapter 3.1B - Grounding in External Knowledge

**Topics:** RAG systems, knowledge graphs, claim verification, entity linking, grounding pipelines

### Learn RAG From Scratch - Python AI Tutorial from a LangChain Engineer
- [https://www.youtube.com/watch?v=sVcwVQRHIc8](https://www.youtube.com/watch?v=sVcwVQRHIc8) ~2 hours 33 minutes
- Covers: RAG fundamentals and architecture, document loading and chunking, embeddings and vector stores, retrieval, generation, and implementation in Python

### Building Production-Ready RAG Applications - Jerry Liu
- [https://www.youtube.com/watch?v=TRjq7t2Ms5I](https://www.youtube.com/watch?v=TRjq7t2Ms5I) ~18 minutes
- Covers: Production RAG design, data and retrieval quality, evaluation, observability, and deployment considerations

### Building Knowledge Graphs with LLMs - Neo4j NODES 2024
- [https://neo4j.com/videos/nodes-2024-building-knowledge-graphs-with-llms/](https://neo4j.com/videos/nodes-2024-building-knowledge-graphs-with-llms/) ~30 minutes
- Covers: Turning text into graph structures, text chunks, entity and relationship extraction, and constrained graph schemas

### Episode 3: RAG Techniques in LlamaIndex
- [https://www.youtube.com/watch?v=Su-ROQMaiaw](https://www.youtube.com/watch?v=Su-ROQMaiaw)
- Covers: LlamaIndex query-engine patterns including SQL routing, sub-question decomposition, recursive routing, and self-correcting retrieval

### Entity Linking: The Symbiosis between Knowledge Graphs and News
- [https://watch.knowledgegraph.tech/videos/d1-v13-silviu-cucerzan](https://watch.knowledgegraph.tech/videos/d1-v13-silviu-cucerzan) ~20 minutes
- Covers: Entity recognition, disambiguation, and linking between news content and knowledge graphs

### AI Agent Evaluation with RAGAS
- [https://www.youtube.com/watch?v=-_52DIIOsCE](https://www.youtube.com/watch?v=-_52DIIOsCE) ~20 minutes
- Covers: Hands-on RAGAS evaluation, including retrieval and generation quality metrics for RAG and agent workflows

### Building and Evaluating Advanced RAG
- [https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/information](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/information) (official 6-lesson video course; ~1 hour 55 minutes)
- Covers: Sentence-window retrieval, auto-merging retrieval, the RAG triad, and systematic evaluation of advanced RAG systems

### RAG from Scratch: Part 5 (Query Translation - Multi Query)
- [https://www.youtube.com/watch?v=JChPi0CRnDY](https://www.youtube.com/watch?v=JChPi0CRnDY)
- Covers: Generating multiple query perspectives, retrieving across them, and combining results to improve retrieval coverage

### Building Corrective RAG from Scratch with Open-Source, Local LLMs
- [https://www.youtube.com/watch?v=E2shqsYwxck](https://www.youtube.com/watch?v=E2shqsYwxck)
- Covers: Corrective retrieval, document grading, web-search fallback, and local open-source implementation in LangGraph

### Graph RAG - Microsoft Research
- [https://www.youtube.com/watch?v=r09tJfON6kE](https://www.youtube.com/watch?v=r09tJfON6kE) ~40 minutes
- Covers: Graph-based retrieval augmentation, community detection for knowledge organization, hierarchical summarization, combining graphs with traditional RAG

### The 5 Levels of Text Splitting for Retrieval
- [https://www.youtube.com/watch?v=8OJC21T2SL4](https://www.youtube.com/watch?v=8OJC21T2SL4) ~45 minutes
- Covers: Five levels of text splitting, from basic chunking through increasingly semantic retrieval-oriented methods

### Reasoning Over Semantic-Level Graph for Fact Checking
- [https://slideslive.com/38928866](https://slideslive.com/38928866) (official ACL presentation)
- Covers: Fact checking through semantic-level graph construction, evidence reasoning, and claim verification

---

<a name="chapter-31c---multi-modal-evaluation"></a>
## Chapter 3.1C - Multi-Modal Evaluation

**Topics:** Vision-language models, audio transcription, OCR, document understanding, cross-modal consistency

### OpenAI CLIP: Connecting Text and Images (Paper Explained) - Yannic Kilcher
- [https://www.youtube.com/watch?v=T9XSU0pKX2E](https://www.youtube.com/watch?v=T9XSU0pKX2E) ~48 minutes
- Covers: Contrastive image-text pretraining, zero-shot transfer, joint embedding spaces, and paper results

### Visual Question Answering with Vision Transformers
- [https://www.youtube.com/watch?v=5tW3y7lm7V0](https://www.youtube.com/watch?v=5tW3y7lm7V0) ~30 minutes
- Covers: VQA task formulation, vision transformer architectures, multi-modal fusion techniques, benchmark datasets and evaluation

### LayoutLM: Pre-training of Text and Layout for Document Image Understanding
- [https://dl.acm.org/doi/10.1145/3394486.3403172#sec-supp](https://dl.acm.org/doi/10.1145/3394486.3403172#sec-supp) (official KDD presentation video)
- Covers: Joint text-and-layout pretraining for document images, including form and document-understanding tasks

### OpenAI Whisper: Robust Speech Recognition via Large-Scale Weak Supervision - Paper Explained
- [https://www.youtube.com/watch?v=AwJf8aQfChE](https://www.youtube.com/watch?v=AwJf8aQfChE)
- Covers: Audio transcription with Whisper, multi-lingual speech recognition, robustness to noise and accents, transcription accuracy evaluation

### Vision Transformers (ViT) Explained
- [https://www.youtube.com/watch?v=TrdevFK_am4](https://www.youtube.com/watch?v=TrdevFK_am4) ~40 minutes
- Covers: Vision transformer architecture, image patch embeddings, attention mechanisms for vision, transfer learning for vision tasks

### Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4 - CVPR Tutorial
- [https://www.youtube.com/watch?v=mkI7EPD1vp8](https://www.youtube.com/watch?v=mkI7EPD1vp8)
- Covers: Large multimodal models, visual instruction tuning, LLaVA within the broader field, evaluation, and multimodal capabilities

### Stanford CME296 Lecture 7 - Evaluation
- [https://www.youtube.com/watch?v=iNaRBp4T57Q](https://www.youtube.com/watch?v=iNaRBp4T57Q) ~1 hour 41 minutes
- Covers: Evaluation of generative and vision-language systems using human ratings, FID, CLIPScore, and multimodal LLM judges

### Audio Classification with Hugging Face Transformers
- [https://www.julien.org/youtube/2022/20220726_Audio_Classification_with_Hugging_Face_Transformers.html](https://www.julien.org/youtube/2022/20220726_Audio_Classification_with_Hugging_Face_Transformers.html) (video and transcript)
- Covers: Fine-tuning a Conformer/Wav2Vec2-family model for keyword classification, testing robustness, and deploying a demo

### What is Document AI?
- [https://www.youtube.com/watch?v=1V96qmfSTe4](https://www.youtube.com/watch?v=1V96qmfSTe4) ~5 minutes
- Covers: An introduction to extracting, classifying, and understanding information from documents with Document AI

### Visually Grounded Language Understanding and Generation
- [https://www.microsoft.com/en-us/research/video/visually-grounded-language-understanding-and-generation/](https://www.microsoft.com/en-us/research/video/visually-grounded-language-understanding-and-generation/) (Microsoft Research talk)
- Covers: Visual grounding, pretrained vision-language representations, and grounded image-caption generation

### Multi-Modal Hallucination Control by Visual Information Grounding
- [https://cvpr.thecvf.com/virtual/2024/poster/30302](https://cvpr.thecvf.com/virtual/2024/poster/30302) (official CVPR 2024 presentation)
- Covers: Reducing multimodal hallucination by grounding generated outputs in visual information

### Evaluating Audio Transcription Quality
- [https://www.youtube.com/watch?v=mRB8tbTkkrA](https://www.youtube.com/watch?v=mRB8tbTkkrA) ~20 minutes
- Covers: WER (Word Error Rate) metrics, transcription quality assessment, robustness evaluation for ASR, multi-lingual evaluation challenges

---

<a name="chapter-32---context-relevance"></a>
## Chapter 3.2 - Context Relevance

**Topics:** Retrieval evaluation, context precision/recall, semantic relevance, noise reduction, query understanding

### Dense Passage Retrieval for Open-Domain Question Answering
- [https://slideslive.com/38939151](https://slideslive.com/38939151) (official ACL presentation)
- Covers: Dense bi-encoder retrieval for open-domain QA, training with negatives, and comparison with sparse retrieval

### What is Retrieval-Augmented Generation (RAG)? - IBM Technology
- [https://www.youtube.com/watch?v=T-D1OfcDW1M](https://www.youtube.com/watch?v=T-D1OfcDW1M) ~7 minutes
- Covers: A concise explanation of RAG architecture, retrieval, grounding, and generation

### Text Embeddings and Semantic Search
- [https://www.youtube.com/watch?v=OATCgQtNX2o](https://www.youtube.com/watch?v=OATCgQtNX2o)
- Covers: Embedding representations, vector similarity, semantic search, and retrieval applications

### RAG from Scratch - Part 9: Hypothetical Document Embeddings (HyDE)
- [https://www.youtube.com/watch?v=SaDzIVkYqyY](https://www.youtube.com/watch?v=SaDzIVkYqyY)
- Covers: Query transformation for improved retrieval, hypothetical answer generation, retrieval with LLM-generated queries, precision improvement techniques

---

<a name="chapter-33---harmfulness--safety-assessment"></a>
## Chapter 3.3 - Harmfulness & Safety Assessment

**Topics:** Red-teaming, jailbreak detection, toxicity classifiers, bias evaluation, safety benchmarks

### Red Teaming LLM Applications
- [https://www.deeplearning.ai/courses/red-teaming-llm-applications](https://www.deeplearning.ai/courses/red-teaming-llm-applications) (official 7-lesson video course; ~1 hour 19 minutes)
- Covers: Systematic red teaming of LLM applications, prompt-injection testing, vulnerability discovery, and practical testing with Giskard

### Intro to AI Safety: Remastered - Rob Miles
- [https://www.youtube.com/watch?v=pYXy-A4siMw](https://www.youtube.com/watch?v=pYXy-A4siMw)
- Covers: A concise introduction to why advanced AI systems can be difficult to specify, control, and align

### ML Practicum: Fairness in Perspective API
- [https://www.youtube.com/watch?v=pHT-ImFXPQo](https://www.youtube.com/watch?v=pHT-ImFXPQo)
- Covers: Fairness analysis and evaluation considerations for toxicity predictions produced by Perspective API

### AI, Ain't I A Woman? - Joy Buolamwini
- [https://www.youtube.com/watch?v=QxuyfWoVV98](https://www.youtube.com/watch?v=QxuyfWoVV98)
- Covers: A spoken-word and visual demonstration of facial-analysis failures affecting women of color

### Holistic Evaluation of Language Models (HELM)
- [https://www.youtube.com/watch?v=A0kD00WdlKY](https://www.youtube.com/watch?v=A0kD00WdlKY)
- Covers: HELM’s multi-scenario, multi-metric framework for transparent and holistic language-model evaluation

### Constitutional AI
- [https://www.youtube.com/watch?v=Tjsox6vfsos](https://www.youtube.com/watch?v=Tjsox6vfsos) ~6 minutes
- Covers: The Constitutional AI approach, including written principles, self-critique, revision, and harmlessness training

### SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner
- [https://www.youtube.com/watch?v=bUamsPUHURA](https://www.youtube.com/watch?v=bUamsPUHURA) (USENIX Security 2025 presentation; ~15 minutes)
- Covers: A practical defense architecture in which LLMs detect and respond to jailbreak attempts

### 21 Fairness Definitions and Their Politics - Arvind Narayanan
- [https://www.youtube.com/watch?v=jIXIuYdnyyk](https://www.youtube.com/watch?v=jIXIuYdnyyk)
- Covers: Competing definitions of algorithmic fairness, incompatibilities among metrics, and the policy choices behind them

### RLHF: From Zero to ChatGPT - Hugging Face
- [https://www.youtube.com/watch?v=2MBJOuVq380](https://www.youtube.com/watch?v=2MBJOuVq380)
- Covers: The RLHF pipeline from supervised fine-tuning through preference modeling and reinforcement learning

---

<a name="chapter-34---behavioral-consistency"></a>
## Chapter 3.4 - Behavioral Consistency

**Topics:** Persona consistency, style adherence, preference drift, multi-turn coherence, agent state management

### Will I Sound Like Me? Improving Persona Consistency in Dialogues through Pragmatic Self-Consciousness
- [https://slideslive.com/38938705](https://slideslive.com/38938705) (official ACL presentation)
- Covers: Improving and evaluating persona consistency across dialogue turns through pragmatic self-consciousness

### Long Term Memory with LangGraph
- [https://www.youtube.com/watch?v=R0OdB-p-ns4](https://www.youtube.com/watch?v=R0OdB-p-ns4) ~1 hour 32 minutes
- Covers: Architectures and implementation patterns for long-term memory in LangGraph-based agents

### LangGraph Persistence for Human-in-the-Loop Workflows
- [https://www.youtube.com/watch?v=9BPCV5TYPmg](https://www.youtube.com/watch?v=9BPCV5TYPmg)
- Covers: Persistence and checkpointing that let a workflow pause for human approval or edits and resume from saved state

---

<a name="chapter-35---prompt-optimization-few-shot-learning-and-fine-tuning"></a>
## Chapter 3.5 - Prompt Optimization, Few-Shot Learning, and Fine-Tuning

**Topics:** Prompt engineering, chain-of-thought, few-shot learning, fine-tuning, LoRA, RLHF, reward modeling

### Attention in Transformers - Visual Explanation
- [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc) ~27 minutes
- Covers: Self-attention mechanism foundations, multi-head attention, context processing in transformers, token embeddings and representation learning

### AI Agents - State of Affairs (Andrew Ng & Harrison Chase)
- [https://www.youtube.com/watch?v=4pYzYmSdSH4](https://www.youtube.com/watch?v=4pYzYmSdSH4) ~45 minutes
- Covers: Agentic design patterns (reflection, tool use, planning, multi-agent), agent evaluation and workflows, production agentic systems development, spectrum-based view of agenticness

### State-of-the-Art Prompting for AI Agents
- [https://www.youtube.com/watch?v=DL82mGde6wo](https://www.youtube.com/watch?v=DL82mGde6wo) ~50 minutes
- Covers: Advanced prompting techniques from production experience, metaprompting and evaluation-driven optimization, rubric-based prompt reliability, real-world prompt engineering insights

### RLHF: From Zero to ChatGPT - Hugging Face
- [https://www.youtube.com/watch?v=2MBJOuVq380](https://www.youtube.com/watch?v=2MBJOuVq380)
- Covers: Supervised fine-tuning, human preference data, reward modeling, and reinforcement learning for language models

### LoRA & QLoRA Fine-Tuning Explained In-Depth
- [https://www.youtube.com/watch?v=t1caDsMzWBk](https://www.youtube.com/watch?v=t1caDsMzWBk)
- Covers: Parameter-efficient fine-tuning with QLoRA, dataset preparation from real data, practical implementation with Hugging Face, conversation chain construction

---

<a name="chapter-36---agent-benchmarking-frameworks"></a>
## Chapter 3.6 - Agent Benchmarking Frameworks

**Topics:** AgentBench, WebArena, GAIA, API-Bank, HumanEval, task-specific evaluation, leaderboards

### AgentBench - Benchmarking LLMs as Agents
- [https://www.youtube.com/watch?v=lREQzTVJbIY](https://www.youtube.com/watch?v=lREQzTVJbIY) ~25 minutes
- Covers: Multi-environment agent evaluation, task diversity in agent benchmarking, performance metrics across domains, comparison of LLMs in agentic settings

### Evaluating Code Generation with HumanEval
- [https://www.youtube.com/watch?v=i8wpLm2j0I0](https://www.youtube.com/watch?v=i8wpLm2j0I0) ~20 minutes
- Covers: HumanEval benchmark for code synthesis, Pass@k metrics for code correctness, functional correctness evaluation, test-based assessment methodology

### Evaluating and Debugging Generative AI
- [https://www.deeplearning.ai/courses/evaluating-debugging-generative-ai](https://www.deeplearning.ai/courses/evaluating-debugging-generative-ai) (official 7-lesson video course; ~50 minutes)
- Covers: Systematic evaluation, debugging, error analysis, and monitoring of generative-AI applications

### Evaluating Tool Use in Language Models
- [https://www.youtube.com/watch?v=kqm8pNGX96k](https://www.youtube.com/watch?v=kqm8pNGX96k) ~40 minutes
- Covers: Tool-calling evaluation metrics, API interaction accuracy, function calling benchmarks, Berkeley Function Calling Leaderboard concepts

### What's Next for AI Agentic Workflows - Andrew Ng
- [https://www.youtube.com/watch?v=sal78ACtGTc](https://www.youtube.com/watch?v=sal78ACtGTc) ~12 minutes
- Covers: Reflection, tool use, planning, and multi-agent collaboration as agentic workflow patterns

### SWE-agent: Software Engineering Agents Evaluated on SWE-bench
- [https://www.youtube.com/watch?v=CeMtJ4XObAM](https://www.youtube.com/watch?v=CeMtJ4XObAM) (official project introduction)
- Covers: Language-model agents that browse, edit, and execute repository code, the Agent-Computer Interface, GitHub issue resolution, and SWE-bench evaluation

### Multi-Task Evaluation of Language Models
- [https://www.youtube.com/watch?v=gEZrGsRMK4k](https://www.youtube.com/watch?v=gEZrGsRMK4k) ~55 minutes
- Covers: Cross-task generalization, multi-domain evaluation strategies, task sampling and aggregation, benchmark design principles

### Towards Unified Dialogue System Evaluation: A Comprehensive Analysis of Current Evaluation Protocols
- [https://www.youtube.com/watch?v=icJNtco4EoI](https://www.youtube.com/watch?v=icJNtco4EoI) (official ACL presentation)
- Covers: Dialogue-system evaluation protocols, automated and human measures, and limitations of current conversational evaluation practice

### Why Evals Matter | LangSmith Evaluations - Part 1
- [https://www.youtube.com/watch?v=vygFgCNR7WA](https://www.youtube.com/watch?v=vygFgCNR7WA)
- Covers: Why structured evaluations matter, how evaluation datasets and criteria support model and prompt decisions, and the LangSmith evaluation series

---

<a name="chapter-37---tool-auditing"></a>
## Chapter 3.7 - Tool Auditing

**Topics:** Tool contracts, JSON schema, validation, hallucination detection, recovery mechanisms, distributed tracing

### Trace-Based Testing with OpenTelemetry
- [https://www.youtube.com/watch?v=WMRicNlaehc](https://www.youtube.com/watch?v=WMRicNlaehc) ~20 minutes
- Covers: Post-execution response validation through traces, testing tool invocations in distributed systems, observability-driven testing approaches

### Jaeger V2 and Distributed Tracing
- [https://www.youtube.com/watch?v=lICivVwm-F8](https://www.youtube.com/watch?v=lICivVwm-F8) ~35 minutes
- Covers: Jaeger V2 distributed tracing platform, OpenTelemetry integration, production monitoring patterns, distributed system debugging

### Monadic Error Handling in Python
- [https://www.youtube.com/watch?v=J-HWmoTKhC8](https://www.youtube.com/watch?v=J-HWmoTKhC8) ~25 minutes
- Covers: Advanced error handling patterns, alternatives to traditional exception handling, recovery mechanisms for tool failures, graceful degradation patterns

### Greg Kamradt LangChain Tutorial Series - Playlist Introduction
- [https://www.youtube.com/watch?v=_v_fgW2SkkQ&list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5](https://www.youtube.com/watch?v=_v_fgW2SkkQ&list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5) (24-video playlist opener)
- Covers: A broad introduction to LangChain and the associated tutorial playlist; not a focused tools-and-functions lesson

---

<a name="chapter-38---action-accuracy"></a>
## Chapter 3.8 - Action Accuracy

**Topics:** Tool selection accuracy, parameter validation, execution paths, trajectory quality, LLM-as-judge

### OpenAI Functions + LangChain - Multi-Tool Agent
- [https://www.youtube.com/watch?v=4KXK6c6TVXQ](https://www.youtube.com/watch?v=4KXK6c6TVXQ) ~20-30 minutes
- Covers: Function calling with LLMs, tool selection and invocation, multi-tool agent architectures, parameter passing to functions

### LLM as a Judge - Evaluation Tutorial
- [https://www.youtube.com/watch?v=kP_aaFnXLmY](https://www.youtube.com/watch?v=kP_aaFnXLmY) ~40 minutes
- Covers: LLM-as-judge evaluation methodology, iterative prompt improvement for evaluation, rubric design for quality assessment, binary classification metrics (precision, recall, F1)

### Multi-Agent AI Systems with AutoGen
- [https://www.youtube.com/watch?v=f5Qr8xUeSH4](https://www.youtube.com/watch?v=f5Qr8xUeSH4) ~40-60 minutes
- Covers: Multi-agent workflows and coordination, agent communication patterns, task decomposition and execution, tool calling in multi-agent contexts

### RAG Components & Troubleshooting with Arize Phoenix
- [https://www.youtube.com/watch?v=hbQYDpJayFw](https://www.youtube.com/watch?v=hbQYDpJayFw) ~45-60 minutes
- Covers: RAG system components and troubleshooting, Arize Phoenix observability platform, tracing LLM application runtime, production monitoring and evaluation

---

<a name="chapter-39---reasoning-quality"></a>
## Chapter 3.9 - Reasoning Quality

**Topics:** Chain-of-thought, reasoning evaluation, LLM-as-judge, self-reflection, NLI, formal logic, reasoning metrics

### Andrew Ng on AI Agents and Agentic Reasoning
- [https://www.youtube.com/watch?v=KrRD7r7y7NY](https://www.youtube.com/watch?v=KrRD7r7y7NY) ~45 minutes
- Covers: Four agentic design patterns (Reflection, Tool use, Planning, Multi-agent), reflection pattern for self-evaluation, performance improvements with agentic workflows, evaluation and error analysis in agents

### Andrej Karpathy - How I Use LLMs
- [https://www.youtube.com/watch?v=EWvNQjAaOHw](https://www.youtube.com/watch?v=EWvNQjAaOHw) ~2 hours 11 minutes
- Covers: Practical LLM workflows for writing, coding, research, automation, and everyday problem solving

### Attention in Transformers - 3Blue1Brown
- [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc) ~25 minutes
- Covers: Attention mechanism in transformers, context processing and token embeddings, mathematical details of attention

### Large Reasoning Models (LRMs) - IBM Technology
- [https://www.youtube.com/watch?v=enLbj0igyx4](https://www.youtube.com/watch?v=enLbj0igyx4) ~10-15 minutes
- Covers: Difference between LLMs and LRMs, chain-of-thought training with logic puzzles, internal verification and deliberation, reasoning accuracy vs computational cost tradeoffs

### Geospatial Trajectory Generation via Efficient Abduction - PyReason / ICLP 2024
- [https://www.youtube.com/watch?v=8nxuIaTpZzM](https://www.youtube.com/watch?v=8nxuIaTpZzM) (ICLP 2024 talk)
- Covers: Abduction over annotated logic programs, A* search, geospatial trajectory generation, explainability, and deployment using PyReason

### SATNet - Constraint Learning and Neural-Symbolic Reasoning
- [https://www.youtube.com/watch?v=IsDpoXExmNA](https://www.youtube.com/watch?v=IsDpoXExmNA) Part of playlist series
- Covers: SATNet architecture for constraint satisfaction, neural-symbolic integration, logical constraint enforcement in neural networks, structured problem solving with differentiable SAT solvers

---

<a name="chapter-310---efficiency-metrics"></a>
## Chapter 3.10 - Efficiency Metrics

**Topics:** Token economics, AgentDiet framework, workflow architecture, model routing, prompt engineering for efficiency

### What's Next for AI Agentic Workflows - Andrew Ng
- [https://www.youtube.com/watch?v=sal78ACtGTc](https://www.youtube.com/watch?v=sal78ACtGTc) ~15 minutes
- Covers: Four agentic design patterns, agentic vs non-agentic workflow comparison, performance benefits of iterative workflows, fast token generation for efficiency, smaller models with agentic patterns outperforming larger models

### Attention in Transformers - 3Blue1Brown
- [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc) ~26 minutes
- Covers: Attention mechanism in transformers, KV pairs in attention computation, efficient context processing

### Visual Guide to Transformer Neural Networks - Episode 2: Multi-Head & Self-Attention
- [https://www.youtube.com/watch?v=mMa2PmYJlCo](https://www.youtube.com/watch?v=mMa2PmYJlCo) ~15 minutes
- Covers: Multi-head attention visualization, transformer architecture components, computational overhead of attention

### Prompt Engineering Best Practices
- [https://www.youtube.com/watch?v=chAQGTBMXXQ](https://www.youtube.com/watch?v=chAQGTBMXXQ) ~25-30 minutes
- Covers: Efficient prompting techniques, token optimization through prompt design, reducing unnecessary verbosity, few-shot learning efficiency

### Let's Build GPT from Scratch - Andrej Karpathy
- [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) ~2 hours
- Covers: Building GPT in PyTorch, tokenization and token economics, attention mechanism implementation, training loop and inference optimization, understanding computational costs at code level

### But What is a GPT? - 3Blue1Brown
- [https://www.youtube.com/watch?v=wjZofJX0v4M](https://www.youtube.com/watch?v=wjZofJX0v4M) ~27 minutes
- Covers: GPT architecture and scale, 175 billion weights organization, word embeddings and tokens, computational scale and efficiency considerations
