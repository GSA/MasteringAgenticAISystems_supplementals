# Part 01: Foundations - Video and Learning Resources

> **Catalog note:** A hyperlink means a resource is currently assigned; it does not by itself guarantee that every title, runtime, or scope claim has been independently verified. Entries without a hyperlink are unverified discovery candidates. See [README.md](README.md) and [video_resource_status.csv](video_resource_status.csv) for status and provenance.

## Table of Contents

- [Chapter 1.1A - UI Foundations](#chapter-11a---ui-foundations)
- [Chapter 1.1B - Accessibility & HITL Patterns](#chapter-11b---accessibility--hitl-patterns)
- [Chapter 1.2 - Core Patterns](#chapter-12---core-patterns)
- [Chapter 1.3 - Multi-Agent Systems](#chapter-13---multi-agent-systems)
- [Chapter 1.4 - Memory & Perception](#chapter-14---memory--perception)
- [Chapter 1.5A - Stateful Orchestration](#chapter-15a---stateful-orchestration)
- [Chapter 1.5B - Stateful Examples](#chapter-15b---stateful-examples)
- [Chapter 1.6 - Orchestration Pitfalls](#chapter-16---orchestration-pitfalls)
- [Chapter 1.7A - Knowledge Graphs](#chapter-17a---knowledge-graphs)
- [Chapter 1.7B - Hybrid RAG+KG](#chapter-17b---hybrid-ragkg)

---

<a name="chapter-11a---ui-foundations"></a>
## Chapter 1.1A - UI Foundations

**Topics:** Agent UI Design, Transparency, Progressive Disclosure, Human-in-the-Loop, Conversational Interfaces, Trust in AI

### Progressive Disclosure
- [https://www.nngroup.com/videos/progressive-disclosure/](https://www.nngroup.com/videos/progressive-disclosure/) ~8 minutes
- Covers: Progressive disclosure pattern, information layering, essential vs advanced views, cognitive load management

### Managing Visual Complexity in Applications and Websites
- [https://www.nngroup.com/videos/managing-visual-complexity/](https://www.nngroup.com/videos/managing-visual-complexity/) ~10 minutes
- Covers: UI complexity reduction, information architecture, clarity in design, cognitive load


### Principles of Human-Centered Design
- [https://www.nngroup.com/videos/principles-human-centered-design-don-norman/](https://www.nngroup.com/videos/principles-human-centered-design-don-norman/) ~15 minutes
- Covers: Human-centered design principles, user mental models, feedback, control, transparency

### But what is a Neural Network?
- [https://www.youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk) ~19 minutes
- Covers: Neural network fundamentals, understanding AI decision-making, building trust through transparency

### Intro to Large Language Models
- [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g) ~60 minutes
- Covers: LLM architecture, context windows, training, finetuning fundamentals

### Lex Fridman Podcast #367 - Sam Altman
- [https://www.youtube.com/watch?v=L_Guz73e6fw](https://www.youtube.com/watch?v=L_Guz73e6fw) ~2 hours 25 minutes
- Covers: OpenAI, GPT-4, AI safety and alignment, governance, and the future of AI

### Google's 9 Hour AI Prompt Engineering Course in 20 Minutes
- [https://www.youtube.com/watch?v=p09yRj47kNM](https://www.youtube.com/watch?v=p09yRj47kNM) ~20 minutes
- Covers: Prompt-engineering concepts and techniques condensed from a longer course; not an agent-coordination tutorial

---

<a name="chapter-11b---accessibility--hitl-patterns"></a>
## Chapter 1.1B - Accessibility & HITL Patterns

**Topics:** HITL patterns, WCAG accessibility, screen readers, ARIA, keyboard navigation

### Introduction to Web Accessibility and W3C Standards
- [https://www.youtube.com/watch?v=20SHvU2PKsM](https://www.youtube.com/watch?v=20SHvU2PKsM) ~4 minutes
- Covers: WCAG introduction, accessibility fundamentals

### Web Accessibility Perspectives - Compilation
- [https://www.youtube.com/watch?v=3f31oufqFSM](https://www.youtube.com/watch?v=3f31oufqFSM) ~8 minutes
- Covers: Real-world accessibility benefits, keyboard navigation, voice recognition

### Semantic HTML and Accessibility
- [https://www.youtube.com/watch?v=qSNUi7pRmWg](https://www.youtube.com/watch?v=qSNUi7pRmWg) ~25 minutes
- Covers: Semantic HTML, screen reader compatibility, proper heading hierarchy

### I Made a BIG Mistake - Kevin Powell
- [https://www.youtube.com/watch?v=YAqRQoN8ykI](https://www.youtube.com/watch?v=YAqRQoN8ykI)
- Covers: Accessible buttons and navigation, semantic HTML, visually hidden content, and corrections to earlier accessibility patterns

### Colors with Good Contrast
- [https://www.youtube.com/watch?v=Hui87z2Vx8o](https://www.youtube.com/watch?v=Hui87z2Vx8o) ~1 minute
- Covers: A concise introduction to readable color contrast for accessible interfaces

---

<a name="chapter-12---core-patterns"></a>
## Chapter 1.2 - Core Patterns

**Topics:** ReAct Pattern, Plan-and-Execute, Reflection Pattern, Tool-Use Architecture

### What's next for AI agentic workflows (Andrew Ng)
- [https://www.youtube.com/watch?v=sal78ACtGTc](https://www.youtube.com/watch?v=sal78ACtGTc) ~12 minutes
- Covers: Four key agentic patterns overview, current state of AI agents

### LangChain Crash Course for Beginners - codebasics
- [https://www.youtube.com/watch?v=nAmC7SoVLd8](https://www.youtube.com/watch?v=nAmC7SoVLd8) ~45 minutes
- Covers: LangChain fundamentals, chains, sequential chains, and introductory agent concepts

### LangChain ReAct Voice Agent Tutorial
- [https://www.youtube.com/watch?v=TdZtr1nrhJg](https://www.youtube.com/watch?v=TdZtr1nrhJg) ~25 minutes
- Covers: ReAct pattern implementation with OpenAI Realtime API, voice agent interactions

### Prompt Injection in LLM Agents
- [https://www.youtube.com/watch?v=43qfHaKh0Xk](https://www.youtube.com/watch?v=43qfHaKh0Xk) ~35 minutes
- Covers: Security vulnerabilities in ReAct agents, tool-use security, jailbreaking techniques

### AI Agent Tool Calling Example
- [https://www.youtube.com/watch?v=4KXK6c6TVXQ](https://www.youtube.com/watch?v=4KXK6c6TVXQ)
- Covers: Tool-Use Architecture, function calling mechanisms, API integration

### LangChain Agents Deep Dive with GPT-3.5 - James Briggs
- [https://www.youtube.com/watch?v=jSP-gSEyVeI](https://www.youtube.com/watch?v=jSP-gSEyVeI) ~32 minutes
- Covers: LangChain tools, agent types, AgentExecutor, calculator and SQL examples, and agent workflows

### Advanced LangChain Concepts (James Briggs)
- [https://www.youtube.com/watch?v=RflBcK0oDH0](https://www.youtube.com/watch?v=RflBcK0oDH0) ~25 minutes
- Covers: Advanced agent patterns, chaining, workflow orchestration

---

<a name="chapter-13---multi-agent-systems"></a>
## Chapter 1.3 - Multi-Agent Systems

**Topics:** Multi-Agent Collaboration, Swarm Intelligence, Communication Mechanisms, Distributed Systems

### LangGraph Multi-Agent Collaboration
- [https://www.youtube.com/watch?v=hvAPnpSfSGo](https://www.youtube.com/watch?v=hvAPnpSfSGo) ~38 minutes
- Covers: Multi-agent patterns, LangGraph implementation, agent coordination

### Distributed Systems - Full Course (Martin Kleppmann)
- [https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB](https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB) ~7 hours
- Covers: Distributed systems fundamentals, consensus algorithms, message passing, replication, Raft protocol

### Distributed Systems Lecture 1
- [https://www.youtube.com/watch?v=UEAMfLPZZhE](https://www.youtube.com/watch?v=UEAMfLPZZhE) ~60 minutes
- Covers: RPC protocols, message passing communication fundamentals, distributed architecture

### LangGraph Crash Course with Code Examples
- [https://www.youtube.com/watch?v=PqS1kib7RTw](https://www.youtube.com/watch?v=PqS1kib7RTw) ~39 minutes
- Covers: Multi-agent coordination, hierarchical orchestration, StateGraph patterns

### Simulating Natural Selection - Emergent Behavior
- [https://www.youtube.com/watch?v=0ZGbIKd0XrM](https://www.youtube.com/watch?v=0ZGbIKd0XrM) ~10 minutes
- Covers: Swarm intelligence, emergent behavior, local interactions creating global patterns

### Ballerina gRPC Service - Simple RPC
- [https://ballerina.io/learn/by-example/grpc-service-simple/](https://ballerina.io/learn/by-example/grpc-service-simple/)
- Official Ballerina documentation
- Covers: Protocol Buffers service definitions, stub generation, listener setup, and a single-request/single-response gRPC service

---

<a name="chapter-14---memory--perception"></a>
## Chapter 1.4 - Memory & Perception

**Topics:** Vector Databases, Memory Systems, Transformer Architecture, RAG, Context Windows

### Visualizing Neural Network Internals
- [https://www.youtube.com/watch?v=ChfEO8l-fas](https://www.youtube.com/watch?v=ChfEO8l-fas) ~18 minutes
- Covers: CNN visualization, perception pipeline, feature extraction techniques

### Attention in Transformers, Visually Explained
- [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc) ~25 minutes
- Covers: Transformer attention mechanism, "lost in the middle" effect, multi-headed self-attention

### RAG from Scratch - Video Playlist
- [https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) ~5-10 minutes per video
- Covers: Query translation, RAPTOR hierarchical retrieval, ColBERT embeddings, advanced RAG techniques

### How to Add Memory to LangGraph Agents
- [https://www.youtube.com/watch?v=wKeFV11Uvds](https://www.youtube.com/watch?v=wKeFV11Uvds) ~35 minutes
- Covers: Persistent memory with PostgreSQL + pgvector, long-term vs short-term memory architectures

---

<a name="chapter-15a---stateful-orchestration"></a>
## Chapter 1.5A - Stateful Orchestration

**Topics:** Stateful Orchestration, ReAct Pattern, Logic Trees, State Machines, LangGraph

### LangGraph Tutorial 1 - Building Stateful Multi-AI Agents
- [https://www.youtube.com/watch?v=gqvFmK7LpDo](https://www.youtube.com/watch?v=gqvFmK7LpDo) ~40 minutes
- Covers: State machine fundamentals, graph-based workflows, multi-agent coordination with LangGraph

### LangGraph Tutorial 2 - Multi AI Agents with External Tools
- [https://www.youtube.com/watch?v=b2iM9bPdAEs](https://www.youtube.com/watch?v=b2iM9bPdAEs) ~27 minutes
- Covers: Tool integration in stateful agents, external API calls, state persistence

### Agentic AI with LangGraph and MCP Crash Course
- [https://www.youtube.com/watch?v=dIb-DujRNEo](https://www.youtube.com/watch?v=dIb-DujRNEo) ~147 minutes
- Covers: Model Context Protocol integration, advanced LangGraph patterns, production deployment

### LangGraph Crash Course with Code Examples
- [https://www.youtube.com/watch?v=PqS1kib7RTw](https://www.youtube.com/watch?v=PqS1kib7RTw) ~39 minutes
- Covers: StateGraph fundamentals, conditional routing, state transitions, debugging

### Development with Large Language Models
- [https://www.youtube.com/watch?v=xZDB1naRUlk](https://www.youtube.com/watch?v=xZDB1naRUlk) ~150 minutes
- Covers: LLM application development, prompt engineering for stateful systems, error handling

---

<a name="chapter-15b---stateful-examples"></a>
## Chapter 1.5B - Stateful Examples

**Topics:** LangGraph StateGraph, Conditional Routing, Error Recovery, Parallel Execution

### NVIDIA NIM Multimodal RAG
- [https://www.youtube.com/watch?v=NaT5Eo97_I0](https://www.youtube.com/watch?v=NaT5Eo97_I0) ~20 minutes
- Covers: NVIDIA NIM optimization, 3× latency reduction, TensorRT acceleration for RAG pipelines

### Getting Started with LangGraph
- [https://www.youtube.com/watch?v=gqvFmK7LpDo](https://www.youtube.com/watch?v=gqvFmK7LpDo) ~40 minutes
- Covers: State schema design, TypedDict patterns, StateGraph basics

### Building Multi AI Agents Chatbots
- [https://www.youtube.com/watch?v=b2iM9bPdAEs](https://www.youtube.com/watch?v=b2iM9bPdAEs) ~27 minutes
- Covers: Conditional routing, tool integration, chatbot workflows with LangGraph

### Agentic AI with LangGraph and MCP Crash Course
- [https://www.youtube.com/watch?v=dIb-DujRNEo](https://www.youtube.com/watch?v=dIb-DujRNEo) ~147 minutes
- Covers: Error handling, state persistence, Human-in-the-Loop patterns, production deployment

### Python Asynchronous Programming
- [https://www.youtube.com/watch?v=t5Bo1Je9EmE](https://www.youtube.com/watch?v=t5Bo1Je9EmE) ~26 minutes
- Covers: Asyncio fundamentals, async/await syntax, parallel execution patterns


---

<a name="chapter-16---orchestration-pitfalls"></a>
## Chapter 1.6 - Orchestration Pitfalls

**Topics:** State Management, Infinite Loops, Parallel Execution, Multi-Agent Coordination

### LangGraph Multi-Agent Workflows
- [https://www.youtube.com/watch?v=hvAPnpSfSGo](https://www.youtube.com/watch?v=hvAPnpSfSGo) ~25 minutes
- Covers: Multi-agent coordination, state sharing, collaboration patterns

### Building AI Agent Systems with LangGraph
- [https://www.youtube.com/watch?v=5h-JBkySK34](https://www.youtube.com/watch?v=5h-JBkySK34) ~35 minutes
- Covers: Tool calls, state management, agent workflow orchestration

### Building Stateful Conversational AI Agents
- [https://www.youtube.com/watch?v=k1OEeqknoR0](https://www.youtube.com/watch?v=k1OEeqknoR0) ~30 minutes
- Covers: Stateful conversations, HITL workflows, state persistence

### How to Add Memory to LangGraph Agents
- [https://www.youtube.com/watch?v=wKeFV11Uvds](https://www.youtube.com/watch?v=wKeFV11Uvds) ~35 minutes
- Covers: LangGraph memory and checkpointing, persistent state, and long-term vs. short-term memory patterns

### LangGraph Persistence
- [https://docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- Official LangGraph documentation
- Covers: Checkpoints, threads, conversational memory, fault-tolerant recovery, pending writes, and time-travel replay

---

<a name="chapter-17a---knowledge-graphs"></a>
## Chapter 1.7A - Knowledge Graphs

**Topics:** Property Graphs, Neo4j, Cypher, NER, Relationship Extraction, Multi-hop Reasoning

### Knowledge Graphs - Computerphile
- [https://www.youtube.com/watch?v=PZBm7M0HGzw](https://www.youtube.com/watch?v=PZBm7M0HGzw) ~18 minutes
- Covers: Knowledge graph fundamentals, graph-based data representation

### Neo4j Graph Database Course - Full Tutorial
- [https://www.youtube.com/watch?v=_IgbB24scLI](https://www.youtube.com/watch?v=_IgbB24scLI) ~5 hours
- Covers: Neo4j fundamentals, Cypher language, integration with Java/Spring Boot/React

### Neo4j Cypher Query Language - Developer Webinar
- [https://www.youtube.com/watch?v=pMjwgKqMzi8](https://www.youtube.com/watch?v=pMjwgKqMzi8) ~52 minutes
- Covers: Cypher fundamentals, pattern matching, MERGE operations, multi-hop traversals

### Get Started with Python, Flask, and Neo4j
- [https://www.youtube.com/watch?v=uZqGKg0ad7k](https://www.youtube.com/watch?v=uZqGKg0ad7k)
- Covers: Connecting a Flask application to Neo4j and performing introductory graph-database operations from Python

### RAG From Scratch - Complete Course
- [https://www.youtube.com/watch?v=sVcwVQRHIc8](https://www.youtube.com/watch?v=sVcwVQRHIc8) ~3.5 hours
- Covers: RAG fundamentals, graph databases in RAG architectures, LangChain integration

### Query Structuring for Graph Databases
- [https://www.youtube.com/watch?v=kl6NwWYxvbM](https://www.youtube.com/watch?v=kl6NwWYxvbM) ~8 minutes
- Covers: Natural language to Cypher query translation, LangChain + Neo4j integration

### Named Entity Recognition with spaCy
- [https://www.youtube.com/watch?v=ytAyCO-n8tY](https://www.youtube.com/watch?v=ytAyCO-n8tY) ~18 minutes
- Covers: NER fundamentals, entity extraction, entity type classification

### How spaCy's Entity Recognition Model Works
- [https://www.youtube.com/watch?v=sqDHBH9IjRU](https://www.youtube.com/watch?v=sqDHBH9IjRU)
- Covers: Incremental parsing, Bloom embeddings, convolutional neural networks, and spaCy NER architecture

---



<a name="chapter-17b---hybrid-ragkg"></a>
## Chapter 1.7B - Hybrid RAG+KG

**Topics:** Hybrid RAG+KG Architecture, Entity Disambiguation, Cypher Optimization, Multi-Agent Coordination

### Reliable Graph RAG with Neo4j and Diffbot
- [https://www.youtube.com/watch?v=RWtQVfRXTjQ](https://www.youtube.com/watch?v=RWtQVfRXTjQ) ~38 minutes
- Covers: GraphRAG implementation, entity extraction and disambiguation with Diffbot

### GraphRAG: LLM-Derived Knowledge Graphs for RAG
- [https://www.youtube.com/watch?v=r09tJfON6kE](https://www.youtube.com/watch?v=r09tJfON6kE) ~30 minutes
- Covers: GraphRAG architecture, comparing vector RAG vs graph-based RAG approaches

### Learn LangGraph and Build Conversational AI
- [https://www.youtube.com/watch?v=jGg_1h0qzaM](https://www.youtube.com/watch?v=jGg_1h0qzaM) ~4 hours
- Covers: LangGraph framework, multi-agent coordination, graph-based state management

### CypherVis3D - 3D Visualization
- [https://www.youtube.com/watch?v=PcEUL_5NXbI](https://www.youtube.com/watch?v=PcEUL_5NXbI) ~15 minutes
- Covers: Cypher query visualization, query optimization through visual analysis

### Entity Linking with spaCy - Hands-on Tutorial
- [https://www.youtube.com/watch?v=8u57WSXVpmw](https://www.youtube.com/watch?v=8u57WSXVpmw) ~23 minutes
- Covers: Entity disambiguation techniques, training custom entity linking models

### spaCy IRL 2019 - Entity Linking Presentation
- [https://www.youtube.com/watch?v=PW3RJM8tDGo](https://www.youtube.com/watch?v=PW3RJM8tDGo) ~30 minutes
- Covers: Entity linking architecture, context-based disambiguation, production systems

### Learn to Build Graph Databases with Neo4j
- [https://www.youtube.com/watch?v=_IgbB24scLI](https://www.youtube.com/watch?v=_IgbB24scLI) ~5 hours
- Covers: Neo4j fundamentals, Cypher optimization, performance best practices, advanced patterns
