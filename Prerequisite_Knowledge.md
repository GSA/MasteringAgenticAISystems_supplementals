# Prerequisite Knowledge for NVIDIA Agentic AI Certification

## How to Use This Guide

This document outlines the foundational knowledge you should have before beginning this textbook. Prerequisites are organized into three tiers:

- **Essential** - You must understand these concepts to benefit from the book. Without them, you'll struggle with fundamental material.
- **Recommended** - Strong understanding significantly improves your learning experience. You can succeed without them, but expect to reference external resources frequently.
- **Beneficial** - These accelerate learning and deepen understanding but aren't strictly required.

Each prerequisite includes:
- **What it is** - Clear definition
- **Why it matters** - Connection to textbook content
- **Self-assessment questions** - Test your readiness
- **If uncertain** - Resources for remediation

**Study Time Estimate**: If you need remediation, expect 10-40 hours depending on your background.

---

## Part 1: Technical Foundations

### ESSENTIAL: Large Language Model (LLM) Fundamentals

**What it is**: Understanding how modern language models work—token-by-token generation, context windows, prompting strategies, and basic capabilities/limitations.

**Why it matters**: Every agent architecture in this book uses LLMs as reasoning engines. You'll design around context limits, plan for inference latency, and choose models based on reasoning capabilities. Without LLM fundamentals, architectural decisions feel arbitrary rather than principled.

**Self-assessment**:
- Can you explain what a token is and estimate token consumption for a given prompt?
- Do you understand the difference between greedy decoding and nucleus sampling, and when you'd choose each?
- Can you explain what temperature controls and how it affects output?
- Can you describe what a context window is and why it limits agent memory?
- Do you know the difference between zero-shot, few-shot, and chain-of-thought prompting?

**If uncertain**:
- Review: "Attention Is All You Need" (Transformers paper) for architecture fundamentals
- Complete: Andrew Ng's "ChatGPT Prompt Engineering for Developers" course (2 hours)
- Read: OpenAI's GPT-4 technical report or Anthropic's Claude documentation
- Time needed: 4-6 hours

**Critical because**: All 10 parts assume you understand what LLMs can and cannot do. Part 1 teaches agent architectures built on LLMs. Part 5 teaches advanced reasoning patterns like Tree-of-Thought that depend on understanding LLM generation. Part 7 teaches NVIDIA platform optimization requiring deep LLM inference knowledge.

---

### ESSENTIAL: Python Programming

**What it is**: Intermediate Python fluency including functions, classes, dictionaries, list comprehensions, exception handling, and the ability to read and modify existing code.

**Why it matters**: All code examples and labs use Python. Framework implementations (LangChain, LangGraph, AutoGen, CrewAI) are Python-based. You'll need to read framework code, implement agents, and debug failures.

**Self-assessment**:
- Can you write a Python function that accepts keyword arguments and returns a dictionary?
- Can you read class-based code and understand method calls, inheritance, and composition?
- Do you understand Python's exception handling (`try`/`except`/`finally`)?
- Can you work with nested data structures (dictionaries containing lists of dictionaries)?
- Can you use list comprehensions and dictionary comprehensions?

**If uncertain**:
- Complete: Python for Everybody specialization (Coursera, weeks 1-5)
- Read: "Fluent Python" chapters 1-4 (functions and objects)
- Practice: Implement 5-10 small programs manipulating data structures
- Time needed: 15-25 hours if starting from basics

**Critical because**: Part 2 (Frameworks) provides Python code examples for every concept. Part 7 (NVIDIA Platform) includes Python implementation of optimization techniques. Part 8 (Operations) includes Python instrumentation code. You cannot learn by copying code you don't understand.

---

### ESSENTIAL: RESTful APIs and HTTP Fundamentals

**What it is**: Understanding HTTP request-response patterns, common verbs (GET, POST, PUT, DELETE), status codes (200, 404, 500), JSON serialization, and how distributed systems communicate through APIs.

**Why it matters**: Agents call external tools through APIs. Multi-agent communication uses API patterns. Deployment architectures rely on understanding API design, latency, and failure modes. Tool orchestration, agent coordination protocols, and production deployment all depend on API fluency.

**Self-assessment**:
- Can you explain the difference between GET and POST requests?
- Do you understand what HTTP status codes mean (2xx success, 4xx client error, 5xx server error)?
- Can you design a REST endpoint for a tool (what verb, what parameters, what responses)?
- Can you parse JSON responses and handle nested objects?
- Do you understand what API timeouts are and why they matter?

**If uncertain**:
- Read: MDN's HTTP guide (2-3 hours)
- Complete: "REST API Best Practices" tutorial
- Practice: Use Python's `requests` library to call 3-5 public APIs
- Time needed: 3-5 hours

**Critical because**: Part 1 Chapter 1.2 (Architecture Patterns) introduces tool-use architecture assuming API fluency. Part 2 Chapter 2.3 (Tool Integration) teaches framework-based tool calling. Part 4 (Deployment) architects microservices communicating via REST. Part 7 (NVIDIA Platform) deploys inference endpoints as REST services.

---

### RECOMMENDED: Software Architecture Patterns

**What it is**: Recognizing common architectural styles—client-server, microservices, pub-sub messaging, layered architectures, event-driven systems. Understanding separation of concerns, loose coupling, and architectural trade-offs (complexity vs. scalability).

**Why it matters**: Agent orchestration patterns (centralized, decentralized, hierarchical) map directly to software architectures. Part 1 Chapter 1.5 on Orchestration Patterns essentially asks: "Which software architecture fits this multi-agent problem?" Understanding architecture patterns accelerates pattern recognition and design decisions.

**Self-assessment**:
- Can you explain the difference between monolithic and microservices architectures?
- Do you understand what "loose coupling" means and why it matters?
- Can you describe pub-sub messaging and when it's appropriate?
- For a system with 200 agents coordinating warehouse operations, would you choose centralized or decentralized orchestration? Why?

**If uncertain**:
- Read: "Software Architecture in Practice" by Len Bass (chapters 1-3)
- Review: "System Design Interview" common patterns (load balancing, caching, replication)
- Study: Real system architectures (Uber, Netflix, AWS) case studies
- Time needed: 5-8 hours

**Recommended because**: Understanding architecture patterns makes Part 1's orchestration patterns (Chapter 1.5) intuitive rather than memorization. Part 4's deployment patterns directly apply software architecture principles.

---

### RECOMMENDED: Database Fundamentals

**What it is**: Understanding relational vs. non-relational databases, basic SQL, indexing strategies, query optimization, ACID guarantees vs. eventual consistency, and storage trade-offs.

**Why it matters**: Part 1 Chapter 1.3 (Memory Systems) requires understanding persistence. Long-term memory lives in databases—vector databases for semantic search, relational databases for structured facts. Part 6 (Knowledge Integration) stores embeddings in vector databases. Part 8 (Operations) logs to databases for monitoring.

**Self-assessment**:
- Can you write a basic SQL SELECT query with WHERE and JOIN clauses?
- Do you understand what database indexes do and why they improve query performance?
- Can you explain ACID properties (Atomicity, Consistency, Isolation, Durability)?
- Can you compare SQL databases (PostgreSQL) vs. NoSQL (MongoDB) vs. vector databases (Pinecone)?

**If uncertain**:
- Complete: SQL basics tutorial (W3Schools or Mode Analytics, 3-4 hours)
- Read: "Designing Data-Intensive Applications" by Martin Kleppmann (chapters 1-3)
- Practice: Set up PostgreSQL, create tables, run queries
- Time needed: 6-10 hours

**Recommended because**: Part 1.3 (Memory Systems) teaches memory architecture requiring database knowledge. Part 6 (RAG and Knowledge Integration) extensively uses vector databases. Without database fundamentals, you'll treat storage as a black box rather than understanding design trade-offs.

---

### BENEFICIAL: Distributed Systems Concepts

**What it is**: Understanding how systems communicate across networks—message passing, eventual consistency, synchronous vs. asynchronous communication, consensus protocols, network failures, and CAP theorem basics.

**Why it matters**: Part 1 Chapters 1.4-1.5 (Multi-Agent Coordination and Orchestration) center on distributed multi-agent systems where agents operate across processes, servers, or organizations. Understanding distributed systems principles prevents architecture mistakes like assuming instant message delivery or perfect ordering.

**Self-assessment**:
- Can you explain the difference between synchronous and asynchronous messaging?
- If three agents coordinate on a task and the network drops for 5 seconds, how should the system recover?
- Do you understand what "eventual consistency" means and why distributed systems accept it?
- Can you explain what consensus protocols solve (Raft, Paxos at high level)?

**If uncertain**:
- Read: "Designing Distributed Systems" by Brendan Burns (chapters 1-4)
- Review: MIT's 6.824 Distributed Systems lecture notes (lectures 1-3)
- Study: CAP theorem and its implications
- Time needed: 8-12 hours

**Beneficial because**: Essential for Part 1.4-1.5 (multi-agent coordination) but less critical for earlier chapters. If you're only building single-agent systems, you can defer this. If you're architecting multi-agent systems or enterprise deployments, this becomes essential.

---

## Part 2: Infrastructure and Tools

### ESSENTIAL: Command Line and Shell Scripting

**What it is**: Ability to navigate filesystems using `cd`, `ls`, `pwd`; manage processes with `ps`, `kill`, `top`; manipulate files with `cat`, `grep`, `sed`; write basic bash scripts with variables, loops, and conditionals.

**Why it matters**: Part 3 (Deployment) and Part 4 (Scaling) require command-line fluency for SSH access, container management, Kubernetes operations, and infrastructure automation. Every deployment step involves terminal commands. Without CLI competency, you'll struggle with tooling.

**Self-assessment**:
- Can you navigate a Linux filesystem and find files using `find` or `grep`?
- Can you write a bash script that loops through files and processes them?
- Do you understand environment variables and how to set them (`export`)?
- Can you manage background processes and redirect output to files?
- Can you use `ssh` to connect to remote servers?

**If uncertain**:
- Read: "The Linux Command Line" by William E. Shotts (first 10 chapters)
- Complete: Linux Command Line Basics tutorial (2-3 hours)
- Practice: Navigate your filesystem, write 3-5 bash scripts
- Time needed: 4-6 hours

**Essential because**: Part 3.1 (Deployment Fundamentals) assumes shell competency. Part 4 (Scaling) involves Kubernetes CLI (`kubectl`), Docker CLI, and infrastructure scripts. Part 8 (Operations) uses command-line monitoring tools.

---

### ESSENTIAL: Docker and Containerization Basics

**What it is**: Understanding what containers are, how they differ from virtual machines, basic Dockerfile syntax, building images, running containers, volume mounts, and networking basics.

**Why it matters**: Part 3 teaches containerization for reproducible deployment. Part 4 scales containerized agents using Kubernetes. Part 7 deploys NVIDIA NIM containers. Without container fundamentals, you'll treat Docker as magic rather than understanding isolation, layers, and resource limits.

**Self-assessment**:
- Can you explain what a Docker image is and how it differs from a container?
- Can you write a basic Dockerfile with `FROM`, `COPY`, `RUN`, `CMD`?
- Do you understand what container isolation provides (filesystem, process, network)?
- Can you run a container, map ports, mount volumes, and view logs?
- Do you understand why containerization solves "works on my machine" problems?

**If uncertain**:
- Complete: Docker's "Getting Started" tutorial (2 hours)
- Read: "Docker Deep Dive" by Nigel Poulton (chapters 1-5)
- Practice: Build 2-3 Dockerfiles, run containers, troubleshoot failures
- Time needed: 5-8 hours

**Essential because**: Part 3.1 immediately teaches containerization. Part 3.3 (Container Orchestration) depends on understanding containers. Part 7 deploys inference servers as containers. You cannot skip Docker and succeed in production deployment chapters.

---

### RECOMMENDED: Kubernetes Fundamentals

**What it is**: Understanding Kubernetes basics—pods (smallest unit), services (networking), deployments (replica management), namespaces, kubectl CLI, and how Kubernetes schedules workloads across clusters.

**Why it matters**: Part 3.3 (Container Orchestration) and Part 4 (Scaling) extensively use Kubernetes. Modern production agent systems deploy to Kubernetes or equivalent orchestrators. Understanding pods, services, and deployments enables you to scale from 1 to 1000 agent instances.

**Self-assessment**:
- Can you explain what a Kubernetes pod is and why it's the smallest deployable unit?
- Do you understand how Kubernetes services expose pods to networks?
- Can you describe how deployments manage replica counts and rolling updates?
- Can you use `kubectl` to view pods, services, and logs?
- Do you understand resource requests and limits (CPU, memory)?

**If uncertain**:
- Complete: Kubernetes "Learn Kubernetes Basics" official tutorial (3-4 hours)
- Read: "Kubernetes Up & Running" by Kelsey Hightower (chapters 1-6)
- Practice: Deploy a simple application to Minikube, scale it, update it
- Time needed: 8-12 hours

**Recommended because**: Part 3.3 assumes Kubernetes basics. Part 4 extensively uses Kubernetes for scaling patterns. If you're deploying to cloud platforms with Kubernetes (EKS, GKE, AKS), this moves from recommended to essential.

---

### BENEFICIAL: GPU Architecture and CUDA Basics

**What it is**: Understanding GPU vs. CPU differences, parallel processing concepts, GPU memory hierarchy, basic CUDA programming model (kernels, threads, blocks), and tensor operations.

**Why it matters**: Part 7 (NVIDIA Platform) optimizes LLM inference on GPUs. Understanding GPU architecture helps you comprehend why quantization works, how batching improves throughput, and what KV cache optimization achieves. Part 4 profiles GPU utilization with NVIDIA Nsight.

**Self-assessment**:
- Can you explain why GPUs excel at matrix multiplication (LLM operations)?
- Do you understand the difference between GPU memory and CPU memory?
- Can you explain what parallel processing means for batch inference?
- Do you know what GPU utilization percentages indicate?

**If uncertain**:
- Read: NVIDIA's "CUDA C Programming Guide" (introduction)
- Review: "Programming Massively Parallel Processors" by Kirk & Hwu (chapter 1)
- Watch: NVIDIA's GPU architecture overview videos
- Time needed: 4-6 hours

**Beneficial because**: You can learn Part 7's optimization techniques without deep GPU knowledge—the book explains what you need. However, GPU fundamentals make optimizations intuitive rather than recipes to follow.

---

## Part 3: Domain Knowledge

### ESSENTIAL: Machine Learning Fundamentals

**What it is**: Understanding supervised vs. unsupervised learning, training vs. inference, model evaluation metrics (accuracy, precision, recall), overfitting, and the concept of embeddings/vector representations.

**Why it matters**: Part 6 (Knowledge Integration) uses embeddings and vector similarity for RAG. Part 7 (NVIDIA Platform) optimizes inference. Part 9 (Safety) addresses algorithmic bias. You need ML fundamentals to understand why agents behave as they do.

**Self-assessment**:
- Can you explain the difference between training a model and using it for inference?
- Do you understand what embeddings are and why semantic similarity works in vector space?
- Can you explain overfitting and why it matters for generalization?
- Do you know what precision and recall measure?

**If uncertain**:
- Complete: Andrew Ng's "Machine Learning" course (Coursera, weeks 1-3)
- Read: "Hands-On Machine Learning" by Aurélien Géron (chapters 1-2)
- Practice: Train a simple scikit-learn classifier, evaluate it
- Time needed: 10-15 hours

**Essential because**: Part 6 extensively uses embeddings and vector databases. Part 7 discusses model quantization and optimization. Part 9 analyzes bias in ML systems. Without ML fundamentals, these sections feel like incantations rather than principled techniques.

---

### RECOMMENDED: Natural Language Processing (NLP) Basics

**What it is**: Understanding tokenization, word embeddings, attention mechanisms, transformer architecture at a high level, and common NLP tasks (classification, generation, question answering).

**Why it matters**: LLMs are NLP models. Understanding transformers and attention helps you comprehend context windows, token limits, and why certain architectural decisions exist. Part 5 (Cognition) teaches advanced reasoning patterns that build on transformer capabilities.

**Self-assessment**:
- Can you explain what tokenization does and why it matters?
- Do you understand what attention mechanisms compute (high-level)?
- Can you describe the transformer architecture (encoder-decoder, self-attention)?
- Do you know what makes transformer models different from RNNs?

**If uncertain**:
- Read: "Attention Is All You Need" paper (original transformers paper)
- Complete: Hugging Face NLP course (chapters 1-2)
- Watch: Stanford CS224N lecture on transformers
- Time needed: 6-10 hours

**Recommended because**: Understanding transformers deepens your LLM comprehension in Part 1. Part 5's advanced reasoning patterns make more sense when you understand transformer mechanics. Part 7's optimization techniques (quantization, KV cache) directly target transformer components.

---

### BENEFICIAL: Prompt Engineering Techniques

**What it is**: Practical skills for designing effective prompts—few-shot examples, chain-of-thought prompting, prompt templates, system messages, role-based prompting, and prompt debugging strategies.

**Why it matters**: Part 1 teaches agent architectures that use prompting. Part 2 teaches frameworks with prompt templates. Part 5 teaches advanced reasoning requiring sophisticated prompts. Better prompting skills accelerate your learning and lab success.

**Self-assessment**:
- Can you write a few-shot prompt that demonstrates a task to an LLM?
- Do you understand chain-of-thought prompting and when to use it?
- Can you debug a failing prompt systematically?
- Do you know the difference between system messages and user messages?

**If uncertain**:
- Complete: "ChatGPT Prompt Engineering for Developers" by DeepLearning.AI (2 hours)
- Read: OpenAI's prompt engineering guide
- Practice: Write 10-15 prompts for various tasks, iterate on failures
- Time needed: 3-5 hours

**Beneficial because**: The book teaches prompting where necessary, but strong prompting skills make labs easier. Part 5's Tree-of-Thought reasoning involves complex prompts. Part 7's guardrails use prompt engineering for safety.

---

## Part 4: Optional Accelerators

### BENEFICIAL: API Design and REST Principles

**What it is**: Understanding RESTful design principles, API versioning, authentication (OAuth, JWT), rate limiting, pagination, and idempotency.

**Why it matters**: Part 2 teaches tool integration with APIs. Part 4 architects microservices with REST endpoints. Part 7 deploys inference APIs. Strong API design skills help you architect better tool interfaces and debug integration failures.

**Self-assessment**:
- Can you design a RESTful API following best practices?
- Do you understand API authentication mechanisms?
- Can you explain rate limiting and why APIs implement it?
- Do you know what idempotency means and why it matters?

**If uncertain**:
- Read: "RESTful Web APIs" by Leonard Richardson
- Review: Best practices guides from Stripe, Twilio, or GitHub API docs
- Practice: Design APIs for 3-5 agent tools
- Time needed: 4-6 hours

**Beneficial because**: The book teaches REST basics, but advanced API design helps in Part 2 (tool integration) and Part 4 (deployment architecture).

---

### BENEFICIAL: Async Programming in Python

**What it is**: Understanding Python's `async`/`await` syntax, event loops, concurrent execution, and the difference between parallelism and concurrency.

**Why it matters**: Part 2 frameworks use async patterns for concurrent tool execution. Part 7 optimizes throughput with async processing. Part 8 implements async monitoring. If you're implementing agents (not just configuring frameworks), async is essential.

**Self-assessment**:
- Can you write an `async` function and use `await`?
- Do you understand the difference between `asyncio.gather()` and sequential execution?
- Can you explain when async provides benefits (I/O-bound tasks)?
- Do you know the difference between threads, processes, and async tasks?

**If uncertain**:
- Read: "Python Concurrency with asyncio" by Matthew Fowler (chapters 1-4)
- Complete: Real Python's "Async IO in Python" tutorial
- Practice: Write 3-5 async programs calling multiple APIs concurrently
- Time needed: 5-8 hours

**Beneficial because**: If you're only using frameworks, async is hidden. If you're implementing custom agent logic or optimizing performance, async becomes essential. Part 2.6 (Advanced Tool Integration) includes async examples.

---

### BENEFICIAL: CI/CD and DevOps Fundamentals

**What it is**: Understanding continuous integration/continuous deployment pipelines, automated testing, infrastructure as code, version control (git), and deployment automation (GitHub Actions, GitLab CI).

**Why it matters**: Part 4 teaches CI/CD for agent systems. Part 8 (Operations) implements monitoring in CI/CD pipelines. Production agents require automated testing, deployment, and rollback strategies.

**Self-assessment**:
- Can you explain what CI/CD achieves and why it matters?
- Do you understand automated testing in deployment pipelines?
- Can you use git for version control (commit, branch, merge)?
- Can you write a simple GitHub Actions workflow?

**If uncertain**:
- Read: "Continuous Delivery" by Jez Humble (chapters 1-3)
- Complete: GitHub Actions tutorial (official docs, 2 hours)
- Practice: Set up a CI/CD pipeline for a simple Python application
- Time needed: 6-10 hours

**Beneficial because**: Part 4 teaches CI/CD patterns. If you understand DevOps principles, you'll implement pipelines faster. If not, the book provides sufficient guidance.

---

## Self-Assessment Summary

Before starting this textbook, you should be able to answer "yes" to:

**Essential (must have all):**
- [ ] I understand how LLMs work (tokens, context windows, prompting)
- [ ] I can write and debug intermediate Python code
- [ ] I understand REST APIs and HTTP basics
- [ ] I can use command-line interfaces and write bash scripts
- [ ] I understand Docker containers and can build images

**Recommended (should have most):**
- [ ] I understand basic software architecture patterns
- [ ] I know database fundamentals (SQL, indexing, trade-offs)
- [ ] I can use Kubernetes basics (pods, services, deployments)
- [ ] I understand machine learning fundamentals and embeddings

**Beneficial (helpful but not required):**
- [ ] I understand distributed systems concepts
- [ ] I know GPU architecture and parallel processing basics
- [ ] I have experience with prompt engineering
- [ ] I can write async Python code
- [ ] I understand CI/CD and DevOps principles

**If you checked fewer than 4/5 essential items**: Invest 2-3 weeks in remediation before starting. The time invested now saves frustration later.

**If you checked 4-5 essential and 2+ recommended**: You're ready to begin. Reference external materials as needed.

**If you checked most items**: You have strong foundations. You'll progress quickly through material.

---

## Recommended Preparation Path

### Path 1: Minimal Preparation (2-3 weeks, 25-35 hours)

**Focus**: Only essential prerequisites for Part 1-2
1. Week 1: LLM fundamentals, Python basics, REST APIs (12-15 hours)
2. Week 2: Command line, Docker basics (10-12 hours)
3. Week 3: Review Part 1 front matter, start reading (3-8 hours)

**Outcome**: Ready to start Part 1, will reference docs frequently in Parts 3-4

---

### Path 2: Recommended Preparation (4-5 weeks, 40-60 hours)

**Focus**: Essential + recommended prerequisites
1. Weeks 1-2: Essential prerequisites from Path 1 (25-30 hours)
2. Week 3: Architecture patterns, database fundamentals (10-15 hours)
3. Week 4: Kubernetes basics, ML fundamentals (12-18 hours)
4. Week 5: Review all Part front matters, start Part 1 (3-8 hours)

**Outcome**: Strong foundation for all 10 parts, smooth progression

---

### Path 3: Comprehensive Preparation (6-8 weeks, 60-90 hours)

**Focus**: Essential + recommended + beneficial
1. Weeks 1-3: Complete Path 2 (45-60 hours)
2. Week 4: Distributed systems, GPU architecture (10-15 hours)
3. Week 5: NLP fundamentals, prompt engineering (8-12 hours)
4. Week 6: Async Python, CI/CD basics (10-15 hours)
5. Weeks 7-8: Review, practice labs, start Part 1 (5-10 hours)

**Outcome**: Comprehensive foundation, rapid progression through all parts

---

## Resources for Remediation

### LLM and ML Fundamentals
- **Free**: Andrew Ng's "Machine Learning" (Coursera)
- **Free**: "ChatGPT Prompt Engineering for Developers" (DeepLearning.AI)
- **Book**: "Hands-On Machine Learning" by Aurélien Géron
- **Paper**: "Attention Is All You Need" (transformer architecture)

### Python Programming
- **Free**: Python for Everybody (Coursera)
- **Free**: Real Python tutorials
- **Book**: "Fluent Python" by Luciano Ramalho
- **Practice**: LeetCode Easy problems (data structures)

### Infrastructure and Tools
- **Free**: Docker "Getting Started" tutorial
- **Free**: Kubernetes "Learn Kubernetes Basics"
- **Book**: "The Linux Command Line" by William E. Shotts
- **Book**: "Docker Deep Dive" by Nigel Poulton

### Software Engineering
- **Book**: "Designing Data-Intensive Applications" by Martin Kleppmann
- **Book**: "Software Architecture in Practice" by Len Bass
- **Free**: System design interview resources (GitHub repos)

---

## Final Recommendations

**For Academic Students**: You likely have strong ML/programming foundations but may lack infrastructure experience. Focus on Docker, Kubernetes, and command-line skills (Path 2, Weeks 3-4).

**For Software Engineers**: You likely have strong infrastructure foundations but may lack ML/LLM experience. Focus on LLM fundamentals, embeddings, and prompt engineering (Path 2, Weeks 1-2).

**For ML Practitioners**: You likely have strong ML foundations but may lack production deployment experience. Focus on Docker, Kubernetes, CI/CD, and distributed systems (Path 3, Weeks 4-6).

**For Career Changers**: You may need comprehensive preparation across all categories. Follow Path 3 systematically, investing 60-90 hours before starting the textbook.

**Key Insight**: The textbook teaches agentic AI comprehensively, but it assumes you bring foundational knowledge. Time invested in prerequisites is not wasted—it's amplified throughout 138-180 hours of textbook study. Start with strong foundations, and every chapter becomes clearer, faster, and more rewarding.
