---
title: Video link check
nav_exclude: true
permalink: /videos/link-check/
---

# Video link check
{: .no_toc }

Result of checking every YouTube link in [`videos/`]({{ site.repo_tree }}/videos) on 2026-09-21, for maintainers and for anyone picking up the
[video library track]({% link contributing.md %}). It explains why some links in the source files are not embedded on the
[Part pages]({% link curriculum/index.md %}).

**Method.** Each link was sent to YouTube's oEmbed endpoint, which reports whether a video or playlist exists and may be
embedded, and returns its real title. A link whose title shares at least half its meaningful words with the entry's title
was accepted; the rest were compared by hand. This is a title check only, and results can change over time.

| Result | Unique links |
|---|---:|
| Available | 229 |
| Not found (removed, private, or wrong ID) | 36 |
| Embedding disabled by the uploader | 2 |
| **Total YouTube links checked** | **267** |

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Not found

38 entries point at a video that YouTube reports as not found. Some IDs may never have existed.

| Chapter | Entry in the source file | Link |
|---|---|---|
| 3.1A | Detecting Hallucinations in Large Language Models Using Semantic Entropy | `https://www.youtube.com/watch?v=15I5rna-gag` |
| 3.1A | Do Androids Know They're Only Dreaming of Electric Sheep? - Hallucination Detection Paper Review | `https://www.youtube.com/watch?v=YkLRGl8wZTM` |
| 3.1A | Introduction to RAG (Retrieval-Augmented Generation) \| LlamaIndex | `https://www.youtube.com/watch?v=A4U5CwcXr0I` |
| 3.1B | Build a Knowledge Graph with LLMs | `https://www.youtube.com/watch?v=qks4UD7q-oE` |
| 3.1B | Claim Verification with Knowledge Bases | `https://www.youtube.com/watch?v=MJ1hOqEfMcU` |
| 3.1B | Corrective RAG (CRAG) Implementation | `https://www.youtube.com/watch?v=EvlPm5iZZjc` |
| 3.1B | Entity Linking and Knowledge Graphs | `https://www.youtube.com/watch?v=H6CgSz4q6wI` |
| 3.1B | Evaluating RAG Systems - DeepLearning.AI | `https://www.youtube.com/watch?v=pGtVWFHHqT0` |
| 3.1B | Multi-Query Retrieval for Better RAG | `https://www.youtube.com/watch?v=SuI7j-hKG-U` |
| 3.1B | RAG++ - Advanced RAG with LlamaIndex | `https://www.youtube.com/watch?v=vIJz4I1vjSE` |
| 3.1B | RAGAS - Evaluation Framework for RAG | `https://www.youtube.com/watch?v=6Q-PH04F_Ow` |
| 3.1C | Audio Processing with Transformers | `https://www.youtube.com/watch?v=9B3A1uPdB_s` |
| 3.1C | Document AI and Intelligent Document Processing | `https://www.youtube.com/watch?v=WHzgz5-W8TU` |
| 3.1C | Evaluating Audio Transcription Quality | `https://www.youtube.com/watch?v=mRB8tbTkkrA` |
| 3.1C | Evaluating Vision-Language Models | `https://www.youtube.com/watch?v=bAcKe2xJdUU` |
| 3.1C | Image Captioning and Visual Grounding | `https://www.youtube.com/watch?v=1h8hAp_HYdU` |
| 3.1C | Multi-Modal Hallucination Detection | `https://www.youtube.com/watch?v=K8R3lZ8X5aE` |
| 3.1C | OCR and Document Understanding with LayoutLM | `https://www.youtube.com/watch?v=K9nzP_vqqns` |
| 3.2 | Dense Passage Retrieval (DPR) for Open-Domain QA | `https://www.youtube.com/watch?v=NUMg4e74Sz4` |
| 3.2 | Evaluating RAG Systems with RAGAS | `https://www.youtube.com/watch?v=6Q-PH04F_Ow` |
| 3.3 | Constitutional AI and Harmlessness Training | `https://www.youtube.com/watch?v=KAgKqMqOMl0` |
| 3.3 | HELM - Holistic Evaluation of Language Models (Stanford) | `https://www.youtube.com/watch?v=6UuCWdR5iHo` |
| 3.3 | Jailbreaking LLMs and Defense Mechanisms | `https://www.youtube.com/watch?v=ov7FbEi1g9E` |
| 3.3 | Red Teaming Language Models - Anthropic Research | `https://www.youtube.com/watch?v=Joz3s6V9fK0` |
| 3.3 | Toxicity Detection in Text - Perspective API | `https://www.youtube.com/watch?v=VoYNwpj7VHE` |
| 3.4 | Memory and State Management in LangChain Agents | `https://www.youtube.com/watch?v=SyU60dr4H3Q` |
| 3.4 | Persona Consistency in Conversational AI | `https://www.youtube.com/watch?v=3ywZqROGdqk` |
| 3.6 | AgentBench - Benchmarking LLMs as Agents | `https://www.youtube.com/watch?v=lREQzTVJbIY` |
| 3.6 | Evaluating Code Generation with HumanEval | `https://www.youtube.com/watch?v=i8wpLm2j0I0` |
| 3.6 | Evaluating Conversational AI Agents | `https://www.youtube.com/watch?v=vN0qKNrJl_Q` |
| 3.6 | Evaluating Tool Use in Language Models | `https://www.youtube.com/watch?v=kqm8pNGX96k` |
| 3.6 | HELM - Holistic Evaluation of Language Models | `https://www.youtube.com/watch?v=6UuCWdR5iHo` |
| 3.6 | LLM Evaluation Fundamentals - DeepLearning.AI | `https://www.youtube.com/watch?v=gsf_rMZJWZQ` |
| 3.6 | LangSmith for Agent Evaluation and Monitoring | `https://www.youtube.com/watch?v=pG_PNcdukUw` |
| 3.6 | Multi-Task Evaluation of Language Models | `https://www.youtube.com/watch?v=gEZrGsRMK4k` |
| 3.6 | SWE-bench - Software Engineering Benchmark for Agents | `https://www.youtube.com/watch?v=8JF-pL2IvKo` |
| 4.2 | MLOps Tutorial - Building End-to-End Machine Learning Pipeline | `https://www.youtube.com/watch?v=hmkF77F9TLw` |
| 5.10 | Law of Diminishing Marginal Utility | `https://www.youtube.com/watch?v=xgLx9CBnm9U` |

## Available, but not the video the entry describes

55 entries point at a video that exists but whose title does not match the entry, judged by title only. They are not
embedded. A few may still be worth keeping under a corrected title.

| Chapter | Entry in the source file | Actual title on YouTube | Link |
|---|---|---|---|
| 1.1B | Accessibility Navigation Tutorial | Why you shouldn't use a div for everything - creating accessible buttons and navigations | `https://www.youtube.com/watch?v=YAqRQoN8ykI` |
| 1.2 | AI Agent Tool Calling Example | [1hr Talk] Intro to Large Language Models | `https://www.youtube.com/watch?v=zjkBMFhNj_g` |
| 1.2 | Advanced LangChain Concepts (James Briggs) | Prompt Templates for GPT 3.5 and other LLMs - LangChain #2 | `https://www.youtube.com/watch?v=RflBcK0oDH0` |
| 1.2 | LangChain Agents Tutorial (James Briggs) | Getting Started with GPT-3 vs. Open Source LLMs - LangChain #1 | `https://www.youtube.com/watch?v=nE2skSRWTTs` |
| 1.3 | Ballerina gRPC Introduction | Create Your First Service With Ballerina | `https://www.youtube.com/watch?v=Kk4tbN8FcZ4` |
| 1.6 | LangGraph Checkpointer - Game-Changer for AI Agents | Is Marriage Necessary?  Strangers Discuss Love  Life Fate in a Beachside Cafe\| AI Film - Romance | `https://youtu.be/0yISjksQ8as` |
| 1.7A | Named Entity Recognition with spaCy | How to Separate Sentences in SpaCy (SpaCy and Python Tutorials for DH - 03) | `https://www.youtube.com/watch?v=ytAyCO-n8tY` |
| 2.2 | Query Routing with LangGraph | RAG from scratch: Part 10 (Routing) | `https://youtu.be/pfpIndq7Fi8` |
| 2.4 | AutoGen Advanced Patterns | A Friendly Introduction to AutoGen Studio v0.4 (UI for Building AI Agents with AutoGen) | `https://youtu.be/oum6EI7wohM` |
| 2.6 | LangChain Tool Use and Integration | The LangChain Cookbook - Beginner Guide To 7 Essential Concepts | `https://www.youtube.com/watch?v=2xxziIWmaSA` |
| 2.7 | LangChain RAG Tutorial | LangGraph Complete Course for Beginners – Complex AI Agents with Python | `https://www.youtube.com/watch?v=jGg_1h0qzaM` |
| 2.8 | Building Robust AI Applications with Error Handling | LangGraph: Intro | `https://www.youtube.com/watch?v=5h-JBkySK34` |
| 2.8 | LangGraph Error Recovery Patterns | How to add short-term memory to LangGraph ReAct agent🤖: Python & Node.js — LangGraph #2 | `https://youtu.be/GMaGG8UBek8` |
| 2.9 | Streaming with LangChain | LangGraph Complete Course for Beginners – Complex AI Agents with Python | `https://www.youtube.com/watch?v=jGg_1h0qzaM` |
| 3.1B | Advanced RAG Techniques - Greg Kamradt | Building Production-Ready RAG Applications: Jerry Liu | `https://www.youtube.com/watch?v=TRjq7t2Ms5I` |
| 3.1C | LLaVA - Large Language and Vision Assistant | [CVPR2023 Tutorial Talk] Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4 | `https://www.youtube.com/watch?v=mkI7EPD1vp8` |
| 3.1C | Visual Question Answering with Vision Transformers | Visualizing weights & intermediate layer outputs of CNN in Keras | `https://www.youtube.com/watch?v=5tW3y7lm7V0` |
| 3.1C | Whisper - Robust Speech Recognition | How to Install & Use Whisper AI Voice to Text | `https://www.youtube.com/watch?v=ABFqbY_rmEk` |
| 3.2 | Advanced RAG Techniques - Hypothetical Document Embeddings (HyDE) | Embeddings: What they are and why they matter | `https://www.youtube.com/watch?v=ArnMdc-ICCM` |
| 3.6 | Building and Evaluating AI Agents - Andrew Ng | What's next for AI agentic workflows ft. Andrew Ng of AI Fund | `https://www.youtube.com/watch?v=sal78ACtGTc` |
| 3.7 | LangChain Tutorial Series - Tools and Functions | What Is LangChain? - LangChain + ChatGPT Overview | `https://www.youtube.com/watch?v=_v_fgW2SkkQ` |
| 3.8 | RAG Components & Troubleshooting with Arize Phoenix | LLM Search & Retrieval Systems with Arize and LlamaIndex: Powering LLMs on Your Proprietary Data | `https://youtube.com/watch?v=hbQYDpJayFw` |
| 3.9 | PyReason - Neuro-Symbolic AI | Trajectory Generation via Abductive Inference (ICLP talk) | `https://www.youtube.com/watch?v=8nxuIaTpZzM` |
| 4.1 | MLOps and Continuous Delivery for Machine Learning | Machine Learning, Technical Debt, and You - D. Sculley (Google) | `https://www.youtube.com/watch?v=V18AsBIHlWs` |
| 4.1 | RabbitMQ Crash Course | How Discord Stores Trillions of Messages \| Deep Dive | `https://www.youtube.com/watch?v=xynXjChKkJc` |
| 4.2 | API Gateway Explained | Distributed Systems in One Lesson by Tim Berglund | `https://www.youtube.com/watch?v=Y6Ev8GIlbxc` |
| 4.2 | Kafka vs RabbitMQ | Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours] | `https://www.youtube.com/watch?v=X48VuDVv0do` |
| 4.2 | Kubernetes Deployment Strategies (Blue-Green, Canary, Rolling) | Free Hosting for Python Scripts on Google Cloud | `https://www.youtube.com/watch?v=5OL7fu2R4M8` |
| 4.2 | Microservice Architecture and System Design with Python & Kubernetes | Microservices explained - the What, Why and How? | `https://www.youtube.com/watch?v=rv4LlmLmVWk` |
| 4.2 | RabbitMQ Crash Course | How Prometheus Monitoring works \| Prometheus Architecture explained | `https://www.youtube.com/watch?v=h4Sl21AKiDg` |
| 4.2 | Serverless Architecture Explained | Mastering Chaos - A Netflix Guide to Microservices | `https://www.youtube.com/watch?v=CZ3wIuvmHeM` |
| 4.3 | MIT 6.S965 - Pruning and Sparsity in Neural Networks | Tensorflow and deep learning - without a PhD by Martin Görner | `https://www.youtube.com/watch?v=vq2nnJ4g6N0` |
| 4.3 | Quantization: Optimize AI Models to Run Everywhere | Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training | `https://www.youtube.com/watch?v=0VdNflU08yA` |
| 4.5 | vLLM Deployment with Hugging Face | The KV Cache: Memory Usage in Transformers | `https://www.youtube.com/watch?v=80bIUggRJf4` |
| 4.6 | How vLLM Optimizes the LLM Serving System | The KV Cache: Memory Usage in Transformers | `https://www.youtube.com/watch?v=80bIUggRJf4` |
| 4.6 | PyTorch Distributed Data Parallel (DDP) | DL4CV@WIS (Spring 2021) Tutorial 13: Training with Multiple GPUs | `https://www.youtube.com/watch?v=TibQO_xv1zc` |
| 4.7 | Complete Kubernetes Tutorial (Playlist) | What is Kubernetes \| Kubernetes explained in 15 mins | `https://www.youtube.com/watch?v=VnvRFRk_51k` |
| 5.4 | AI-Driven Workflows and Parallel Agent Management | The creator of OpenClaw: "I ship code I don't read" | `https://www.youtube.com/watch?v=8lF7HmQ_RgY` |
| 5.4 | Building Content Creator Agents with CrewAI | How To Create Ai Agents From Scratch (CrewAI, Zapier, Cursor) | `https://www.youtube.com/watch?v=PM9zr7wgJX4` |
| 5.4 | Computational Thinking - Problem Decomposition | 139. OCR A Level (H446) SLR24 - 2.2 Features of a problem | `https://www.youtube.com/watch?v=r2c_SfdEQ84` |
| 5.4 | HTN Planning Tutorial Series - Video 1 | Hierarchical Task Network Planning - Georgia Tech - KBAI: Part 3 | `https://www.youtube.com/watch?v=7L3tcoFMR7w` |
| 5.4 | HTN Planning Tutorial Series - Video 2 | Invited Talk at HPlan 2020: HTN Planning: Interactions of Theory and Practice (Robert Goldman) | `https://www.youtube.com/watch?v=MypF9_5wvlM` |
| 5.4 | HTN Planning Tutorial Series - Video 4 | HTN Planning in Transformers: Fall of Cybertron \| AI and Games #14 | `https://www.youtube.com/watch?v=kXm467TFTcY` |
| 5.4 | HTN Planning Tutorial Series - Video 5 | The AI of Horizon Zero Dawn \| Part 1: Rise of the Machines \| AI and Games #37 | `https://www.youtube.com/watch?v=XxuSFBVQULY` |
| 5.4 | Production Research Agent with Graph Workflows | How to Build an Advanced AI Agent with Search (LangGraph, Python, Bright Data & More) | `https://www.youtube.com/watch?v=cUC-hyjpNxk` |
| 5.5 | AlphaZero Paper Explained | AlphaGo - How AI mastered the hardest boardgame in history | `https://www.youtube.com/watch?v=MgowR4pq3e8` |
| 5.6 | A\* Algorithm in Action | My Pathfinding Visualizer Project | `https://www.youtube.com/watch?v=O0MvdhQdj6I` |
| 5.7 | Neo4j Quick Start | Quickly create example graph data for Neo4j using Arrows | `https://www.youtube.com/watch?v=fkD1agLtQ4I` |
| 5.9 | vLLM Optimization | The KV Cache: Memory Usage in Transformers | `https://www.youtube.com/watch?v=80bIUggRJf4` |
| 5.12 | Multi-Agent RL | CSL seminar: Jakob Foerster​​ | `https://www.youtube.com/watch?v=ii_SwIsY8aU` |
| 5.12 | Multi-Agent RL Logic | 1v10 AI Dodgeball (deep reinforcement learning) | `https://www.youtube.com/watch?v=8fICnUvIw6g` |
| 5.12 | Thompson Sampling for RL | Coordinated Exploration in Concurrent Reinforcement Learning | `https://www.youtube.com/watch?v=xjGK-wm0PkI` |
| 6.2B | Weaviate Tutorial | How to Build a Recommendation System with AI and Semantic Search | `https://www.youtube.com/watch?v=SF1ZlRjVsxw` |
| 9.1 | LLM Guardrails with Llama Guard 3 Vision | Getting Started with watsonx and Watson Machine Learning | `https://www.youtube.com/watch?v=3sav6vUG_XQ` |
| 10.4 | Tornado Human-in-the-Loop ML Tool Demo | Human in the Loop Machine Learning (HITL) manufacturing Image labelling | `https://www.youtube.com/watch?v=zBe6b_vxs_I` |

## Embedding disabled

2 entries point at a video or playlist that exists but cannot be embedded. They are listed as plain links under the
chapter.

| Chapter | Entry in the source file | Link |
|---|---|---|
| 5.12 | Proximal Policy Optimization (PPO) Explained | `https://www.youtube.com/watch?v=vQ_ifavFBkI` |
| 5.13 | Stanford CS229 Machine Learning | `https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU` |
