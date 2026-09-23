---
title: Video link check
nav_exclude: true
permalink: /videos/link-check/
---

# Video link check
{: .no_toc }

On 2026-09-23, every YouTube link in [`videos/`]({{ site.repo_tree }}/videos) was checked against YouTube's public oEmbed
endpoint, which reports whether a video or playlist exists and may be embedded, and returns its real title. This page
records what that check found, for maintainers and for anyone picking up the
[video library track]({{ site.baseurl }}/contributing/).

**Method.** A link whose YouTube title shared at least half its meaningful words with the entry's title was accepted;
the rest were compared by hand. This is a title check only, and it will go stale as videos come and go — re-run it before
relying on it.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## What the check found, and what happened to it

| Result | Unique links | Entries | Action taken |
|---|---:|---:|---|
| Available and matched the entry | — | 261 | Kept, and embedded on the Part pages |
| Available, but embedding disabled by the uploader | 2 | 2 | Kept as a plain link |
| Available, but a different video from the one the entry described | — | 55 | **Removed** from `videos/` on 2026-09-23 |
| Not found (removed, private, or a wrong ID) | 36 | 38 | **Removed** from `videos/` on 2026-09-23 |

93 entries were removed in total. Four chapters lost their only video and now carry a short
"Note:" line instead, in the style already used elsewhere in these files for chapters with no direct links: 2.8, 3.1A,
6.2B, and 3.6 (which had ten entries, all ten of them wrong).

## Archive: removed entries

Kept for context — if you are curating new videos for one of these chapters, these are the ones that did not hold up,
so you can avoid proposing the same link again.

<details markdown="block">
<summary>Not found (38)</summary>

| Part | Entry (as it read in the source file) | Link | Reason |
|---|---|---|---|
| Part 3 | AgentBench - Benchmarking LLMs as Agents | `https://www.youtube.com/watch?v=lREQzTVJbIY` | not found (status 404) |
| Part 3 | Audio Processing with Transformers | `https://www.youtube.com/watch?v=9B3A1uPdB_s` | not found (status 404) |
| Part 3 | Build a Knowledge Graph with LLMs | `https://www.youtube.com/watch?v=qks4UD7q-oE` | not found (status 404) |
| Part 3 | Claim Verification with Knowledge Bases | `https://www.youtube.com/watch?v=MJ1hOqEfMcU` | not found (status 404) |
| Part 3 | Constitutional AI and Harmlessness Training | `https://www.youtube.com/watch?v=KAgKqMqOMl0` | not found (status 404) |
| Part 3 | Corrective RAG (CRAG) Implementation | `https://www.youtube.com/watch?v=EvlPm5iZZjc` | not found (status 404) |
| Part 3 | Dense Passage Retrieval (DPR) for Open-Domain QA | `https://www.youtube.com/watch?v=NUMg4e74Sz4` | not found (status 404) |
| Part 3 | Detecting Hallucinations in Large Language Models Using Semantic Entropy | `https://www.youtube.com/watch?v=15I5rna-gag` | not found (status 404) |
| Part 3 | Do Androids Know They're Only Dreaming of Electric Sheep? - Hallucination Detection Paper Review | `https://www.youtube.com/watch?v=YkLRGl8wZTM` | not found (status 404) |
| Part 3 | Document AI and Intelligent Document Processing | `https://www.youtube.com/watch?v=WHzgz5-W8TU` | not found (status 404) |
| Part 3 | Entity Linking and Knowledge Graphs | `https://www.youtube.com/watch?v=H6CgSz4q6wI` | not found (status 404) |
| Part 3 | Evaluating Audio Transcription Quality | `https://www.youtube.com/watch?v=mRB8tbTkkrA` | not found (status 404) |
| Part 3 | Evaluating Code Generation with HumanEval | `https://www.youtube.com/watch?v=i8wpLm2j0I0` | not found (status 404) |
| Part 3 | Evaluating Conversational AI Agents | `https://www.youtube.com/watch?v=vN0qKNrJl_Q` | not found (status 404) |
| Part 3 | Evaluating RAG Systems - DeepLearning.AI | `https://www.youtube.com/watch?v=pGtVWFHHqT0` | not found (status 404) |
| Part 3 | Evaluating RAG Systems with RAGAS | `https://www.youtube.com/watch?v=6Q-PH04F_Ow` | not found (status 404) |
| Part 3 | Evaluating Tool Use in Language Models | `https://www.youtube.com/watch?v=kqm8pNGX96k` | not found (status 404) |
| Part 3 | Evaluating Vision-Language Models | `https://www.youtube.com/watch?v=bAcKe2xJdUU` | not found (status 404) |
| Part 3 | HELM - Holistic Evaluation of Language Models | `https://www.youtube.com/watch?v=6UuCWdR5iHo` | not found (status 404) |
| Part 3 | HELM - Holistic Evaluation of Language Models (Stanford) | `https://www.youtube.com/watch?v=6UuCWdR5iHo` | not found (status 404) |
| Part 3 | Image Captioning and Visual Grounding | `https://www.youtube.com/watch?v=1h8hAp_HYdU` | not found (status 404) |
| Part 3 | Introduction to RAG (Retrieval-Augmented Generation) \| LlamaIndex | `https://www.youtube.com/watch?v=A4U5CwcXr0I` | not found (status 404) |
| Part 3 | Jailbreaking LLMs and Defense Mechanisms | `https://www.youtube.com/watch?v=ov7FbEi1g9E` | not found (status 404) |
| Part 3 | LLM Evaluation Fundamentals - DeepLearning.AI | `https://www.youtube.com/watch?v=gsf_rMZJWZQ` | not found (status 404) |
| Part 3 | LangSmith for Agent Evaluation and Monitoring | `https://www.youtube.com/watch?v=pG_PNcdukUw` | not found (status 404) |
| Part 3 | Memory and State Management in LangChain Agents | `https://www.youtube.com/watch?v=SyU60dr4H3Q` | not found (status 404) |
| Part 3 | Multi-Modal Hallucination Detection | `https://www.youtube.com/watch?v=K8R3lZ8X5aE` | not found (status 404) |
| Part 3 | Multi-Query Retrieval for Better RAG | `https://www.youtube.com/watch?v=SuI7j-hKG-U` | not found (status 404) |
| Part 3 | Multi-Task Evaluation of Language Models | `https://www.youtube.com/watch?v=gEZrGsRMK4k` | not found (status 404) |
| Part 3 | OCR and Document Understanding with LayoutLM | `https://www.youtube.com/watch?v=K9nzP_vqqns` | not found (status 404) |
| Part 3 | Persona Consistency in Conversational AI | `https://www.youtube.com/watch?v=3ywZqROGdqk` | not found (status 404) |
| Part 3 | RAG++ - Advanced RAG with LlamaIndex | `https://www.youtube.com/watch?v=vIJz4I1vjSE` | not found (status 404) |
| Part 3 | RAGAS - Evaluation Framework for RAG | `https://www.youtube.com/watch?v=6Q-PH04F_Ow` | not found (status 404) |
| Part 3 | Red Teaming Language Models - Anthropic Research | `https://www.youtube.com/watch?v=Joz3s6V9fK0` | not found (status 404) |
| Part 3 | SWE-bench - Software Engineering Benchmark for Agents | `https://www.youtube.com/watch?v=8JF-pL2IvKo` | not found (status 404) |
| Part 3 | Toxicity Detection in Text - Perspective API | `https://www.youtube.com/watch?v=VoYNwpj7VHE` | not found (status 404) |
| Part 4 | MLOps Tutorial - Building End-to-End Machine Learning Pipeline | `https://www.youtube.com/watch?v=hmkF77F9TLw` | not found (status 404) |
| Part 5 | Law of Diminishing Marginal Utility | `https://www.youtube.com/watch?v=xgLx9CBnm9U` | not found (status 404) |

</details>

<details markdown="block">
<summary>Wrong video (55)</summary>

| Part | Entry (as it read in the source file) | Link | Actual video |
|---|---|---|---|
| Part 1 | AI Agent Tool Calling Example | `https://www.youtube.com/watch?v=zjkBMFhNj_g` | wrong video (YouTube title: '[1hr Talk] Intro to Large Language Models') |
| Part 1 | Accessibility Navigation Tutorial | `https://www.youtube.com/watch?v=YAqRQoN8ykI` | wrong video (YouTube title: "Why you shouldn't use a div for everything - creating accessible buttons and navigations") |
| Part 1 | Advanced LangChain Concepts (James Briggs) | `https://www.youtube.com/watch?v=RflBcK0oDH0` | wrong video (YouTube title: 'Prompt Templates for GPT 3.5 and other LLMs - LangChain #2') |
| Part 1 | Ballerina gRPC Introduction | `https://www.youtube.com/watch?v=Kk4tbN8FcZ4` | wrong video (YouTube title: 'Create Your First Service With Ballerina') |
| Part 1 | LangChain Agents Tutorial (James Briggs) | `https://www.youtube.com/watch?v=nE2skSRWTTs` | wrong video (YouTube title: 'Getting Started with GPT-3 vs. Open Source LLMs - LangChain #1') |
| Part 1 | LangGraph Checkpointer - Game-Changer for AI Agents | `https://youtu.be/0yISjksQ8as` | wrong video (YouTube title: 'Is Marriage Necessary?  Strangers Discuss Love  Life Fate in a Beachside Cafe\| AI Film - Romance') |
| Part 1 | Named Entity Recognition with spaCy | `https://www.youtube.com/watch?v=ytAyCO-n8tY` | wrong video (YouTube title: 'How to Separate Sentences in SpaCy (SpaCy and Python Tutorials for DH - 03)') |
| Part 2 | AutoGen Advanced Patterns | `https://youtu.be/oum6EI7wohM` | wrong video (YouTube title: 'A Friendly Introduction to AutoGen Studio v0.4 (UI for Building AI Agents with AutoGen)') |
| Part 2 | Building Robust AI Applications with Error Handling | `https://www.youtube.com/watch?v=5h-JBkySK34` | wrong video (YouTube title: 'LangGraph: Intro') |
| Part 2 | LangChain RAG Tutorial | `https://www.youtube.com/watch?v=jGg_1h0qzaM` | wrong video (YouTube title: 'LangGraph Complete Course for Beginners – Complex AI Agents with Python') |
| Part 2 | LangChain Tool Use and Integration | `https://www.youtube.com/watch?v=2xxziIWmaSA` | wrong video (YouTube title: 'The LangChain Cookbook - Beginner Guide To 7 Essential Concepts') |
| Part 2 | LangGraph Error Recovery Patterns | `https://youtu.be/GMaGG8UBek8` | wrong video (YouTube title: 'How to add short-term memory to LangGraph ReAct agent🤖: Python & Node.js — LangGraph #2') |
| Part 2 | Query Routing with LangGraph | `https://youtu.be/pfpIndq7Fi8` | wrong video (YouTube title: 'RAG from scratch: Part 10 (Routing)') |
| Part 2 | Streaming with LangChain | `https://www.youtube.com/watch?v=jGg_1h0qzaM` | wrong video (YouTube title: 'LangGraph Complete Course for Beginners – Complex AI Agents with Python') |
| Part 3 | Advanced RAG Techniques - Greg Kamradt | `https://www.youtube.com/watch?v=TRjq7t2Ms5I` | wrong video (YouTube title: 'Building Production-Ready RAG Applications: Jerry Liu') |
| Part 3 | Advanced RAG Techniques - Hypothetical Document Embeddings (HyDE) | `https://www.youtube.com/watch?v=ArnMdc-ICCM` | wrong video (YouTube title: 'Embeddings: What they are and why they matter') |
| Part 3 | Building and Evaluating AI Agents - Andrew Ng | `https://www.youtube.com/watch?v=sal78ACtGTc` | wrong video (YouTube title: "What's next for AI agentic workflows ft. Andrew Ng of AI Fund") |
| Part 3 | LLaVA - Large Language and Vision Assistant | `https://www.youtube.com/watch?v=mkI7EPD1vp8` | wrong video (YouTube title: '[CVPR2023 Tutorial Talk] Large Multimodal Models: Towards Building and Surpassing Multimodal GPT-4') |
| Part 3 | LangChain Tutorial Series - Tools and Functions | `https://www.youtube.com/watch?v=_v_fgW2SkkQ` | wrong video (YouTube title: 'What Is LangChain? - LangChain + ChatGPT Overview') |
| Part 3 | PyReason - Neuro-Symbolic AI | `https://www.youtube.com/watch?v=8nxuIaTpZzM` | wrong video (YouTube title: 'Trajectory Generation via Abductive Inference (ICLP talk)') |
| Part 3 | RAG Components & Troubleshooting with Arize Phoenix | `https://youtube.com/watch?v=hbQYDpJayFw` | wrong video (YouTube title: 'LLM Search & Retrieval Systems with Arize and LlamaIndex: Powering LLMs on Your Proprietary Data') |
| Part 3 | Visual Question Answering with Vision Transformers | `https://www.youtube.com/watch?v=5tW3y7lm7V0` | wrong video (YouTube title: 'Visualizing weights & intermediate layer outputs of CNN in Keras') |
| Part 3 | Whisper - Robust Speech Recognition | `https://www.youtube.com/watch?v=ABFqbY_rmEk` | wrong video (YouTube title: 'How to Install & Use Whisper AI Voice to Text') |
| Part 4 | API Gateway Explained | `https://www.youtube.com/watch?v=Y6Ev8GIlbxc` | wrong video (YouTube title: 'Distributed Systems in One Lesson by Tim Berglund') |
| Part 4 | Complete Kubernetes Tutorial (Playlist) | `https://www.youtube.com/watch?v=VnvRFRk_51k` | wrong video (YouTube title: 'What is Kubernetes \| Kubernetes explained in 15 mins') |
| Part 4 | How vLLM Optimizes the LLM Serving System | `https://www.youtube.com/watch?v=80bIUggRJf4` | wrong video (YouTube title: 'The KV Cache: Memory Usage in Transformers') |
| Part 4 | Kafka vs RabbitMQ | `https://www.youtube.com/watch?v=X48VuDVv0do` | wrong video (YouTube title: 'Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]') |
| Part 4 | Kubernetes Deployment Strategies (Blue-Green, Canary, Rolling) | `https://www.youtube.com/watch?v=5OL7fu2R4M8` | wrong video (YouTube title: 'Free Hosting for Python Scripts on Google Cloud') |
| Part 4 | MIT 6.S965 - Pruning and Sparsity in Neural Networks | `https://www.youtube.com/watch?v=vq2nnJ4g6N0` | wrong video (YouTube title: 'Tensorflow and deep learning - without a PhD by Martin Görner') |
| Part 4 | MLOps and Continuous Delivery for Machine Learning | `https://www.youtube.com/watch?v=V18AsBIHlWs` | wrong video (YouTube title: 'Machine Learning, Technical Debt, and You - D. Sculley (Google)') |
| Part 4 | Microservice Architecture and System Design with Python & Kubernetes | `https://www.youtube.com/watch?v=rv4LlmLmVWk` | wrong video (YouTube title: 'Microservices explained - the What, Why and How?') |
| Part 4 | PyTorch Distributed Data Parallel (DDP) | `https://www.youtube.com/watch?v=TibQO_xv1zc` | wrong video (YouTube title: 'DL4CV@WIS (Spring 2021) Tutorial 13: Training with Multiple GPUs') |
| Part 4 | Quantization: Optimize AI Models to Run Everywhere | `https://www.youtube.com/watch?v=0VdNflU08yA` | wrong video (YouTube title: 'Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training') |
| Part 4 | RabbitMQ Crash Course | `https://www.youtube.com/watch?v=xynXjChKkJc` | wrong video (YouTube title: 'How Discord Stores Trillions of Messages \| Deep Dive') |
| Part 4 | RabbitMQ Crash Course | `https://www.youtube.com/watch?v=h4Sl21AKiDg` | wrong video (YouTube title: 'How Prometheus Monitoring works \| Prometheus Architecture explained') |
| Part 4 | Serverless Architecture Explained | `https://www.youtube.com/watch?v=CZ3wIuvmHeM` | wrong video (YouTube title: 'Mastering Chaos - A Netflix Guide to Microservices') |
| Part 4 | vLLM Deployment with Hugging Face | `https://www.youtube.com/watch?v=80bIUggRJf4` | wrong video (YouTube title: 'The KV Cache: Memory Usage in Transformers') |
| Part 5 | A\* Algorithm in Action | `https://www.youtube.com/watch?v=O0MvdhQdj6I` | wrong video (YouTube title: 'My Pathfinding Visualizer Project') |
| Part 5 | AI-Driven Workflows and Parallel Agent Management | `https://www.youtube.com/watch?v=8lF7HmQ_RgY` | wrong video (YouTube title: 'The creator of OpenClaw: "I ship code I don\'t read"') |
| Part 5 | AlphaZero Paper Explained | `https://www.youtube.com/watch?v=MgowR4pq3e8` | wrong video (YouTube title: 'AlphaGo - How AI mastered the hardest boardgame in history') |
| Part 5 | Building Content Creator Agents with CrewAI | `https://www.youtube.com/watch?v=PM9zr7wgJX4` | wrong video (YouTube title: 'How To Create Ai Agents From Scratch (CrewAI, Zapier, Cursor)') |
| Part 5 | Computational Thinking - Problem Decomposition | `https://www.youtube.com/watch?v=r2c_SfdEQ84` | wrong video (YouTube title: '139. OCR A Level (H446) SLR24 - 2.2 Features of a problem') |
| Part 5 | HTN Planning Tutorial Series - Video 1 | `https://www.youtube.com/watch?v=7L3tcoFMR7w` | wrong video (YouTube title: 'Hierarchical Task Network Planning - Georgia Tech - KBAI: Part 3') |
| Part 5 | HTN Planning Tutorial Series - Video 2 | `https://www.youtube.com/watch?v=MypF9_5wvlM` | wrong video (YouTube title: 'Invited Talk at HPlan 2020: HTN Planning: Interactions of Theory and Practice (Robert Goldman)') |
| Part 5 | HTN Planning Tutorial Series - Video 4 | `https://www.youtube.com/watch?v=kXm467TFTcY` | wrong video (YouTube title: 'HTN Planning in Transformers: Fall of Cybertron \| AI and Games #14') |
| Part 5 | HTN Planning Tutorial Series - Video 5 | `https://www.youtube.com/watch?v=XxuSFBVQULY` | wrong video (YouTube title: 'The AI of Horizon Zero Dawn \| Part 1: Rise of the Machines \| AI and Games #37') |
| Part 5 | Multi-Agent RL | `https://www.youtube.com/watch?v=ii_SwIsY8aU` | wrong video (YouTube title: 'CSL seminar: Jakob Foerster\u200b\u200b') |
| Part 5 | Multi-Agent RL Logic | `https://www.youtube.com/watch?v=8fICnUvIw6g` | wrong video (YouTube title: '1v10 AI Dodgeball (deep reinforcement learning)') |
| Part 5 | Neo4j Quick Start | `https://www.youtube.com/watch?v=fkD1agLtQ4I` | wrong video (YouTube title: 'Quickly create example graph data for Neo4j using Arrows') |
| Part 5 | Production Research Agent with Graph Workflows | `https://www.youtube.com/watch?v=cUC-hyjpNxk` | wrong video (YouTube title: 'How to Build an Advanced AI Agent with Search (LangGraph, Python, Bright Data & More)') |
| Part 5 | Thompson Sampling for RL | `https://www.youtube.com/watch?v=xjGK-wm0PkI` | wrong video (YouTube title: 'Coordinated Exploration in Concurrent Reinforcement Learning') |
| Part 5 | vLLM Optimization | `https://www.youtube.com/watch?v=80bIUggRJf4` | wrong video (YouTube title: 'The KV Cache: Memory Usage in Transformers') |
| Part 6 | Weaviate Tutorial | `https://www.youtube.com/watch?v=SF1ZlRjVsxw` | wrong video (YouTube title: 'How to Build a Recommendation System with AI and Semantic Search') |
| Part 9 | LLM Guardrails with Llama Guard 3 Vision | `https://www.youtube.com/watch?v=3sav6vUG_XQ` | wrong video (YouTube title: 'Getting Started with watsonx and Watson Machine Learning') |
| Part 10 | Tornado Human-in-the-Loop ML Tool Demo | `https://www.youtube.com/watch?v=zBe6b_vxs_I` | wrong video (YouTube title: 'Human in the Loop Machine Learning (HITL) manufacturing Image labelling') |

</details>

## Still present: embedding disabled

These 2 links are valid and point at the right video, but the uploader has disabled embedding, so they are shown as
plain links rather than players.

| Chapter | Entry in the source file | Link |
|---|---|---|
| 5.12 | Proximal Policy Optimization (PPO) Explained | `https://www.youtube.com/watch?v=vQ_ifavFBkI` |
| 5.13 | Stanford CS229 Machine Learning | `https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU` |
