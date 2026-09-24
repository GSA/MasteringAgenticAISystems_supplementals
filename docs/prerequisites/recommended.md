---
title: Recommended
parent: Prerequisites
nav_order: 2
permalink: /prerequisites/recommended/
---

# Recommended prerequisites
{: .no_toc }

You can succeed without these, but expect to reference outside material often if you skip them.
See [Essential]({{ site.baseurl }}/prerequisites/essential/) for what you must have first, and
[Prerequisites]({{ site.baseurl }}/prerequisites/) for the tier overview.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Software architecture patterns

[1.5A/1.5B]({{ site.baseurl }}/curriculum/part-01/) on orchestration patterns is, underneath, asking
which software architecture fits a given multi-agent problem. Recognizing the patterns first makes
that chapter feel like applying something you already know, not memorizing a new taxonomy.

### Monolith vs. microservices, and loose coupling
Can you explain the difference between a monolithic and a microservices architecture, and what
"loose coupling" buys you?

- [Software Architecture in Practice](https://www.oreilly.com/library/view/software-architecture-in/9780132942799/) by Len Bass, chapters 1–3

### Pub-sub messaging and event-driven design
Can you describe publish-subscribe messaging and identify when it's the right choice over a direct
call?

- [System Design Primer](https://github.com/donnemartin/system-design-primer) &middot; free, community-maintained reference covering pub-sub, load balancing, and replication

### Architectural trade-offs in practice
For a system with 200 agents coordinating warehouse operations, would you reach for centralized or
decentralized orchestration, and why? Real case studies sharpen this instinct faster than
definitions do.

- Read a real architecture case study (Uber, Netflix, or AWS engineering blogs) and identify which
  patterns above it actually uses

---

## Database fundamentals

[1.4]({{ site.baseurl }}/curriculum/part-01/) (memory systems) requires understanding persistence, and
[Part 6]({{ site.baseurl }}/curriculum/part-06/) (RAG and knowledge integration) runs on vector
databases specifically — [6.2A]({{ site.baseurl }}/curriculum/part-06/) assumes you already know SQL,
indexing, and ACID before it explains why vector databases relax some of those guarantees.

### SQL basics
Can you write a `SELECT` query with a `WHERE` clause and a `JOIN`?

- [SQL Tutorial](https://www.w3schools.com/sql/) &middot; W3Schools, ~3–4 hours, free

### Indexing and query performance
Do you understand what a database index does and why it speeds up some queries but slows down
writes?

- [Designing Data-Intensive Applications](https://dataintensive.net/) by Martin Kleppmann, chapters 1–3

### ACID vs. eventual consistency
Can you name the four ACID properties, and explain why a distributed system might deliberately give
one of them up?

- Same source as above; practice by setting up PostgreSQL locally and running a transaction that
  violates isolation on purpose

### SQL vs. NoSQL vs. vector databases
Can you compare a relational database (PostgreSQL), a document store (MongoDB), and a vector
database (Pinecone) on what each is actually optimized for?

- Covered from the RAG side in [6.2A: Vector Database Selection]({{ site.baseurl }}/curriculum/part-06/)

---

## Kubernetes fundamentals

[Part 4]({{ site.baseurl }}/curriculum/part-04/) (production deployment and scaling) runs on Kubernetes
throughout, particularly [4.3]({{ site.baseurl }}/curriculum/part-04/) (container orchestration). If
you're deploying to a managed Kubernetes service (EKS, GKE, AKS), this moves from recommended to
essential.

### Pods, services, and deployments
Can you explain what a pod is, why it's the smallest deployable unit, and how a service exposes a
set of pods to the network?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/X48VuDVv0do" title="Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=X48VuDVv0do">Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]</a> &middot; TechWorld with Nana</div>
</div>

A natural next step after the Docker video in
[Essential &gt; Docker]({{ site.baseurl }}/prerequisites/essential/#docker-and-containerization-basics)
— same host channel style, picks up where containers leave off.

</details>

### kubectl and resource limits
Can you use `kubectl` to view pods, services, and logs, and explain what a CPU/memory resource
request vs. limit does?

- Practice: deploy a simple application to Minikube, scale its replica count, and trigger a rolling
  update

---

## Natural language processing (NLP) basics

LLMs are NLP models underneath. [Part 5]({{ site.baseurl }}/curriculum/part-05/)'s advanced reasoning
patterns build on transformer mechanics, and [Part 7]({{ site.baseurl }}/curriculum/part-07/)'s
optimization techniques (quantization, KV cache) target transformer components directly. This tier
goes one level deeper than [Essential &gt; LLM Fundamentals]({{ site.baseurl }}/prerequisites/essential/#large-language-model-llm-fundamentals) —
read that first.

### Tokenization
Can you explain what tokenization does, why a word can become multiple tokens, and why that matters
for cost and context-window budgeting?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/zduSFxRajkE" title="Let's build the GPT Tokenizer" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let's build the GPT Tokenizer</a> &middot; Andrej Karpathy</div>
</div>

Hands-on and code-level — build the mechanics yourself rather than take them on faith. For the
attention mechanism and full transformer architecture, see
[Essential &gt; LLM Fundamentals]({{ site.baseurl }}/prerequisites/essential/#large-language-model-llm-fundamentals)
rather than duplicating those videos here.

</details>

### Why transformers replaced RNNs
Do you know what made transformer models different from — and generally better than — the
recurrent networks (RNNs) that came before them?

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — read at least the introduction and
  the architecture diagram; this is the paper that made the case

### Common NLP task types
Can you name and distinguish a few standard NLP task shapes — classification, generation,
question answering — and place "agent tool-calling" among them?

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course), chapters 1–2, free

---

Continue to [Beneficial]({{ site.baseurl }}/prerequisites/beneficial/) prerequisites, back to
[Essential]({{ site.baseurl }}/prerequisites/essential/), or to the
[tier overview]({{ site.baseurl }}/prerequisites/).
