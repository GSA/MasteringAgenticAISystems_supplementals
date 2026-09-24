---
title: Beneficial
parent: Prerequisites
nav_order: 3
permalink: /prerequisites/beneficial/
---

# Beneficial prerequisites
{: .no_toc }

These accelerate learning and deepen understanding but aren't strictly required — the book explains
what you need when you get there. Worth prioritizing if you're architecting multi-agent or
enterprise systems rather than single-agent ones. See [Essential]({{ site.baseurl }}/prerequisites/essential/)
and [Recommended]({{ site.baseurl }}/prerequisites/recommended/) for the tiers above this one.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Distributed systems concepts

[1.3]({{ site.baseurl }}/curriculum/part-01/) (multi-agent systems) and
[1.5A–1.6]({{ site.baseurl }}/curriculum/part-01/) (stateful orchestration) center on multi-agent
coordination across processes, servers, or organizations — exactly the setting where
assuming instant, perfectly-ordered message delivery causes real bugs. If you're only building
single-agent systems, you can defer this tier entirely.

### Synchronous vs. asynchronous messaging, and failure recovery
If three agents are coordinating on a task and the network drops for five seconds, how should the
system recover? Can you explain the difference between synchronous and asynchronous messaging?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/videoseries?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB" title="Distributed Systems lecture series" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/playlist?list=PLeKd45zvjcDFUEv_ohr_HdUFe97RItdiB">Distributed Systems lecture series</a> &middot; Martin Kleppmann</div>
</div>

~7-hour playlist; the same author as *Designing Data-Intensive Applications* (cited in
[Recommended &gt; Database Fundamentals]({{ site.baseurl }}/prerequisites/recommended/#database-fundamentals)).
The first 2–3 lectures cover this and the next subtopic.

</details>

### Eventual consistency, CAP theorem, and consensus
Do you understand what "eventual consistency" means and why distributed systems accept it? At a
high level, what problem do consensus protocols like Raft or Paxos solve?

- Continue the same lecture series above, or read the original
  [CAP Theorem](https://www.julianbrowne.com/article/brewers-cap-theorem) explainer for the
  condensed version

---

## GPU architecture and CUDA basics

[Part 7]({{ site.baseurl }}/curriculum/part-07/) optimizes LLM inference on GPUs — quantization,
batching, and KV-cache tuning all make more sense once you know why GPUs are fast at the operations
LLMs need. [4.4]({{ site.baseurl }}/curriculum/part-04/) profiles GPU utilization directly. You can
learn Part 7's techniques without this — the book explains what you need — but it moves from recipe
to intuition with this background.

### Why GPUs are fast for LLM workloads
Can you explain why GPUs excel at matrix multiplication specifically, and what GPU utilization
percentages actually indicate?

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html),
  introduction — read for the architecture rationale, not to become a CUDA programmer

### Profiling GPU workloads in practice
Once you understand the "why," can you read a GPU profiler's output and identify a memory- vs.
compute-bound kernel?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/LuhJEEJQgUM" title="Lecture 1: How to profile CUDA kernels in PyTorch" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=LuhJEEJQgUM">Lecture 1: How to profile CUDA kernels in PyTorch</a> &middot; GPU MODE</div>
</div>

More hands-on than a pure fundamentals video — useful once the "why" above makes sense, less useful
before it.

</details>

---

## Prompt Engineering Techniques

[Part 1]({{ site.baseurl }}/curriculum/part-01/) and every framework in [Part 2]({{ site.baseurl }}/curriculum/part-02/)
use prompting; [5.2]({{ site.baseurl }}/curriculum/part-05/)'s Tree-of-Thought reasoning depends on
sophisticated prompt construction. Better prompting skill makes every lab in the book faster. This
extends the brief mention in
[Essential &gt; LLM Fundamentals &gt; Prompting paradigms]({{ site.baseurl }}/prerequisites/essential/#large-language-model-llm-fundamentals).

### Zero-shot, few-shot, and chain-of-thought prompting
Can you write a few-shot prompt that demonstrates a task to a model, and explain when
chain-of-thought prompting is worth the extra tokens?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/b2t4pa9lOIc" title="Prompt Engineering Explained – Zero, Few, CoT, Self Consistency, Persona" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=b2t4pa9lOIc">Prompt Engineering Explained – Zero, Few, CoT, Self Consistency, Persona</a> &middot; Loop Kaka</div>
</div>

</details>

### System messages, templates, and role-based prompting
Do you know the practical difference between a system message and a user message, and can you build
a reusable prompt template?

- [OpenAI's prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) &middot; free, official, kept current

### Debugging a failing prompt
When a prompt doesn't produce what you expect, do you have a systematic way to isolate why — versus
guessing and rephrasing at random?

- Practice: take 10–15 prompts across different tasks, deliberately break each one, and diagnose why
  before fixing it

---

## Async programming in Python

If you're only configuring frameworks, async stays hidden. If you're implementing custom agent
logic, it's essential: [2.9]({{ site.baseurl }}/curriculum/part-02/) (streaming) uses async patterns
for concurrent tool execution, and [Part 7]({{ site.baseurl }}/curriculum/part-07/) optimizes
throughput with it.

### `async`/`await` and the event loop
Can you write an `async` function, use `await`, and explain what the event loop is actually doing
while your coroutine waits on I/O?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/ngXbyui-weA" title="Python Asynchronous Programming Tutorial: Asyncio, async &amp; await Explained" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=ngXbyui-weA">Python Asynchronous Programming Tutorial: Asyncio, async &amp; await Explained</a> &middot; Code with Josh</div>
</div>

</details>

### Concurrent execution and when async actually helps
Do you understand the difference between `asyncio.gather()` and running the same calls
sequentially, and why async helps I/O-bound work but not CPU-bound work?

- Practice: write a script that calls 3–5 APIs concurrently with `asyncio.gather()` and time it
  against calling them one at a time

### Threads vs. processes vs. async tasks
Can you explain when you'd reach for a thread, a process, or an async task instead of each other?

- [Real Python: Async IO in Python](https://realpython.com/async-io-python/)

---

## CI/CD and DevOps fundamentals

[Part 4]({{ site.baseurl }}/curriculum/part-04/) teaches CI/CD for agent systems directly, and
[Part 8]({{ site.baseurl }}/curriculum/part-08/) implements monitoring inside those pipelines.
Production agents need automated testing, deployment, and rollback — the book provides sufficient
guidance if you're new to this, but you'll move faster with the basics already in place.

### What CI/CD achieves
Can you explain what continuous integration and continuous deployment each solve, and why automated
testing has to run before a deploy, not after?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/aZzV6X7XhyI" title="Automate your Docker Build/Test/Deploy pipelines!" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=aZzV6X7XhyI">Automate your Docker Build/Test/Deploy pipelines!</a> &middot; Bret Fisher</div>
</div>

A concrete build/test/deploy pipeline, not just the theory — and containerized, which is how this
book deploys everything from [Part 4]({{ site.baseurl }}/curriculum/part-04/) onward.

</details>

### Git version control
Can you commit, branch, and merge confidently, including resolving a real conflict rather than
avoiding branches to avoid one?

- [Git documentation](https://git-scm.com/doc) &middot; official, or any interactive git-branching tutorial

### Writing a pipeline
Can you write a simple GitHub Actions workflow file from scratch — not copy one and hope?

- [GitHub Actions Quickstart](https://docs.github.com/en/actions/quickstart) &middot; official docs, ~2 hours

---

Back to [Recommended]({{ site.baseurl }}/prerequisites/recommended/), [Essential]({{ site.baseurl }}/prerequisites/essential/),
or the [tier overview]({{ site.baseurl }}/prerequisites/).
