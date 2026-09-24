---
title: Essential
parent: Prerequisites
nav_order: 1
permalink: /prerequisites/essential/
---

# Essential prerequisites
{: .no_toc }

You must be comfortable with all six of these before starting. Each is broken into the specific
sub-skills the book leans on, with a self-check question and one resource per sub-skill — a short
video where a good one exists, otherwise a course, book, or official guide. This expands on
[`Prerequisite_Knowledge.md`]({{ site.repo_blob }}/Prerequisite_Knowledge.md); see [Prerequisites]({{ site.baseurl }}/prerequisites/)
for the tier overview and study paths.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Large Language Model (LLM) fundamentals

Every agent architecture in this book uses an LLM as its reasoning engine. You'll design around
context limits, plan for inference latency, and choose models by reasoning capability — all of
which assume you already know how an LLM generates text. Load-bearing throughout: decoding
parameters (temperature, nucleus sampling) reappear in [5.2]({{ site.baseurl }}/curriculum/part-05/),
[5.3]({{ site.baseurl }}/curriculum/part-05/) and [6.6]({{ site.baseurl }}/curriculum/part-06/); context
windows and attention in [5.9]({{ site.baseurl }}/curriculum/part-05/); quantization and precision
depend on transformer internals throughout [Part 7]({{ site.baseurl }}/curriculum/part-07/).

### What a neural network is
Can you explain, in your own words, what a neural network computes and why training it is different
from running it?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/aircAruvnKk" title="But what is a neural network? | Deep learning chapter 1" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=aircAruvnKk">But what is a neural network? | Deep learning chapter 1</a> &middot; 3Blue1Brown</div>
</div>

</details>

### Transformer architecture
Can you sketch, at a block-diagram level, how a transformer turns a sequence of tokens into a
prediction for the next one?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/wjZofJX0v4M" title="Transformers, the tech behind LLMs | Deep Learning Chapter 5" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=wjZofJX0v4M">Transformers, the tech behind LLMs | Deep Learning Chapter 5</a> &middot; 3Blue1Brown</div>
</div>

</details>

### Attention and context windows
Can you explain what the attention mechanism computes, and why a longer context window costs more
compute?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/eMlx5fFNoYc" title="Attention in transformers, step-by-step | Deep Learning Chapter 6" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Attention in transformers, step-by-step | Deep Learning Chapter 6</a> &middot; 3Blue1Brown</div>
</div>

</details>

### Generation in practice: decoding, sampling, capabilities and limits
Can you explain the difference between greedy decoding and nucleus sampling and when you'd choose
each, and describe what temperature controls?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/zjkBMFhNj_g" title="[1hr Talk] Intro to Large Language Models" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a> &middot; Andrej Karpathy</div>
</div>

</details>

### Prompting paradigms
Do you know the practical difference between zero-shot, few-shot, and chain-of-thought prompting,
and when each is worth the extra tokens? (Depth on this is in [Prompt Engineering]({{ site.baseurl }}/prerequisites/beneficial/#prompt-engineering-techniques) below.)

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) &middot; DeepLearning.AI, ~2 hours, free

---

## Python programming

All code examples and every framework in the book (LangChain, LangGraph, AutoGen, CrewAI) are
Python. You'll read framework internals, implement agents, and debug failures — copying code you
don't understand won't get you through [Part 2]({{ site.baseurl }}/curriculum/part-02/),
[Part 7]({{ site.baseurl }}/curriculum/part-07/), or [Part 8]({{ site.baseurl }}/curriculum/part-08/).

### Functions and keyword arguments
Can you write a function that accepts keyword arguments and returns a dictionary?

- [Python for Everybody](https://www.coursera.org/specializations/python) &middot; University of Michigan (Coursera), weeks 1–3, free to audit

### Classes and object composition
Can you read class-based code and follow method calls, inheritance, and composition without
tracing through a debugger first?

- *Fluent Python* by Luciano Ramalho, chapters 1–4 (functions and objects as first-class citizens)

### Exception handling
Do you reach for `try`/`except`/`finally` naturally, or do you write code that assumes nothing
ever fails?

- [Python Exceptions: An Introduction](https://realpython.com/python-exceptions/) &middot; Real Python

### Nested data structures and comprehensions
Can you work with a dictionary containing a list of dictionaries, and write a list or dict
comprehension without reaching for a `for` loop first?

- Practice: implement 5–10 small programs manipulating nested JSON-shaped data (this is exactly
  the shape framework configs and tool responses take)

---

## REST APIs, HTTP, and API design

Agents call tools through APIs. Multi-agent communication, orchestration protocols, and every
production deployment in this book assume you're fluent in HTTP. Load-bearing in
[1.2]({{ site.baseurl }}/curriculum/part-01/) (tool-use architecture), [2.6]({{ site.baseurl }}/curriculum/part-02/)
(framework tool calling), [Part 4]({{ site.baseurl }}/curriculum/part-04/) (microservice deployment), and
[Part 7]({{ site.baseurl }}/curriculum/part-07/) (inference endpoints).

**Core** — needed from Part 1 onward:

### HTTP verbs and status codes
Can you explain the difference between GET and POST, and what a 4xx versus a 5xx status code
tells you about who's at fault?

- [MDN: An overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview) &middot; ~2–3 hours

### JSON and endpoint design
Can you parse a JSON response with nested objects, and design a REST endpoint for a tool — verb,
parameters, response shape?

- Practice: call 3–5 public APIs with Python's `requests` library and handle their responses

### Timeouts and failure modes
Do you understand what an API timeout is, why it matters for an agent calling a tool, and what
should happen when one fires?

- Covered practically in [2.8: Error Handling and Resilience]({{ site.baseurl }}/curriculum/part-02/)

**Going deeper** — matters once you're designing tool interfaces or deployment architecture, not
just calling existing APIs (Part 2 tool integration, Part 4 deployment):

### Authentication and rate limiting
Do you understand OAuth/JWT at a conceptual level, and why an API enforces rate limits?

- [Stripe API reference](https://docs.stripe.com/api) — read as a worked example of a well-designed,
  well-documented production API

### Idempotency and API gateways
Do you know what idempotency means and why it matters for retries, and what an API gateway adds
in front of a set of services?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/20rOdqag4Dw" title="Kong Gateway Tutorial | API Gateway For Beginners" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=20rOdqag4Dw">Kong Gateway Tutorial | API Gateway For Beginners</a> &middot; Redhwan Nacef</div>
</div>

</details>

---

## Command line and shell scripting

[Part 4]({{ site.baseurl }}/curriculum/part-04/) (deployment and scaling) and
[Part 8]({{ site.baseurl }}/curriculum/part-08/) (operations) assume terminal fluency for SSH access,
container management, `kubectl`, and infrastructure automation. Every deployment step in this book
involves a terminal.

### Filesystem navigation and search
Can you navigate a Linux filesystem and locate files with `find` or `grep` without opening a file
manager?

- [The Linux Command Line](https://linuxcommand.org/tlcl.php) by William E. Shotts, chapters 1–10 (free PDF)

### Bash scripting
Can you write a bash script with variables, a loop, and a conditional — say, one that processes
every file in a directory?

- Practice: write 3–5 small bash scripts against your own filesystem

### Process management and environment variables
Can you manage a background process, redirect its output to a file, and set/read an environment
variable with `export`?

- Same source as above, chapters on job control and environment

### Remote access
Can you `ssh` into a remote server and work there as comfortably as locally?

- Practice: spin up a free-tier cloud VM and do a day's work over SSH

---

## Docker and containerization basics

[4.1]({{ site.baseurl }}/curriculum/part-04/) teaches containerization directly, [4.3]({{ site.baseurl }}/curriculum/part-04/)
builds container orchestration on top of it, and [Part 7]({{ site.baseurl }}/curriculum/part-07/) deploys every
inference server as a container. You cannot skip this and follow the deployment chapters.

### Images, containers, and isolation
Can you explain how a Docker image differs from a running container, and what isolation
(filesystem, process, network) a container actually provides?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Wf2eSG3owoA" title="Docker and Kubernetes - Full Course for Beginners" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Wf2eSG3owoA">Docker and Kubernetes - Full Course for Beginners</a> &middot; freeCodeCamp.org</div>
</div>

Covers Docker fundamentals before moving into Kubernetes — watch the Docker portion now, come back
for the rest when you reach [Kubernetes Fundamentals]({{ site.baseurl }}/prerequisites/recommended/#kubernetes-fundamentals).

</details>

### Writing a Dockerfile
Can you write a Dockerfile using `FROM`, `COPY`, `RUN`, and `CMD` from a blank file, not by copying
one you found?

- [Docker: Getting Started](https://docs.docker.com/get-started/) &middot; official tutorial, ~2 hours

### Volumes, networking, and ports
Can you run a container, map a port, mount a volume, and view its logs?

- Practice: build 2–3 Dockerfiles yourself, run them, and deliberately break something to debug it

---

## Machine learning fundamentals

[Part 6]({{ site.baseurl }}/curriculum/part-06/) (RAG and knowledge integration) runs on embeddings and
vector similarity. [Part 7]({{ site.baseurl }}/curriculum/part-07/) optimizes inference in ways that assume
you know what a model's outputs mean. [Part 9]({{ site.baseurl }}/curriculum/part-09/) addresses
algorithmic bias, which assumes evaluation-metric literacy.

### Training vs. inference, supervised vs. unsupervised
Can you explain the difference between training a model and running inference on it, and between
supervised and unsupervised learning?

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
  &middot; Andrew Ng (Coursera), first 1–2 weeks, free to audit

### Embeddings and vector similarity
Can you explain why semantic similarity works as a distance calculation in vector space?

<details markdown="block">
<summary>Video</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/OATCgQtNX2o" title="Text embeddings &amp; semantic search" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=OATCgQtNX2o">Text embeddings &amp; semantic search</a> &middot; Hugging Face</div>
</div>

</details>

### Evaluation metrics
Do you know what precision and recall each measure, and why a model can have high accuracy and
still be useless?

- *Hands-On Machine Learning* by Aurélien Géron, chapter 3 (classification and metrics)

### Overfitting and generalization
Can you explain overfitting and why a model that scores perfectly on its training data can still
fail in production?

- *Hands-On Machine Learning* by Aurélien Géron, chapter 4; practice by training a scikit-learn
  classifier and deliberately overfitting it

---

Continue to [Recommended]({{ site.baseurl }}/prerequisites/recommended/) or
[Beneficial]({{ site.baseurl }}/prerequisites/beneficial/) prerequisites, or back to the
[tier overview]({{ site.baseurl }}/prerequisites/).
