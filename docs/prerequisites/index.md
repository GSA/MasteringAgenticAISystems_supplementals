---
title: Prerequisites
nav_order: 3
has_children: true
permalink: /prerequisites/
---

# Prerequisites
{: .no_toc }

What to know before starting, organized into three tiers, each broken into topics and further into
specific sub-skills with a self-check question and one curated resource apiece — a short video
where a good one exists, otherwise a course, book, or official guide. This section expands on
[`Prerequisite_Knowledge.md`]({{ site.repo_blob }}/Prerequisite_Knowledge.md), which has the full
narrative version (why each topic matters, chapter-by-chapter) if you want more context than the
condensed pages here give.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## The three tiers

Browse a tier below, via the sidebar, or jump straight to a topic — each tier page has its own
table of contents.

| Tier | Meaning | Topics |
|---|---|---|
| [**Essential**]({{ site.baseurl }}/prerequisites/essential/) | You must understand these to benefit from the book | LLM Fundamentals, Python, REST APIs & API Design, Command Line & Shell, Docker, Machine Learning Fundamentals |
| [**Recommended**]({{ site.baseurl }}/prerequisites/recommended/) | You can succeed without them, but expect to look things up often | Software Architecture Patterns, Database Fundamentals, Kubernetes, NLP Basics |
| [**Beneficial**]({{ site.baseurl }}/prerequisites/beneficial/) | Accelerate learning and deepen understanding, not required | Distributed Systems, GPU/CUDA Basics, Prompt Engineering, Async Python, CI/CD |

One change from the narrative document: "RESTful APIs and HTTP Fundamentals" and "API Design and
REST Principles" covered mostly the same ground at two depths, so they're merged into one Essential
topic — a "core" group and a "going deeper" group — rather than duplicated across two tiers.

## Quick self-assessment

You should be able to answer yes to all five before starting — each links to its full subtopic
if you can't:

- I understand how LLMs work: tokens, context windows, prompting ([Essential &gt; LLM Fundamentals]({{ site.baseurl }}/prerequisites/essential/#large-language-model-llm-fundamentals))
- I can write and debug intermediate Python ([Essential &gt; Python]({{ site.baseurl }}/prerequisites/essential/#python-programming))
- I understand REST APIs and HTTP basics ([Essential &gt; REST APIs, HTTP, and API design]({{ site.baseurl }}/prerequisites/essential/#rest-apis-http-and-api-design))
- I can use command-line interfaces and write shell scripts ([Essential &gt; Command Line]({{ site.baseurl }}/prerequisites/essential/#command-line-and-shell-scripting))
- I understand Docker containers and can build images ([Essential &gt; Docker]({{ site.baseurl }}/prerequisites/essential/#docker-and-containerization-basics))

If you cannot check at least four of the five, plan two to three weeks of remediation before starting.

## Preparation paths

| Path | Duration | Effort | Covers |
|---|---|---|---|
| Minimal | 2–3 weeks | 25–35 hours | [Essential]({{ site.baseurl }}/prerequisites/essential/) only, enough to start Parts 1–2 |
| Recommended | 4–5 weeks | 40–60 hours | Essential plus [Recommended]({{ site.baseurl }}/prerequisites/recommended/) |
| Comprehensive | 6–8 weeks | 60–90 hours | Essential, Recommended, and [Beneficial]({{ site.baseurl }}/prerequisites/beneficial/) |

Background-specific guidance (academic students, software engineers, ML practitioners, career
changers) is in the closing section of
[`Prerequisite_Knowledge.md`]({{ site.repo_blob }}/Prerequisite_Knowledge.md#final-recommendations).
