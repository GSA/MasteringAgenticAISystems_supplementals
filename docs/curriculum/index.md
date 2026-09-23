---
title: Curriculum
nav_order: 4
has_children: true
permalink: /curriculum/
---

# Curriculum
{: .no_toc }

The material is organized into 10 Parts. Each Part page below lists every chapter with its
summary, study hours, slide deck, quiz, videos, figures, code examples, and which certification
maps rate it highly relevant.

| Part | Topic | Chapters | Study hrs | Slide decks | Videos | Code files |
|---|---|---:|---:|---:|---:|---:|
| [Part 1]({{ site.baseurl }}/curriculum/part-01/) | Agent Fundamentals | 11 | 28.3 | 5 | 52 | 53 |
| [Part 2]({{ site.baseurl }}/curriculum/part-02/) | Framework & Tool Integration | 9 | 22.1 | 4 | 25 | 66 |
| [Part 3]({{ site.baseurl }}/curriculum/part-03/) | Evaluation & Optimization | 12 | 63.8 | 11 | 35 | 35 |
| [Part 4]({{ site.baseurl }}/curriculum/part-04/) | Production Deployment & Scaling | 7 | 24.4 | 7 | 39 | 45 |
| [Part 5]({{ site.baseurl }}/curriculum/part-05/) | Advanced Reasoning & Decision Making | 13 | 70.4 | 13 | 82 | 8 |
| [Part 6]({{ site.baseurl }}/curriculum/part-06/) | Retrieval-Augmented Generation (RAG) | 14 | 26.2 | 8 | 21 | 59 |
| [Part 7]({{ site.baseurl }}/curriculum/part-07/) | NVIDIA NeMo Framework & Optimization | 8 | 20.0 | 0 | 3 | 76 |
| [Part 8]({{ site.baseurl }}/curriculum/part-08/) | Reliability & Cost Management | 5 | 10.9 | 0 | 2 | 29 |
| [Part 9]({{ site.baseurl }}/curriculum/part-09/) | Safety & Governance | 10 | 35.4 | 0 | 5 | 27 |
| [Part 10]({{ site.baseurl }}/curriculum/part-10/) | Human-in-the-Loop & Integration | 7 | 38.3 | 0 | 2 | 4 |
| **Total** | | **96** | **339.8** | **48** | **266** | **402** |

## Where the chapter list comes from

Chapter IDs in this repository are not perfectly consistent across files, so these pages use the
union of the chapter IDs in the five [certification mapping]({{ site.baseurl }}/certifications/) CSVs and the
section headings of [`Study_Plan.md`]({{ site.repo_blob }}/Study_Plan.md), which gives 96 chapters. The
counts therefore differ from the "86 theory chapters" in the Study Plan's introduction and the
"94 chapters" in the certification mapping. Where a slide deck, quiz, or figure set is numbered
differently from a chapter, the Part page says so instead of guessing (see the † notes).

## Cross-cutting complexity categories

[`complexityCategories.md`]({{ site.repo_blob }}/complexityCategories.md) maps chapters onto eight categories
used to judge how complex an agentic AI project is:

| Category | What it covers |
|---|---|
| Perception (`Perc`) | how an agent senses and preprocesses input across modalities |
| Memory (`Mem`) | short-term, long-term, episodic and semantic storage and retrieval |
| Orchestration (`Orch`) | control flow, multi-agent coordination and stateful execution |
| Reasoning (`Rsn`) | planning and reasoning strategies |
| Tool calling (`Tool`) | tool integration and function calling |
| Integration (`Integ`) | connecting agents to systems, data and interfaces |
| Error handling (`Err`) | detecting, classifying and recovering from failures |
| Resilience (`Resil`) | staying available and correct under load and faults |
