---
title: AI tutor
nav_order: 11
permalink: /ai-tutor/
---

# AI tutor
{: .no_toc }

A companion AI assistant for working through the material by dialogue instead of passive answer delivery. It grounds its
answers in the chapter text, cites the chapter, and prefers to ask you questions over giving answers. This page
summarizes [`ai_tutor/README.md`]({{ site.repo_blob }}/ai_tutor/README.md); read that for the full rules and references.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## What a session looks like

The tutor opens by asking what you want to work on, your level (beginner, intermediate, or advanced), and the kind of
help you need. During the session it searches the chapter files before answering, tracks what you get right and wrong,
pushes back on flawed reasoning, and verifies your answers against the text rather than agreeing to be agreeable. After a
substantial session it offers a short reflection on what you covered and what to study next.

**Scope.** The tutor only knows the chapters it has been given, and will say so when you ask about something outside them.
The README currently describes coverage as limited to Parts 1–2.

## Tutoring protocols

You can ask for a protocol by letter, or let the tutor choose (protocol L). Worked examples of each are in
[`example_prompts.md`]({{ site.repo_blob }}/ai_tutor/example_prompts.md).

| Protocol | Name | When it applies |
|---|---|---|
| A | Direct Explanation | You ask "what is X" and need a clear, concise explanation |
| B | Socratic Definition | You are using terms vaguely or confusing related concepts |
| C | Socratic Elenchus | You present reasoning the tutor suspects is flawed |
| D | Socratic Dialectic / Counterfactual | You are exploring design trade-offs with no single right answer |
| E | Prompted Self-Explanation | After seeing a solution — you explain why each step is correct |
| F | Worked Example with Fading | You are new to a procedure and need a full model first |
| G | Step-by-Step Hinting | You are stuck mid-problem and need minimal guidance to continue |
| H | Repeated Practice / Drills | You understand a concept but need speed and fluency |
| I | Quiz / Exam Coaching | You are preparing for the NCP-AAI exam or a course assessment |
| J | Reflection | End of session — meta-cognitive wrap-up and study planning |
| K | Integrity Guardrail | You are working on graded coursework |
| L | Automatic Selection | Default — tutor picks the right protocol from context |
| M | Error Diagnosis | You gave a wrong answer — tutor classifies and remediates the error |
| N | Productive Failure | New concept — you attempt it first, then the tutor teaches from your attempts |
| O | Affective Support | You show signs of frustration, confusion, or boredom |
| P | Collaborative Facilitation | A group of learners working together |

## Proper use

- Concept clarification, reasoning checks, and debugging your understanding
- Drills, quiz and exam practice, and study planning
- Exploring design trade-offs and coaching yourself to explain each step of a solution

## Prohibited use

- Asking for full solutions to graded work, or pasting questions from an open or proctored exam
- Having the tutor ghost-write anything you submit as your own
- Sharing personal data, proprietary code, or restricted datasets: treat the chat as a semi-public notebook, because
  prompts may be logged

When unsure whether AI help is allowed on graded work, treat the assessment as AI-free until your instructor says otherwise.
