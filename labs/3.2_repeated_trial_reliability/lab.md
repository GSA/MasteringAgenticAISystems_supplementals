---
title: "Build a Repeated-Trial Stability Check for LLM-Judge and Agent Evaluations"
chapter: "3.2"
knowledge_items:
  - "3.2: non-determinism effects, measured as how often an agent's task outcome and an LLM judge's verdict change across identical re-runs"
  - "3.2: tau-bench pass^k reliability versus single-run task success rate"
  - "3.2: paired evaluation of two agent versions on the same items, with a confidence interval on the difference"
prerequisites:
  - "Chapters 3.1A and 3.2 read first; 3.1B helps for the LLM-as-judge part"
  - "Comfortable with Python lists, functions and basic probability (a proportion, a confidence interval)"
  - "Python 3.10+"
learning_objectives:
  - "Measure how often an agent's outcome and an LLM judge's verdict change between identical re-runs"
  - "Compute pass^k from repeated trials and explain why it falls below the single-run success rate"
  - "Build a paired bootstrap confidence interval for the difference between two agent versions, resampling items"
  - "Determine how many trials per item a stable deploy verdict needs"
estimated_lab_time: "60-90 minutes"
frameworks:
  - "none, Python 3.10+ standard library only"
maturity: "not started"
related_reference_lab: "labs/8.2B_circuit_breaker/lab.md"
---

# Lab 3.2: Repeated-Trial Stability Check

> **In progress (#199).** This part adds the objectives and the scenario. Setup,
> the guided exercise, the independent exercise, the self-check and the citation
> table follow in later pull requests.

## Objectives

By the end of this lab you will be able to:

1. Measure how often an agent's outcome, and separately an LLM judge's verdict,
   changes when you re-run the identical evaluation.
2. Compute pass^k from repeated trials, and explain why it is lower than the
   single-run success rate.
3. Build a paired bootstrap confidence interval for "v2 minus v1", resampling
   items rather than individual runs.
4. Work out how many trials per item you need before the verdict stops changing.

## Scenario

A public benefits help line uses an assistant to tell callers whether they are
likely to qualify for a program, based on the program's published rules. The
team has built v2 and wants to know if it should replace v1. Their offline suite
has 200 eligibility questions with checkable answers. Each question is run once,
and an LLM judge also grades transcripts for tone and completeness.

Someone re-ran the identical suite, and the deploy gate gave a different answer.
That is the problem this lab is about. A wrong answer here sends a caller away
from benefits they qualify for, so the team needs a verdict that holds up when
the evaluation is repeated, not one that depends on which day it ran.

The team re-ran everything 15 times. You have those logs. They are simulated
by a seeded generator, so that everyone gets the same numbers.
