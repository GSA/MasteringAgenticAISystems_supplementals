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

> **In progress (#199).** The objectives, scenario, setup and guided exercise are
> in place. The independent exercise, the self-check and the citation table follow
> in later pull requests.

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
(see `trial_data.py`) so that everyone gets the same numbers.

## Setup

No installation needed beyond Python itself. This lab has no third-party
dependencies (see `requirements.txt`).

```bash
cd labs/3.2_repeated_trial_reliability
python3 --version    # 3.10 or newer
python3 starter.py   # prints the stability report; unfinished parts say TODO
```

## We Do: what one run hides

Open `trial_data.py` first and skim `generate()`. Every list it returns is
indexed `[item][trial]`:

- `data.v1` and `data.v2`: did that version get question `item` right on re-run
  `trial`?
- `data.judge_regrades`: the LLM judge's pass/fail verdict when it grades the
  *same* v2 transcript again. The input is identical every time, so any change
  comes from the judge itself.

Now open `starter.py` and read the four provided functions at the top.

**Step 1: `single_run_winner`.** This is the gate the team had: run the suite
once, compare the two success rates, ship the higher one. Run
`python3 starter.py` and look at the first line of the report. Re-running that
gate 15 times gives v2 12 wins and v1 3 wins. One time in five, the same
evaluation reverses the deploy decision.

**Step 2: `flip_rate`.** Why does that happen? Look at how `flip_rate` is
computed. For one item with `c` passes out of `n` trials, it counts the share of
pairs of trials that disagree: `2c(n - c) / (n(n - 1))`. That gives the chance
that two re-runs of the same item disagree. We use pairs instead of "did any trial
disagree" because that version climbs toward 100% as you record more trials,
even when nothing about the agent has changed.

The report shows the agent flips on about 18% of items and the judge on about 12%,
**on identical input**. A one-run comparison adds all of that noise to both
versions, and the true gap between v1 and v2 is only about 2 points.

**Step 3: `per_item_rate`.** Every TODO below starts from per-item rates:
the fraction of an item's trials that passed. Averaging trials *within* an item
first, then comparing items, is what keeps the analysis paired.

**Checkpoint:** `python3 starter.py` prints numbers for the one-run gate (v2 wins
12, v1 wins 3) and both flip rates (18.1% and 11.6%). The last three lines say
`TODO`. Those are the functions you write next.
