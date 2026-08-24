<!--
LAB_TEMPLATE.md — fill-in skeleton for a real, standalone, hands-on lab.

HOW TO USE THIS FILE
1. Copy this file to labs/<chapter-id>_<short-slug>/lab.md (a folder per lab,
   not a bare .md file in labs/ — see the worked layout at the end of this
   file and the reference example at labs/8.2B_circuit_breaker/).
2. Replace every [bracketed instruction] and HTML comment block with real
   content, then delete the instruction/comment itself. Nothing bracketed
   or commented should remain in the finished lab.
3. Check your finished lab against labs/README.md's "Lab quality rubric"
   before you call it `draft`. The rubric and the sections below are the
   same design — this template does not invent a second standard.
4. Do not inline starter code or solution code as fenced blocks in this
   markdown file beyond short illustrative snippets. Starter and solution
   code are SEPARATE FILES alongside this lab.md (see "Required companion
   files" at the end). This keeps the lab runnable, diffable, and testable
   the way prose-with-embedded-code cannot be.
-->

---
<!--
FRONT MATTER — all fields required. This is the lab's citation and
discovery metadata; a reviewer or a future indexing script should be able
to read only this block and know what the lab is, who it's for, and what
it proves.
-->
title: "[Short, concrete lab title — name the artifact the learner builds, not the topic. e.g. 'Build a Circuit Breaker for a Flaky Downstream API', not 'Resilience Patterns']"
chapter: "[Canonical chapter ID from cert_mapping/README.md, e.g. 8.2B — exactly one ID, not a range. If the material genuinely spans chapters, pick the one chapter this lab primarily reinforces and note the others in Prerequisites instead.]"
knowledge_items:
  <!--
  Citation convention (defined in labs/README.md, "Citation convention").
  One entry per chapter knowledge item this lab reinforces. Each entry is
  the chapter ID plus a short phrase naming the specific knowledge item,
  in the same style as the H/M/L/N items in cert_mapping/*.csv — specific
  enough that a reviewer can find the matching row in that chapter's
  knowledge-item list, not a restatement of the chapter title.
  -->
  - "[chapter-id: specific knowledge item phrase, e.g. '8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and failure-threshold tuning']"
  - "[chapter-id: second knowledge item, if this lab covers more than one]"
prerequisites:
  - "[Prior lab or chapter the learner should already have completed, or 'None' if this is a first lab]"
  - "[Required tool/account/local setup, e.g. 'Docker Desktop running locally', 'Python 3.11+']"
learning_objectives:
  <!--
  Carpentries-style: 3-5 objectives, each a single observable action the
  learner will be able to perform, starting with a verb. Not "understand"
  or "learn about" — those aren't checkable. Each objective must be
  traceable to (a) a knowledge_items entry above, (b) a "You Do" or "We Do"
  section below that practices it, and (c) a Self-Check item that verifies
  it. See the citation line at the bottom of this template.
  -->
  - "[Verb + concrete outcome, e.g. 'Implement a state machine that trips from CLOSED to OPEN after N consecutive failures']"
  - "[...]"
estimated_lab_time: "[Wall-clock time for a learner who has the prerequisites, e.g. '60-90 minutes'. This is learner time, NOT the effort it took you to write the lab.]"
frameworks:
  <!--
  Every framework/library the learner's code touches, with the exact
  version pinned in the companion requirements.txt — not restated loosely
  here. List name + pinned version so a reviewer can spot a drift between
  this line and requirements.txt at a glance.
  -->
  - "[library==X.Y.Z]"
maturity: "[example | draft | piloted | stable — see labs/README.md's maturity ladder. New labs start at 'draft' at the earliest, once they meet the rubric.]"
related_reference_lab: "labs/8.2B_circuit_breaker/lab.md"  <!-- Optional: keep only if genuinely related; otherwise delete this line. It is included here to show the expected link format. -->
---

# [Lab Title]

<!--
OPENING — Carpentries-style. State the objectives up front in plain prose
(a short paragraph or restated bullet list is fine — the front-matter
learning_objectives are the source of truth; this is the human-readable
version the learner sees first).
-->

## Objectives

By the end of this lab you will be able to:
- [Restate learning_objectives in learner-facing language]

## Scenario

<!--
A REALISTIC scenario, not a toy. Ground it in a situation a practitioner
would actually face: a named system, a concrete failure mode, a business
or operational consequence if the learner's solution doesn't work. Avoid
"imagine you have a function that adds two numbers." One paragraph is
usually enough — this sets up why the exercise matters, it is not a
requirements spec.
-->

[2-4 sentences: who is the learner acting as, what system are they working
on, what broke or needs building, what happens if they get it wrong.]

## Setup

<!--
Minimal, copy-pasteable. Point to the companion requirements.txt rather
than restating versions. If there's a repo/dir structure the learner
needs to create, show it here.
-->

```bash
[commands to clone/cd into the lab dir, create a venv, install pinned deps]
pip install -r requirements.txt
```

## We Do (Guided)

<!--
A worked, guided walkthrough of a similar-but-not-identical problem to the
one in "You Do" below. The learner should type/run real code here, with
your explanation of *why* each step is structured the way it is — this is
scaffolding, not a lecture. End this section with something runnable that
the learner can confirm works before moving on.
-->

[Guided steps, each with a short "why this approach" explanation and a
command or code reference the learner runs. Reference starter.py sections
by function/class name rather than pasting large blocks here.]

**Checkpoint:** [A concrete thing the learner runs or observes to confirm
the guided section worked before proceeding, e.g. "running `pytest
test_lab.py::test_guided_example` passes".]

## You Do (Independent)

<!--
The learner implements something themselves, in starter.py, without being
walked through it step by step. State the task precisely enough to be
gradable, but do not give away the implementation approach — that's what
the hint ladder is for.
-->

### Task

[Precise task statement: what function/class/behavior the learner must
implement in starter.py, its expected inputs/outputs/edge cases.]

### Hint Ladder

<!--
House convention, kept from the legacy files. Four levels, each one more
revealing than the last. Do not skip levels or collapse them — a learner
should be able to stop at whichever level unblocks them.
-->

**Hint 1 (Gentle — nudge toward the right question):**
[Point at the concept or design decision to think about, without naming
the technique.]

**Hint 2 (Moderate — narrows the approach):**
[Name the pattern/algorithm/API to use, still without giving the code.]

**Hint 3 (Strong — near-implementation guidance):**
[Pseudocode-level detail, key edge cases to handle, or a partial code
skeleton. The learner should be able to finish from here.]

**Solution:**
[Point to solution.py — do not inline the full solution in this markdown
file. e.g. "See solution.py for a complete reference implementation."]

## Self-Check

<!--
Learner-executed, pass/fail, concrete. NOT narrated Q&A ("Ask yourself:
did you...?"). The learner should be able to run one command and get an
unambiguous yes/no signal, ideally the same test suite that validates
solution.py.
-->

Run:
```bash
python3 test_lab.py
```
<!-- Use pytest instead only if your lab's requirements.txt already needs
     pytest for another reason — the reference lab uses plain unittest via
     `python3 test_lab.py` specifically to keep zero third-party dependencies.
     Match whichever your lab actually uses; don't add pytest just to run tests. -->

You have succeeded if:
- [ ] [Concrete, observable pass condition #1, e.g. "all tests in
      test_lab.py pass"]
- [ ] [Concrete, observable pass condition #2, if the automated tests
      don't cover everything worth checking, e.g. "manually triggering 5
      consecutive failures causes the breaker to log a state transition
      to OPEN"]

## Key Points

<!--
Carpentries-style closing recap: short bullet list, one line per
learning objective, stated as a fact/takeaway rather than a task.
-->

- [Takeaway restating learning objective 1 as a fact]
- [Takeaway restating learning objective 2 as a fact]
- [...]

## Citation: Objectives → Practice → Check

<!--
Required traceability line (see labs/README.md, "Design philosophy" —
adopted from the JOSE / Achieve OER / Quality Matters alignment
principle). One row per learning objective. This is what lets a reviewer
confirm in under a minute that every objective is actually practiced and
actually verified, not just declared in the front matter.
-->

| Objective | Chapter knowledge item | Practiced in | Verified by |
|---|---|---|---|
| [Objective 1, short form] | [knowledge_items entry it maps to] | [We Do step / You Do task] | [Self-Check item] |
| [Objective 2, short form] | [knowledge_items entry it maps to] | [We Do step / You Do task] | [Self-Check item] |

---

<!--
REQUIRED COMPANION FILES — model this on labs/8.2B_circuit_breaker/.
This lab.md should live in its own directory alongside:

  <lab-dir>/
    lab.md            <- this file
    starter.py        <- skeleton the learner completes (the "You Do" task)
    solution.py        <- complete reference implementation
    test_lab.py         <- the executable self-check; also validates solution.py
    requirements.txt     <- pinned dependencies (exact versions, matching
                            the `frameworks` front-matter field)

Do not inline starter or solution code as full files in this markdown.
Short illustrative snippets (a few lines, to explain a concept in "We Do")
are fine; the actual gradable artifacts belong in the separate files so
they can be run, diffed, and tested independently of the prose.
-->
