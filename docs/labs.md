---
title: Labs
nav_order: 9
permalink: /labs/
---

# Labs
{: .no_toc }

Hands-on exercises are the project's biggest gap. Every chapter needs a real lab; today one chapter has one, and it is
a draft.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## What exists today

| What | Status |
|---|---|
| [Lab 8.2B — Build a Circuit Breaker for a Flaky Downstream Tool]({{ site.repo_blob }}/labs/8.2B_circuit_breaker/lab.md) | **Draft.** Written to the lab template; the maintainer has run the solution and its tests pass; no outside learner has piloted it. About 60–90 minutes; Python 3.10+ standard library only. |
| [`labs/archive/`]({{ site.repo_tree }}/labs/archive) | 17 legacy example files: prose excerpts from the chapters with code embedded. They are **not** labs and do not count as coverage. |
| [`LAB_TEMPLATE.md`]({{ site.repo_blob }}/labs/LAB_TEMPLATE.md) | The standard every new lab is written against. |

The reference lab's files, in [`labs/8.2B_circuit_breaker/`]({{ site.repo_tree }}/labs/8.2B_circuit_breaker): `lab.md`, `starter.py`,
`solution.py`, `test_lab.py`, and `requirements.txt`. Run the self-check with `python3 test_lab.py`.

## What a finished lab contains

Following [`LAB_TEMPLATE.md`]({{ site.repo_blob }}/labs/LAB_TEMPLATE.md), each lab has: **Objectives**, a **Scenario**, **Setup**, a
guided **We Do** walkthrough, an independent **You Do** task with a **Hint Ladder** (gentle, moderate, strong, then the
solution), a runnable **Self-Check**, **Key Points**, and a closing table tracing each objective to the practice that
teaches it and the check that verifies it.

## Quality rubric

A reviewer checks each of these as a yes/no call:

1. **Cites specific knowledge items**, by chapter ID and phrase, not just chapter number.
2. **The learner does the work:** the *You Do* section requires writing or completing real code.
3. **Stands alone:** completable without the book chapter open.
4. **Starter code runs as given:** it may fail its own tests, but must not crash on import or setup.
5. **Solution runs, dependencies pinned:** `solution.py` passes `test_lab.py`; `requirements.txt` pins versions.
6. **Learner-verifiable success:** a test suite or script yields pass/fail, not a narrated Q&A.
7. **Realistic scenario:** a plausible practitioner situation, not a toy.

## Maturity ladder

| Stage | Meaning |
|---|---|
| `example` | Prose with embedded code, not written to the template. The 17 archived files sit here. |
| `draft` | Written to the template and passes the rubric on inspection; not yet worked through by anyone else. |
| `piloted` | At least one person other than the author has completed it and logged issues. |
| `stable` | Piloted, revised from feedback, and the solution re-verified. |

The per-chapter status table is in [`labs/README.md`]({{ site.repo_blob }}/labs/README.md).

## Legacy example files

- [`Part_01_Chapter_1.1_Labs.md`]({{ site.repo_blob }}/labs/archive/Part_01_Chapter_1.1_Labs.md)
- [`Part_01_Chapter_1.6_Labs.md`]({{ site.repo_blob }}/labs/archive/Part_01_Chapter_1.6_Labs.md)
- [`Part_01_Chapter_1.7_Labs.md`]({{ site.repo_blob }}/labs/archive/Part_01_Chapter_1.7_Labs.md)
- [`Part_01_Chapter_1.8_Lab.md`]({{ site.repo_blob }}/labs/archive/Part_01_Chapter_1.8_Lab.md)
- [`Part_02_Chapter_2.10_Lab.md`]({{ site.repo_blob }}/labs/archive/Part_02_Chapter_2.10_Lab.md)
- [`Part_02_Chapter_2.7_Labs.md`]({{ site.repo_blob }}/labs/archive/Part_02_Chapter_2.7_Labs.md)
- [`Part_04_Chapter_4.1_Labs1.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs1.md)
- [`Part_04_Chapter_4.1_Labs2.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs2.md)
- [`Part_04_Chapter_4.6.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.6.md)
- [`Part_06_Chapter_6.3C_ETL_Practice.md`]({{ site.repo_blob }}/labs/archive/Part_06_Chapter_6.3C_ETL_Practice.md)
- [`Part_06_Chapter_6.4B_Data_Quality_Practice.md`]({{ site.repo_blob }}/labs/archive/Part_06_Chapter_6.4B_Data_Quality_Practice.md)
- [`Part_06_Chapter_6.5B_Production_RAG_Practice.md`]({{ site.repo_blob }}/labs/archive/Part_06_Chapter_6.5B_Production_RAG_Practice.md)
- [`Part_06_Chapter_6.6A_Reranking_Implementation.md`]({{ site.repo_blob }}/labs/archive/Part_06_Chapter_6.6A_Reranking_Implementation.md)
- [`Part_06_Chapter_6.6C_Advanced_Retrieval_Practice.md`]({{ site.repo_blob }}/labs/archive/Part_06_Chapter_6.6C_Advanced_Retrieval_Practice.md)
- [`Part_07_Chapter_7.2A_Local_Development.md`]({{ site.repo_blob }}/labs/archive/Part_07_Chapter_7.2A_Local_Development.md)
- [`Part_07_Chapter_7.7_Labs_Practice.md`]({{ site.repo_blob }}/labs/archive/Part_07_Chapter_7.7_Labs_Practice.md)
- [`Part_08_Chapter_8.2B_Labs.md`]({{ site.repo_blob }}/labs/archive/Part_08_Chapter_8.2B_Labs.md)

## Write a lab

Labs are the priority track in the [call for collaborators]({{ site.baseurl }}/contributing/): pick a chapter, design a realistic
scenario, and write it against the template. Expect roughly 4–12 hours for a self-contained algorithmic lab and 12–20
for one that needs an external service, submitted as a sequence of small pull requests.
