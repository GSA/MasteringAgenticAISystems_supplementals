# Labs

## 1. What belongs here

Every file directly in this folder should be a real, standalone, runnable lab
written against [`LAB_TEMPLATE.md`](LAB_TEMPLATE.md): a learner opens it, works
through a realistic scenario with guided and independent sections, and can tell
for themselves — by running something, not by reading a narrated answer — whether
they succeeded. The 17 legacy `Part_*.md` files live under
[`labs/archive/`](archive/), moved out of the main listing on purpose: they are
**examples**, prose excerpts from the book with code embedded inline, written
before this template existed, kept as source material pending replacement. They
are not labs, they do not meet the rubric below, and they do not count as coverage
for any chapter — a chapter with only a legacy example file listed against it is
still `not started` in the status table in Section 5.

## 2. Design philosophy

This lab spec borrows deliberately from four existing courseware standards rather than inventing conventions from scratch. In each case, one core idea was adopted and the surrounding apparatus was rejected as too heavy for a volunteer, plain-Markdown repository.

**[The Carpentries Workbench](https://carpentries.github.io/workbench/)** — adopted: every lab opens with explicit, learner-facing objectives and closes with a "Key Points" recap. This is the single most effective structural device Carpentries lessons use to keep a lab honest about what it's actually teaching. Rejected: the full R/Quarto build-tooling pipeline (episode YAML, rendering pipeline, lesson-check CI). This repo is plain Markdown; standing up a Quarto toolchain for ~94 labs would be a bigger project than the labs themselves.

**[freeCodeCamp's challenge specification](https://contribute.freecodecamp.org/)** — adopted: the separation of required front matter, a description/scenario, a hint ladder, seed (starter) contents, and a maintained solution that must pass its own tests. This maps directly onto this repo's existing "gentle → moderate → strong → solution" hint-ladder convention (already used in the legacy 8.2B file), so it was kept rather than replaced. Rejected: freeCodeCamp's rigid, platform-specific automated-grader test-string format. This repo adopts the *spirit* — an executable, learner-run, pass/fail self-check — not freeCodeCamp's literal grading DSL, which assumes a hosted platform this repo doesn't have.

**JOSE (Journal of Open Source Education) review criteria, Achieve OER rubrics, and the Quality Matters higher-ed rubric** — despite very different scopes, all three converge on one demand: objectives, activities, and assessment must visibly align — a reviewer should be able to trace each stated objective to the exercise that practices it and the check that verifies it. Adopted: a required citation/traceability line (a table) mapping each learning objective to its knowledge-item citation, the section that practices it, and the self-check item that verifies it. Rejected: Quality Matters' full 8-standard, 42-specific-review institutional accreditation process, and JOSE's formal peer-review/editorial workflow. Those are built for institutional certification of courses; this repo needs the *alignment principle*, not the certification apparatus, so only the traceability requirement was kept.

## 3. Lab quality rubric

A reviewer should be able to check a lab against this list in a few minutes. Each line is a yes/no call against the lab as written — not a subjective quality judgment.

- [ ] **Cites specific knowledge items.** The lab's front matter names one or more specific chapter knowledge items it reinforces (chapter ID + phrase, per the citation convention in Section 6) — not just a chapter number.
- [ ] **The learner does the work.** The "You Do" section requires the learner to write or complete real code/config themselves; the lab is not a walkthrough the learner reads passively.
- [ ] **Stands alone.** The lab is comprehensible and completable without the learner having the surrounding book chapter open — necessary context is restated in the Scenario/Setup, not assumed.
- [ ] **Starter code runs as given.** `starter.py` (or equivalent) installs and executes without errors before the learner has changed anything — it may fail its own tests (that's the point), but it must not crash on import/setup.
- [ ] **Solution runs, dependencies pinned.** `solution.py` passes `test_lab.py` (or equivalent), and `requirements.txt` pins exact versions matching the lab's `frameworks` front-matter field.
- [ ] **Learner-verifiable success.** There is a concrete, learner-executed check (a test suite, a script, an observable behavior) that yields a pass/fail signal — not a narrated "did you get the right answer?" Q&A.
- [ ] **Realistic scenario.** The exercise is framed around a plausible practitioner situation (a named system, a real failure mode or requirement), not an arbitrary toy ("add two numbers").

## 4. Maturity ladder

Adapted, lighter-weight, from the [Carpentries lesson life cycle](https://cdh.carpentries.org/lesson-life-cycle.html):

| Stage | Meaning |
|---|---|
| `example` | Prose-with-embedded-code, not written against the template — this is where the 17 legacy files sit today, under `labs/archive/`, pending replacement. Not counted as chapter coverage. |
| `draft` | Written against `LAB_TEMPLATE.md` and passes the Section 3 rubric on inspection, but has not yet been worked through by anyone other than its author. |
| `piloted` | At least one person other than the author has worked through the lab end-to-end and logged any issues found. |
| `stable` | Piloted, revised in response to that feedback, and the solution has been re-verified to still run (dependencies current, tests passing). |

`labs/8.2B_circuit_breaker/lab.md` enters at a **maintainer-verified `draft`** stage: the maintainer has run the solution and confirmed its tests pass, but no external learner has piloted it yet, so it has not yet earned `piloted`.

## 5. Chapter status table

One row per canonical chapter (94 total, from `cert_mapping/README.md`, Part 4 extended to include chapter 4.1). "Existing example file(s)" lists legacy files worth checking for salvageable content — being listed does **not** imply any maturity beyond `example`; per Section 1, legacy files never count as coverage on their own.

**Path note:** filenames below are bare (e.g. `Part_01_Chapter_1.1_Labs.md`) — every legacy example file actually lives under [`labs/archive/`](archive/), so that filename is at `labs/archive/Part_01_Chapter_1.1_Labs.md`. They were moved there deliberately, out of the main `labs/` listing, so a contributor browsing `labs/` sees real labs and the template first, not 17 files that aren't labs.

### Part 1

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 1.1A | Part_01_Chapter_1.1_Labs.md | not started | Ambiguous — file covers 1.1A or 1.1B, never specifies which. |
| 1.1B | Part_01_Chapter_1.1_Labs.md | not started | Ambiguous — same file as 1.1A; never specifies which. |
| 1.2 | none | not started | |
| 1.3 | none | not started | |
| 1.4 | none | not started | |
| 1.5A | none | not started | |
| 1.5B | none | not started | |
| 1.6 | Part_01_Chapter_1.6_Labs.md | not started | File self-declares an invented "1.6C" section — ignore; canonical target is plain 1.6. |
| 1.7A | Part_01_Chapter_1.7_Labs.md | not started | Ambiguous — file covers 1.7A or 1.7B, never specifies which. |
| 1.7B | Part_01_Chapter_1.7_Labs.md | not started | Ambiguous — same file as 1.7A; never specifies which. |
| 1.8 | Part_01_Chapter_1.8_Lab.md | not started | Clean mapping. |

### Part 2

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 2.1 | Part_02_Chapter_2.10_Lab.md | not started | Unreliable target — see footnote †. |
| 2.2 | Part_02_Chapter_2.7_Labs.md; Part_02_Chapter_2.10_Lab.md | not started | Part_02_Chapter_2.7_Labs.md: filename says 2.7, but content is entirely about 2.2 — clean *content* mapping despite the filename. Part_02_Chapter_2.10_Lab.md: unreliable target, see footnote †. |
| 2.3 | Part_02_Chapter_2.10_Lab.md | not started | Unreliable target — see footnote †. |
| 2.4 | Part_02_Chapter_2.10_Lab.md | not started | Unreliable target — see footnote †. |
| 2.5 | Part_02_Chapter_2.10_Lab.md | not started | Unreliable target — see footnote †. |
| 2.6 | Part_02_Chapter_2.10_Lab.md | not started | Unreliable target — see footnote †. |
| 2.7 | none | not started | Not to be confused with Part_02_Chapter_2.7_Labs.md, whose *content* maps to 2.2, not 2.7 — see the 2.2 row. |
| 2.8 | none | not started | |
| 2.9 | none | not started | |

† **Part_02_Chapter_2.10_Lab.md** has no reliable canonical target: its filename says "2.10", its own H1 heading says "2.11", and its own metadata says "Chapter 2.1-2.6" — none of these are trustworthy, and neither "2.10" nor "2.11" is a canonical chapter ID. It is listed against 2.1–2.6 (its own claimed range) only so a reviewer checking any of those chapters knows to look at it; treat it as unmapped and salvage content from it only after manual confirmation of relevance to the specific chapter.

### Part 3

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 3.1A | none | not started | |
| 3.1B | none | not started | |
| 3.1C | none | not started | |
| 3.2 | none | not started | |
| 3.3 | none | not started | |
| 3.4 | none | not started | |
| 3.5 | none | not started | |
| 3.6 | none | not started | |
| 3.7 | none | not started | |
| 3.8 | none | not started | |
| 3.9 | none | not started | |
| 3.10 | none | not started | |

### Part 4

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 4.1 | Part_04_Chapter_4.1_Labs1.md; Part_04_Chapter_4.1_Labs2.md | not started | Labs1 is a clean mapping to 4.1. Labs2 is broader — its mini-labs touch 4.1 through 4.5. |
| 4.2 | Part_04_Chapter_4.1_Labs2.md | not started | Filed under 4.1, but contains a mini-lab touching this chapter — broader than one chapter; check for salvageable content specific to 4.2. |
| 4.3 | Part_04_Chapter_4.1_Labs2.md | not started | Same situation as 4.2, for 4.3. |
| 4.4 | Part_04_Chapter_4.1_Labs2.md | not started | Same situation as 4.2, for 4.4. |
| 4.5 | Part_04_Chapter_4.1_Labs2.md | not started | Same situation as 4.2, for 4.5. |
| 4.6 | Part_04_Chapter_4.6.md | not started | File self-declares an invented "4.6B" — ignore; canonical target is plain 4.6. |
| 4.7 | none | not started | |

### Part 5

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 5.1 | none | not started | |
| 5.2 | none | not started | |
| 5.3 | none | not started | |
| 5.4 | none | not started | |
| 5.5 | none | not started | |
| 5.6 | none | not started | |
| 5.7 | none | not started | |
| 5.8 | none | not started | |
| 5.9 | none | not started | |
| 5.10 | none | not started | |
| 5.11 | none | not started | |
| 5.12 | none | not started | |
| 5.13 | none | not started | |

### Part 6

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 6.1A | none | not started | |
| 6.1C | none | not started | |
| 6.2A | none | not started | |
| 6.2B | none | not started | |
| 6.3A | none | not started | |
| 6.3B | none | not started | |
| 6.3C | Part_06_Chapter_6.3C_ETL_Practice.md | not started | Clean mapping; richest legacy file in the set. |
| 6.4 | none | not started | |
| 6.4B | Part_06_Chapter_6.4B_Data_Quality_Practice.md | not started | Clean mapping. |
| 6.5 | none | not started | |
| 6.5B | Part_06_Chapter_6.5B_Production_RAG_Practice.md | not started | Clean mapping. |
| 6.6A | Part_06_Chapter_6.6A_Reranking_Implementation.md | not started | Clean mapping. |
| 6.6 | none | not started | |
| 6.6C | Part_06_Chapter_6.6C_Advanced_Retrieval_Practice.md | not started | Clean mapping. |

### Part 7

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 7.1A | none | not started | |
| 7.1B | none | not started | |
| 7.2A | Part_07_Chapter_7.2A_Local_Development.md | not started | Clean mapping. |
| 7.2 | none | not started | |
| 7.3 | none | not started | |
| 7.4 | none | not started | |
| 7.5 | none | not started | |
| 7.6 | none | not started | See footnote ‡ re: Part_07_Chapter_7.7_Labs_Practice.md. |

‡ **Part_07_Chapter_7.7_Labs_Practice.md** is unmapped: "7.7" is not a canonical chapter ID — Part 7 stops at 7.6. It is not listed against 7.6 or any other row above because there is no basis for assuming it belongs to 7.6 specifically; treat any content in it as salvageable only after manual review determines which chapter, if any, it actually belongs to.

### Part 8

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 8.1 | none | not started | |
| 8.2A | none | not started | |
| 8.2B | none | **draft** | Reference lab — see [`labs/8.2B_circuit_breaker/lab.md`](8.2B_circuit_breaker/lab.md). The legacy file Part_08_Chapter_8.2B_Labs.md is **superseded** by this reference lab (built from scratch, not derived from the legacy file) and should not be used as a salvage source. |
| 8.3 | none | not started | |
| 8.4 | none | not started | |

### Part 9

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 9.1 | none | not started | |
| 9.2 | none | not started | |
| 9.3 | none | not started | |
| 9.4 | none | not started | |
| 9.5 | none | not started | |
| 9.6 | none | not started | |
| 9.7 | none | not started | |
| 9.8 | none | not started | |

### Part 10

| Chapter ID | Existing example file(s) | Current maturity | Notes |
|---|---|---|---|
| 10.1 | none | not started | |
| 10.2 | none | not started | |
| 10.3A | none | not started | |
| 10.3B | none | not started | |
| 10.4 | none | not started | |
| 10.5 | none | not started | |
| 10.6 | none | not started | |

### Summary

94 chapters total. 93 are `not started` (no chapter has yet reached `draft` via a real lab written against the template, except one). 1 chapter — **8.2B** — is at `draft`, backed by the reference lab at `labs/8.2B_circuit_breaker/lab.md`. None are yet `piloted` or `stable`. The 17 legacy files remain at `example` stage as source material only; they are not chapter-maturity entries themselves (per Section 1, they don't count as coverage), which is why every chapter they touch still shows `not started` above.

## 6. Citation convention

A lab states which chapter knowledge items it reinforces in its front-matter `knowledge_items` field: a list where each entry is the **canonical chapter ID**, a colon, and a **short phrase naming the specific knowledge item(s)** — in the same style as the knowledge items enumerated per chapter in `cert_mapping/*.csv` (the H/M/L/N-rated rows), so a reviewer can locate the matching item in that chapter's row without guessing. This is deliberately more specific than restating the chapter title: "8.2B: resilience patterns" is not a valid entry; "8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and failure-threshold tuning" is.

Every learning objective in the lab must trace to at least one `knowledge_items` entry, at least one section of the lab body that practices it, and at least one line of the Self-Check that verifies it. This three-way link is recorded explicitly in the lab's closing "Citation: Objectives → Practice → Check" table (see `LAB_TEMPLATE.md`).

**Worked example (chapter 8.2B):**

```yaml
chapter: "8.2B"
knowledge_items:
  - "8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions"
  - "8.2B: failure-threshold and sliding-window tuning for trip/reset behavior"
```

| Objective | Chapter knowledge item | Practiced in | Verified by |
|---|---|---|---|
| Implement OPEN-state fast-fail rejection and timed transition to HALF_OPEN | 8.2B: circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition conditions | "You Do" TODO 1, lab.md | `test_lab.py::TestOpenState` (`test_open_state_rejects_calls_without_running_them`, `test_open_state_transitions_to_half_open_after_timeout`) |
| Implement HALF_OPEN single-trial dispatch and reset-to-CLOSED on success / reopen on failure | 8.2B: failure-threshold and sliding-window tuning for trip/reset behavior | "You Do" TODOs 2-3, lab.md | `test_lab.py::TestHalfOpenState` (`test_half_open_success_resets_to_closed`, `test_half_open_failure_reopens_circuit`, `test_half_open_rejects_concurrent_trial`) |

## 7. A note on this scheme

This maturity ladder and chapter status table are new as of this pass — the table in Section 5 was populated by hand from the files present in `labs/` at the time of writing, cross-referenced against the canonical chapter list in `cert_mapping/README.md`. It will drift as real labs get written and as legacy files are replaced or removed. Future audits should re-run Section 5 from the actual files present in `labs/` rather than trust this snapshot indefinitely — treat this table as a starting baseline, not a live source of truth.
