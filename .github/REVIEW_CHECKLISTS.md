# Maintainer Review Checklists

One checklist per track, each derived directly from the acceptance criteria already
established for that track — nothing here is invented fresh. A reviewer should be
able to work through any PR against its track's checklist without rereading the
whole project's documentation first. Every issue form (`.github/ISSUE_TEMPLATE/`)
links back to this file.

## Labs (`labs/README.md` §3's rubric, operationalized as a review sequence)

1. **Layout.** The lab lives in its own `labs/<chapter-id>_<slug>/` directory with
   `lab.md`, `starter.py`, `solution.py`, `test_lab.py`, `requirements.txt` as
   separate files — not one bare markdown file, not code inlined in the markdown
   beyond short illustrative snippets.
2. **Front matter is complete and real.** Every field in `LAB_TEMPLATE.md`'s front
   matter is filled in — no leftover `[bracketed instruction]` text, no
   placeholder chapter ID. The `knowledge_items` entries name a specific item, not
   just the chapter number (reject "8.2B: resilience patterns"; accept "8.2B:
   circuit breaker state machine (CLOSED/OPEN/HALF_OPEN) and transition
   conditions").
3. **Starter code runs as given.** `python3 -c "import starter"` (or the
   equivalent for the lab's language) succeeds without error before any learner
   change. It's fine — expected — for `starter.py`'s own TODOs to fail
   `test_lab.py`; it must not crash on import.
4. **Solution passes its own tests.** Run `LAB_MODULE=solution python3
   test_lab.py` (or the lab's equivalent invocation) and confirm every test
   passes cleanly. This is not optional and not something to take the author's
   word for — actually run it.
5. **Dependencies are pinned or explicitly absent.** `requirements.txt` either
   pins exact versions matching the `frameworks` front-matter field, or states
   plainly that there are no third-party dependencies (as the reference lab
   does) — never an unpinned `pip install <package>` with no version.
6. **The self-check is a real, learner-executed pass/fail signal**, not narrated
   Q&A the lab answers for the learner. Confirm the "Self-Check" section names an
   actual command and an unambiguous success condition.
7. **The citation table is complete.** Every learning objective has a row in
   "Citation: Objectives → Practice → Check" mapping it to a real
   `knowledge_items` entry, a real section of the lab body, and a real,
   by-name test (not "the tests" generically — the specific test name).
8. **Realistic, not toy.** The scenario names a plausible system, a real failure
   mode or requirement — reject anything framed as "imagine a function that..."
   with no grounding.
9. **No legacy-file leakage.** The lab doesn't inherit a legacy example file's
   structure, and nothing in it or its PR description calls a legacy `labs/*.md`
   file a "lab" or claims it as prior coverage.
10. **Size.** If the PR exceeds roughly 20 changed items, check whether it's the
    scaffolding/guided/solution+check/polish split described in
    `_cfc/workstreams.md` §5 — if not, ask the contributor to split it before a
    full content review, per `CONTRIBUTING.md`.

Result: `draft` maturity if all 10 pass. Do not merge a lab that fails #3, #4, or
#5 — those aren't judgment calls, they're pass/fail facts you can verify yourself
in under five minutes.

## Instructional content review (chapter text, slides, quizzes)

Modeled on the JOSE process (`_cfc/workstreams.md` §2) — the reviewer's own
submission gets checked here, since "content review" produces a PR against the
finding, or the finding itself as an issue comment:

1. **Location is exact.** A chapter heading, a slide number, or a quiz row's
   chapter ID — not "somewhere in Part 6."
2. **Description is specific and falsifiable.** "This is wrong" is not a finding;
   "this claims LangGraph is stateless, which contradicts the same chapter's own
   text on checkpointing" is.
3. **Evidence is quoted, not summarized.** The exact passage, slide content, or
   quiz row is reproduced (or, for a slide, precisely described) alongside a
   citation to whatever authoritative source shows the discrepancy.
4. **Proposed fix is concrete.** Corrected text, a specific chapter-ID
   reassignment, or a recommendation to merge/retire a duplicate — not just "this
   needs fixing."
5. **The fix doesn't introduce a new inconsistency.** Check the proposed
   correction against the canonical chapter list (`cert_mapping/README.md` plus
   the chapter-4.1 correction noted in `_cfc/repo_audit.md` §1.1) — don't let a
   fix for one drift problem invent a new non-canonical ID.
6. **Verdict is one of three, always.** Accept, minor revisions, or major
   revisions — never a bare rejection. A "major revisions" verdict leaves the
   issue open with your findings attached for anyone (including the original
   submitter) to act on.

## Certification mapping

1. **Row completeness.** The `.csv` has one row per canonical chapter (currently
   94 — see `_cfc/repo_audit.md` §1 for how that number was reconciled) in the
   same order as the existing 5 CSVs. Missing or extra rows get flagged before
   anything else.
2. **Rubric fidelity.** Every cell is H, M, L, or N — nothing else, no blanks
   (the audit already found one stray blank cell in an existing CSV; don't add
   another).
3. **Shape matches the existing convention.** Header row lists knowledge items
   exactly as the exam guide's own numbering/naming (see `nvidia_NCP-AAI.csv` as
   the shape reference), first column is the chapter ID.
4. **The `.md` summary exists and follows the established convention** —
   compare against `databricks_genAI_EngAsc.md` for structure.
5. **`cert_mapping/README.md` is updated in the same PR**, not left stale: the
   overview's certification list, the "Chapter Coverage by Certification" table,
   and the "Understanding the Mappings" bullet list all need the new
   certification added together.
6. **Filename hygiene.** ASCII characters only in filenames (the audit found a
   real non-breaking-hyphen bug from this — check with `hexdump -C | grep -c
   e28091` or similar if anything looks off).

## Video library

1. **Format match.** `### <Title>` / `- [<URL>](<URL>) ~<duration>` /
   `- Covers: <topics>` / `- Verified: <YYYY-MM-DD>` — all four lines, in that
   order, for any new entry.
2. **The URL actually resolves and is on-topic.** Click it. Confirm it still
   shows what "Covers:" claims. This is the single most important check on this
   track — link rot is the whole problem this track exists to fix.
3. **Not a playlist mislabeled as a single video**, unless explicitly and
   visibly marked as a playlist.
4. **Verification date is genuinely today's date** (or the date it was actually
   checked), not copy-pasted from a template.
5. **Earns its place against the Part's scope statement** (once one exists per
   `_cfc/workstreams.md` §4 — until every Part has one, use judgment: is this the
   clearest or most authoritative explanation available, not just "found via
   search").
6. **Duplication check.** The audit found some videos cited 10+ times across
   different chapters — a new entry citing an already-heavily-used video should
   have a specific reason (a different, better excerpt/timestamp, or genuinely
   the best source for this specific chapter).
