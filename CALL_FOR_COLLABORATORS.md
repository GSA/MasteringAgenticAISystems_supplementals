# Call for Collaborators

<!--
STATUS: DRAFT — DO NOT PUBLISH YET.
This call instructs readers to browse GitHub issues by track label and pick a
"good first issue." As of this draft, the seed backlog described in
_cfc/backlog/README.md exists only as local files — nothing has been posted to
GitHub. Publishing this call before that backlog (and its milestone) is live
means every "where to start" instruction below points at an empty tracker,
which is worse for credibility than not publishing at all. Post the backlog
first (see _cfc/backlog/README.md's "what happens after approval" section),
confirm the issues and milestone are live, THEN remove this comment and
publish. This notice is intentionally invisible in rendered Markdown as an
HTML comment, so its presence doesn't affect the reading experience once
someone does view the published version — but check for it before pinning.
-->

## What this project is

*Mastering Agentic AI Systems* is a public-domain study guide built primarily for
the **NVIDIA NCP-AAI (NVIDIA Certified Professional — Agentic AI)** exam, and useful
preparation for four others besides (AWS AIP-C01, Databricks Generative AI Engineer
Associate, Google Cloud PMLE, Microsoft AI-102). If you're studying for one of
those, teaching a course that touches agentic AI systems, or building curriculum on
top of this material, you're who this repository is for — and you're also who we
need help from, because the people best placed to notice what's missing are the
people actually using it to study.

This repository is a **United States government work, in the public domain within
the US**, with copyright and related rights waived worldwide through the
[CC0 1.0 Universal dedication](https://creativecommons.org/publicdomain/zero/1.0/).
Every contribution — prose, code, everything — is released under that same
dedication. We're saying this early and plainly because it's a real decision for a
contributor, not a footnote: if you write something for this project, it enters the
public domain, and you should be comfortable with that before you start. If you
adapt or reuse material you wrote elsewhere, you need to actually hold the rights to
dedicate it this way — content you don't have clear rights to isn't something we
can accept, no matter how good it is. The exception is `videos/`: those YouTube
links are references to third-party work, not contributions, and the dedication
doesn't apply to them.

## Why we're asking, in real numbers

We audited this repository against what it's supposed to offer a reader before
writing this call, and we're not going to soften or round the numbers:

- **`labs/` has zero real coverage.** The 17 files currently in that folder are
  prose excerpts from the chapter text that happen to contain code — not
  standalone exercises. Every one of the book's roughly 94 chapters still needs a
  real lab written from scratch.
- **Slides don't exist for 28 of the book's 94 chapters** — all of Parts 7, 8, 9, and 10, 30%
  of the book. Figures don't exist for any chapter in Parts 1 or 2 (20 chapters).
- **Video coverage looks fine until you click.** Parts 7-10 have 203 entries
  between them, but only 14 actually link anywhere — the rest are titles with no
  URL.
- **One certification is sitting half-done.** Google's "Generative AI Leader" exam
  guide has been in this repo for a while with zero mapping work against it, and
  it isn't even mentioned in `cert_mapping/README.md`'s list of what we cover.

None of this is a crisis. It's a normal state for a project built by one person
writing fast, and it's exactly the kind of gap that's fixable in small pieces by
people who aren't us. That's what this call is for.

## The four tracks, in priority order

### 1. Labs — the focus of this call

This is a build from zero, not a cleanup. Say it plainly to yourself before you
pick this track: the files presently in `labs/` are bare examples, not finished
exercises, and a real lab is needed for every chapter. You are not editing what's
there — you're writing new material against a published template
(`labs/LAB_TEMPLATE.md`) and a worked reference lab
(`labs/8.2B_circuit_breaker/`) that we built, ran, and verified ourselves before
asking anyone else to meet the same bar.

**What you'd be doing:** pick a chapter, design a realistic scenario, write a
guided ("We Do") walkthrough, an independent ("You Do") exercise with a hint
ladder, a working solution with pinned dependencies, and an automated self-check
the learner runs themselves. Existing example files for your chapter (if any) are
worth reading for a reusable scenario or code fragment — not for their structure,
which won't meet the bar.

**Skills:** Python, and whatever framework the chapter covers (LangChain,
LangGraph, CrewAI, AutoGen, Semantic Kernel, or infrastructure tools, depending on
the chapter).

**Setup:** no repository-wide install step exists yet — each lab is
self-contained with its own `requirements.txt`. Before writing one, clone the
reference lab and run it to see the expected shape:

```bash
git clone <your fork>
cd MasteringAgenticAISystems_supplementals/labs/8.2B_circuit_breaker
python3 --version   # 3.10+
python3 test_lab.py # confirm it runs before you model a new lab on it
```

If your repository has `.devcontainer/devcontainer.json` open in a Codespace or
VS Code Dev Containers, Python and a baseline set of agentic-AI framework packages
are pre-installed — no local setup needed at all.

**Time:** roughly 8-12 hours for a self-contained algorithmic lab, 12-20 hours for
one needing an external service — big enough that we expect it as a sequence of
several small pull requests, not one, and we tell you how to split it below.

**Where to start:** browse open issues labeled `track:labs` (a seed set exists,
weighted toward this track deliberately, since it's the biggest gap).

This is also the most attractive track to get wrong in a call: framed honestly, as
authorship of something that will carry your name, it's a real draw; framed as
"tidy up some old files," it undersells the work and misleads you about what
you're signing up for. It's the former.

### 2. Instructional content review

Three different jobs live here: fact-checking the chapter text against current
vendor/framework documentation, reviewing slide decks for accuracy and clarity, and
reviewing the quiz/practice-test index for chapter-mapping consistency (there's
real, confirmed drift in several Parts). Review happens in the open, on a GitHub
issue thread, against a checklist — and the outcome is always accept, minor
revisions, or major revisions, never outright rejection. Your work stays visible
either way.

**Skills:** subject familiarity with the area you're reviewing; no coding required
for slide or quiz review.

**Time:** roughly 1-3 hours per finding for most sub-tasks.

**Where to start:** issues labeled `track:content-review`.

### 3. Certification mapping

Take an exam's official guide, extract its knowledge items, rate every chapter
against every item on our H/M/L/N scale (documented in `cert_mapping/README.md`),
and produce a CSV and summary. The natural first pick is Google's "Generative AI
Leader" — its source exam guide is already in the repo, so the hardest part
(finding and rights-clearing the material) is done for you.

**Skills:** careful, structured reading; no coding required.

**Time:** the Generative AI Leader mapping is a multi-session project (call it
8+ hours across a few sittings); several smaller fixes to existing mappings
(a missing summary file, a misfiled row, a filename bug) are under an hour each.

**Where to start:** issues labeled `track:cert-mapping`.

### 4. Video library

Verify that existing links still work and still show what they claim (start with 9
literally broken placeholder links we already know about — the fastest, most
certain win in this whole call), or add new curated entries where coverage is thin.
New entries need a real link, a topic line, and — a new requirement, since none of
the 548 existing entries have one — a verification date, so link rot gets caught by
someone other than a frustrated reader. Format follows the existing entry style,
with the date as a fourth line:

```
### <Title>
- [<URL>](<URL>) ~<duration>
- Covers: <topics>
- Verified: <YYYY-MM-DD>
```

**Skills:** none beyond being able to judge whether a video is actually a clear,
credible explanation of the topic it's attached to.

**Time:** 15-30 minutes per link check; a bit longer to source and vet a new entry.

**Where to start:** issues labeled `track:video`.

## What you get

Attribution recorded through the [All Contributors](https://allcontributors.org/)
specification, which has real categories for exactly this kind of work —
documentation, tutorial, video, content, review, ideas, examples — not just code
commits, so a lab author, a slide reviewer, and a video curator are all credited by
name for what they actually did. You keep that credit; it accrues as you contribute
more, it doesn't reset. Beyond that: the material itself, whatever depth on the
subject you get from working through it closely enough to write or review it, and —
for people who stick around — a path to becoming a maintainer with real merge
authority, described in [GOVERNANCE.md](GOVERNANCE.md). That last one is one of the
few things a volunteer project can genuinely offer, and we mean it.

## Time commitment

There's no minimum. A single small fix is a complete, welcome contribution on its
own — most people who show up to an open-source project do exactly one thing and
that's a fine outcome, not a failure to convert them into something bigger. If
you want to go further, a sustainable pace for most contributors is a few hours a
week; a full lab at that pace takes 2-4 weeks end to end, split into the smaller
pull requests described below.

## Our capacity, stated plainly

Review is one person (Tam Nguyen), at roughly 3-5 hours a week. There's no backup
reviewer today — if that changes, it'll be because someone who contributes
consistently gets asked to co-maintain, which is a real, open path (see
[GOVERNANCE.md](GOVERNANCE.md)), not because of some larger team standing by. We're
telling you this so your expectations calibrate to it: a small PR gets reviewed
fast; if you send something huge, it waits longer, not because it's unwelcome but
because that's the real math of the time available. This is also why the pull
request size guidance below isn't bureaucracy — it's how we keep the queue moving
at all.

## What we will and won't accept

We'll accept: new labs built against the template and rubric; content corrections
backed by evidence (a quote, a citation, a link); new or corrected certification
mappings following the documented rubric; video entries that meet the curation
standard once each Part's scope statement exists. We won't accept: labs that don't
run or don't follow the template; content changes that alter a chapter's exam
scope or alignment without a maintainer discussion first (open an issue describing
the proposed change before writing the PR — that's the discussion; no separate
process exists); unverified or uncredited video additions; and — this one matters
because of the licensing position above — anything you don't hold clear rights to
dedicate to the public domain. A "no" on any of these is a scope decision stated in
advance, not a personal judgment made up on the spot.

## How to start

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) — the full mechanics live there.
2. Browse issues labeled with the track that interests you (`track:labs`,
   `track:content-review`, `track:cert-mapping`, `track:video`); if it's your
   first contribution here, look for one also labeled `good first issue`.
3. Comment on the issue saying you're starting it. We'll assign it to you and
   reply within 3 business days.
4. Fork, branch, do the work against the definition of done stated in the issue,
   and open a pull request that references it.

## Pull request size

Keep each pull request to roughly 20 changed items or fewer — one section,
exercise, or code block for prose; one file or ~100 lines for code. This isn't a
restriction so much as how we keep review fast on a small team: a PR at this size
gets read and responded to quickly; a 2,000-line PR sits in a queue regardless of
how good it is, because review quality genuinely degrades past this range (Google's
own engineering guidance treats 100 lines as reasonable and 1,000 as too large; the
classic Cisco/SmartBear code-review study found defect-finding drops off past
roughly 200-400 lines and after about an hour of continuous review). A full lab
won't fit in one PR — split it into scaffolding, then the guided section, then the
solution and self-check, then polish, each one mergeable on its own.
`CONTRIBUTING.md` has the full splitting guidance. An oversized PR isn't rejected,
just slower — we'll ask you to split it if it comes to that.

## What review checks, and how fast

Your PR gets checked against the definition of done stated in its issue and the
track's rubric (for labs, the 7-line checklist in `labs/README.md`). We aim for a
first response — even just an acknowledgment — within **3 business days**. If
you've heard nothing after that, ping the issue; it happens on a small team, and a
nudge is always welcome, never annoying.

## Code of conduct and questions

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). For anything
not covered here, see [SUPPORT.md](SUPPORT.md) — GitHub Issues is the primary
channel, with `cto@gsa.gov` as a fallback.

## Is this open-ended?

Yes — this is a standing call, not a time-boxed drive with a deadline. The backlog
gets refreshed on a recurring cycle rather than run down once and left empty; if
you show up and the tracker looks thin, that's a sign we're due for a refill, not
that the project stalled. Check back, or just ask.

## Your very first move

Pick one issue labeled `good first issue` in the track that interests you most,
and leave a comment that says: *"I'd like to work on this."* That's the whole
first step. We'll reply within 3 business days to confirm it's yours and answer
anything the issue didn't already cover.
