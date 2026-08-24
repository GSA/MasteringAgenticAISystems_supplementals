# Governance

This document describes who maintains this repository and how decisions get
made. It is intentionally lightweight: the repository is a public-domain
(CC0 1.0) collection of study material for the NVIDIA NCP-AAI exam, with no
installable package, no CI, and one person actually responsible for it.

## Current maintainer

[@Cybonto](https://github.com/Cybonto) (Tam Nguyen) 

Seven other GitHub accounts hold technical admin+push access to this
repository as an artifact of GSA's broader organizational settings. They are
**not** involved in maintaining this project, are not a backup review path,
and should not be `@`-mentioned expecting a response.

Additional maintainer from GSA is needed.

## Decision-making model

One person makes every decision here — there's no group to reach consensus
among. The distinction that matters isn't who decides, it's how much public
notice a change gets before it lands:

* **Day-to-day changes** (typo fixes, small content corrections, adding an
  example, updating a lab or slide) are reviewed against the relevant
  track-specific rubric/checklist (e.g., the lab standard, the certification
  mapping) and merged directly, no waiting period.
* **Larger or structural changes** — for example, changing the lab
  standard/template, altering the certification mapping (`cert_mapping/`),
  or adding/removing a major section of the curriculum — get posted as an
  issue or pull request and left open for public comment for **one week**
  by default before being merged, so contributors and readers have a real
  chance to react before it's final. This window is adjustable for a given
  proposal (shortened for something time-sensitive, extended for something
  with wide impact) — the window is stated explicitly whenever it differs
  from the default.

## Becoming a maintainer

There's no fixed process, since there's no group to vote within it. In
practice, someone becomes a co-maintainer by:

1. Making sustained, quality contributions across multiple merged pull
   requests over time (not a single large PR).
2. Being asked directly by the current maintainer, based on that track
   record — this is the mechanism by which the single-point-of-failure risk
   above actually gets fixed, and it's a real, open door, not a formality.

The maintainer must be with GSA.

There's no application process beyond this — if you've been consistently
contributing and think you'd be a good fit, it's reasonable to ask.

## Maintainer time budget

Active maintenance capacity right now is roughly **1-5 hours/week**. This is
a placeholder assumption used for planning response times (see
[SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md)); if it's inaccurate,
the repository owner should update this figure — it directly affects how
much review throughput can realistically be promised.

## Attribution model

Most contributions to this repository aren't code commits — they're labs, content
reviews, certification mappings, and curated videos. A model that only credits code
would credit almost nobody who actually does the work, so three options were
weighed on that basis specifically — what a maintainer with 3-5 hours/week will
actually keep current, not what looks most sophisticated on paper:

- **A hand-maintained `CONTRIBUTORS.md` file**, updated manually per PR — rejected.
  Manual upkeep is exactly the kind of small recurring task that falls off a
  time-constrained maintainer's list first, and it has no built-in categories for
  non-code work.
- **Per-file credit lines** (a comment or byline in each lab/review/mapping) —
  rejected. Fragments credit across dozens of files with no single place to see a
  contributor's full record, and doesn't survive a file being restructured or
  superseded (which, per the labs maturity ladder, is expected to happen routinely
  as `example`-stage files get replaced).
- **The [All Contributors](https://allcontributors.org/) specification** —
  adopted. It has existing categories that map directly onto this project's real
  contribution types (content, code, documentation, review, tutorial, video, ideas,
  maintenance), is driven by a bot comment plus a single `.all-contributorsrc`
  config file rather than manual file-hunting, and renders as one table in
  `README.md`. The config and table already exist, seeded with the current
  maintainer; the bot integration itself needs a one-time GitHub App install (see
  `_cfc/publication_plan.md`), but the mechanism works today even without it — a
  maintainer can add an entry by editing `.all-contributorsrc` and running
  `npx all-contributors generate`.

**Credit is cumulative, not recomputed.** Following [The Turing Way](https://book.the-turing-way.org/)'s
practice of accruing authorship across releases rather than recalculating it from
scratch each time, a contributor's recorded contribution types only grow — a lab
author who later also reviews content gets both categories added, never
reassigned or reset.

## Changing this document

Governance changes are made the same way as any other larger/structural
change: open a pull request against this file and leave it open for the
one-week public comment window described above before it's merged.
