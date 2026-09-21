---
title: Contributing
nav_order: 12
permalink: /contributing/
---

# Contributing
{: .no_toc }

This is a small project maintained by one person, and it needs help. Every contribution is released under the same
[CC0 1.0 dedication]({{ site.repo_blob }}/LICENSE.md) as the rest of the repository. If you are unsure about anything, open the issue
or pull request anyway.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Four ways to help

Listed in priority order; the [call for collaborators]({{ site.repo_blob }}/CALL_FOR_COLLABORATORS.md) has the full detail.

| Track | What you would do | Skills | Typical effort |
|---|---|---|---|
| **Labs** (the focus) | Write a hands-on lab for a chapter against the [lab template]({% link labs.md %}) | Python, plus the chapter's framework | 4–12 hours self-contained; 12–20 with an external service; submitted as several small PRs |
| **Instructional content review** | Fact-check chapter text against current vendor documentation, review slides, check quiz-to-chapter mapping | Subject expertise; no coding | 1–3 hours per finding |
| **Certification mapping** | Rate every chapter against an exam's knowledge items (H/M/L/N), for example Google's Generative AI Leader | Careful structured reading | Varies |
| **Video library** | Verify links still work and match their descriptions, or add curated entries | Judgment of what makes a good explanation | 15–30 minutes per link check |

## How to contribute

1. Browse the [open issues](https://github.com/GSA/MasteringAgenticAISystems_supplementals/issues) and pick one labeled with your track (`track:labs`,
   `track:content-review`, `track:cert-mapping`, `track:video`); `good first issue` marks starting points.
2. Comment to say you are taking it, so nobody duplicates the work.
3. Fork the repository and branch from `master`, named `<track>/<short-description>` (for example
   `labs/chapter-6-3c-etl`).
4. Do the work against the issue's definition of done.
5. Open a pull request using the template and reference the issue with `Closes #<number>`.

Keep each pull request to roughly 20 changed items or fewer; a full lab is expected to arrive as a sequence of small,
independently mergeable PRs. Review checks your work against the track's checklist in
[`.github/REVIEW_CHECKLISTS.md`]({{ site.repo_blob }}/.github/REVIEW_CHECKLISTS.md). Expect a first response within a few
business days.

You can also start from an issue form: [New lab, content review, certification mapping, or video entry](https://github.com/GSA/MasteringAgenticAISystems_supplementals/issues/new/choose).

## What is accepted

Accepted: labs built against the template and rubric, evidence-backed content corrections, certification mappings
that follow the rubric, and video entries that meet the curation standard. Not accepted: labs that do not run or diverge
significantly from the template, changes to a chapter's exam scope without first discussing it in an issue, unverified or
uncredited video additions, and anything you do not hold clear rights to dedicate to the public domain.

## Project documents

| Document | What it covers |
|---|---|
| [`CONTRIBUTING.md`]({{ site.repo_blob }}/CONTRIBUTING.md) | Full contribution mechanics |
| [`GOVERNANCE.md`]({{ site.repo_blob }}/GOVERNANCE.md) | Who maintains the project and how decisions are made |
| [`CODE_OF_CONDUCT.md`]({{ site.repo_blob }}/CODE_OF_CONDUCT.md) | Expected behavior |
| [`SUPPORT.md`]({{ site.repo_blob }}/SUPPORT.md) | Where to ask questions and get content errors fixed |
| [`SECURITY.md`]({{ site.repo_blob }}/SECURITY.md) | Reporting an insecure code pattern or a tooling concern |

Questions and content errors go to [GitHub Issues](https://github.com/GSA/MasteringAgenticAISystems_supplementals/issues); include the file path.
