## How to contribute

> All contributions to this project will be released under the CC0
> dedication. By submitting a pull request, or filing a bug, issue, or
> feature-request you are agreeing to comply with this waiver of copyright interest.
> Details can be found in our [LICENSE](LICENSE.md).

We're so glad you're thinking about contributing to a GSA open source project! If you're unsure about anything, just ask -- or submit the issue or pull request anyway. The worst that can happen is you'll be politely asked to change something. We love all friendly contributions.

Looking for a place to start? See our [call for collaborators](CALL_FOR_COLLABORATORS.md) — it lays out the four tracks we're recruiting for (labs, instructional content review, certification mapping, and video curation), roughly how long one unit of work takes, and what you get out of it.

## Submit an issue

Use the issue tracker to suggest feature requests, report bugs, and ask questions. This is also a great way to connect with the developers of the project as well as others who are interested in this solution.

## Picking up an issue

1. Browse the [open issues](../../issues) and find one labeled with a track you want
   (`track:labs`, `track:content-review`, `track:cert-mapping`, `track:video`). If
   you're new to the repository, start with something also labeled
   `good first issue`.
2. Comment on the issue to say you're picking it up, so two people don't duplicate
   the work. A maintainer will assign it to you.
3. Fork the repository and create a branch off `master` (this repository's
   default branch).
4. Do the work, following the issue's stated definition of done and the exemplar it
   links to.
5. Open a pull request using the PR template, and reference the issue with
   `Closes #<issue number>`.

If an issue turns out to be bigger or smaller than it looked, say so in a comment —
we'll adjust the effort label or split it.

## Branch and commit conventions

- Branch names: `<track>/<short-description>`, e.g. `labs/chapter-6-3c-etl`,
  `content-review/part-9-slides`, `cert-mapping/google-genai-leader`,
  `video/part-8-verification`.
- Commit messages: a short imperative summary line (`Add lab for Chapter 6.3C`, not
  `Added` or `Adding`), with more detail in the body if the change needs it. One
  logical change per commit is preferred but not required — we squash-merge, so your
  branch's commit history doesn't need to be tidy.
- Keep commits (and the PR as a whole) scoped to one issue. Don't bundle unrelated
  fixes into the same PR.

## Pull request size

Keep each pull request to roughly **20 changed items or fewer**. For prose or lab
content, one "item" is one added, changed, or removed section, exercise, or code
block within a file. For code, one item is one added, changed, or removed file, or
roughly 100 changed lines, whichever is smaller. This keeps review fast — reviews
degrade past about an hour of continuous effort, and a PR at this size is reviewable
in well under that.

A full chapter lab will not fit in one PR at this size. Split it: scaffolding first
(structure, front matter, objectives), then the guided exercise, then the solution
and self-check, then any polish. Each PR in the sequence should be independently
mergeable — later PRs can depend on earlier ones being merged, but each one should
leave the repository in a working state on its own.

An oversized PR won't be rejected, but it will be slower to review. If a maintainer
asks you to split one, that's why.

## What review checks, and how fast

A maintainer will check your PR against the definition-of-done and the rubric linked
from its issue (for labs, see `labs/README.md`'s quality rubric — 7 checkable lines;
every other track's checklist lives in the same file,
[`.github/REVIEW_CHECKLISTS.md`](.github/REVIEW_CHECKLISTS.md)). Content review
specifically resolves to one of three outcomes: accept, minor revisions, or major
revisions — never outright rejection. A "major revisions" verdict means the original
issue stays open with the reviewer's findings attached; either you or another
contributor can pick it back up against those findings, the same way any other open
issue works. We aim to leave a
first response — even just an acknowledgment — within **3 business days**. If you
haven't heard anything after that, ping the issue; things do fall through the cracks
on a small team.

## Requesting a change

Fork this repository, make changes in your own fork, and submit a pull request.
This is primarily a content repository — most contributions are prose, labs, or
illustrative code snippets rather than a shipped application, so we don't require
unit tests for everything. What we do require: if you're contributing a runnable
lab or code snippet, confirm you ran it yourself and it works as described (say so
in the PR). Follow the stylistic and structural conventions already used in the
track you're contributing to — the issue you picked up will link you to an exemplar
to follow.

## Licensing intake — what we check before merging

This project is a US government work in the public domain, and everything in it —
prose and code alike — is dedicated worldwide under CC0 1.0. That means a
contributor has to actually hold the right to make that dedication for whatever
they submit. Before merging, a reviewer checks:

- **Contributed prose** (lab text, content corrections, cert-mapping summaries):
  did you write this yourself, or do you hold clear rights to release it under
  CC0? Content adapted from a copyrighted source (a paywalled article, another
  book, a proprietary course) isn't something we can accept, even if it's good.
- **Contributed code** (lab solutions, `starter.py`/`solution.py`, `code_examples/`
  additions): same standard. Code adapted from an existing project needs a
  license compatible with being re-released as CC0/public domain — copying
  GPL-licensed code, for instance, would create a real conflict, since a
  contributor can't unilaterally relicense someone else's copyleft work. When in
  doubt, write it from scratch against the pattern rather than adapting.
- **Linked third-party videos** (`videos/`): these are references, not
  contributions — no rights transfer is implied or required. The check here is
  simpler: is the link accurate, and does it point at something publicly
  viewable (not a private or paywalled video)?

**Origin attestation:** commits should carry a `Signed-off-by` trailer (`git
commit -s`), the standard Developer Certificate of Origin mechanism — it's a
single git flag, not a separate signing service, and it's the low-friction choice
here on purpose. A Contributor License Agreement was considered and rejected: CLAs
generally require a separate signing step through a third-party service, which is
real friction for a volunteer making a small, one-off contribution, and this
project doesn't need the additional protections a CLA is usually adopted for.

*Open question, not yet resolved — flagged here rather than acted on:* this
repository currently applies one CC0 dedication uniformly to both prose and Python
content. Whether that's the intended, GSA-required approach, or whether code and
prose should be licensed differently, hasn't been confirmed with the publishing
organization. The public-domain status itself isn't something this project can
unilaterally change — any adjustment here needs sign-off from GSA, not just a PR.

## Further inquiry

We encourage you to read this project's CONTRIBUTING policy (you are here), its [LICENSE](LICENSE.md), its [README](README.md), our [SUPPORT](SUPPORT.md), [SECURITY](SECURITY.md), and [GOVERNANCE](GOVERNANCE.md) documents, and adhere to its [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md).

If you have any questions or want to read more, check out the [GSA Open Source Policy](https://open.gsa.gov/oss-policy/) and [Guidance repository](https://github.com/GSA/open-source-policy), or just [shoot us an email](mailto:cto@gsa.gov).

---

## Public domain

This project is in the public domain within the United States, and
copyright and related rights in the work worldwide are waived through
the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).

All contributions to this project will be released under the CC0
dedication. By submitting a pull request, you are agreeing to comply
with this waiver of copyright interest.
