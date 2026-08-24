# Mastering Agentic AI Systems — Supplementals

Companion material for *Mastering Agentic AI Systems: Guide for the NVIDIA NCP-AAI
Exam*. It was written primarily to support the **NVIDIA NCP-AAI (NVIDIA Certified
Professional — Agentic AI)** certification, and its breadth also makes it useful
preparation for **AWS AIP-C01**, **Databricks Generative AI Engineer Associate**,
**Google Cloud Professional Machine Learning Engineer**, and **Microsoft AI-102**. See
[`cert_mapping/README.md`](cert_mapping/README.md) for exactly which chapters map to
which certification's knowledge items.

If you're studying for one of those exams, teaching a course that covers agentic AI
systems, or building your own curriculum on top of this material, this repository is
for you.

## Repository contents

The book itself is organized into 10 Parts and roughly 94 chapters (see
[`cert_mapping/README.md`](cert_mapping/README.md) for the authoritative chapter list).
Everything else in this repository is keyed to that chapter numbering.

| Folder | What's in it |
|---|---|
| `Study_Plan.md` | The full chapter text — the primary reading material. |
| `Prerequisite_Knowledge.md` | What to know before starting, and where to fill gaps. |
| `complexityCategories.md` | How topics are classified by difficulty across the book. |
| `quizzes_ver20JUN26.md` | Per-chapter quiz links (external, Google Forms). |
| `simulated_tests_ver20JUN26.md` | Full-length practice exam links (external, Google Forms). |
| `cert_mapping/` | Chapter-to-certification knowledge-item mappings (H/M/L/N relevance ratings), one CSV per certification, plus source exam guides. |
| `labs/` | Hands-on exercises. **This folder is being rebuilt from scratch** — see the note below. |
| `slides/` | Presentation decks, one or more per chapter, PDF. |
| `videos/` | Curated third-party YouTube resources, one file per Part. These are references, not contributed material. |
| `code_examples/` | Standalone illustrative code snippets pulled from the chapter text. |
| `more_examples/` | Additional worked examples. |
| `figures/` | Diagrams and figures used in the book, one folder per chapter. |
| `References/` | Source material and vendored third-party reference documentation. |
| `ai_tutor/` | Resources for using an AI assistant as a study tutor, with example prompts. |

### About `labs/`

The 17 files under [`labs/archive/`](labs/archive/) are prose excerpts from the
chapter text that happen to contain worked code — not finished, standalone
exercises, and moved out of the main `labs/` listing on purpose so browsing
`labs/` shows real labs and the template first. A real lab for every chapter is
still to be written. If you're looking for hands-on practice today, start with
[`labs/LAB_TEMPLATE.md`](labs/LAB_TEMPLATE.md) and the reference lab it links to, which
define what a finished lab looks like and are the standard new labs are built against.

## How to use this material to study

1. Start with `Prerequisite_Knowledge.md` to check what background you need.
2. Work through `Study_Plan.md` chapter by chapter, using
   `cert_mapping/README.md` to prioritize chapters most relevant to your target
   certification.
3. Use the per-chapter quiz links in `quizzes_ver20JUN26.md` to check retention as you
   go, and the full-length practice exams in `simulated_tests_ver20JUN26.md` closer to
   exam day.
4. Reinforce concepts hands-on with `labs/` (see the note above — coverage is still
   being built out) and by reading the illustrative snippets in `code_examples/`.
5. Use `slides/` for a condensed pass and `videos/` for third-party explanations of
   the same concepts from a different angle.

## Licensing

As a work of the United States government, this project is in the public domain
within the United States. We also waive copyright and related rights in the work
worldwide through the
[CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this project — prose, code, and otherwise — are released under
that same dedication. Third-party video links in `videos/` are references and are not
themselves contributed material covered by this dedication. See
[LICENSE.md](LICENSE.md) for the full text.

## Contributors

Recognized via the [All Contributors](https://allcontributors.org/) specification,
which credits documentation, tutorial, video, content, review, ideas, and
maintenance work by name — not only code. Credit accrues as you contribute more; it
isn't recomputed or reset. The bot integration that auto-updates this table from a
PR comment (`@all-contributors please add @user for content`) needs a one-time
GitHub App installation by a repository admin — until that's done, entries are
added by editing `.all-contributorsrc` and running `npx all-contributors generate`
as part of the PR.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Cybonto"><img src="https://avatars.githubusercontent.com/Cybonto" width="100px;" alt="Cybonto"/><br /><sub><b>Cybonto</b></sub></a><br /><a title="Content">🖋</a> <a title="Code">💻</a> <a title="Documentation">📖</a> <a title="Maintenance">🚧</a> <a title="Project Management">📆</a></td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## We're looking for help

The biggest gap here is `labs/`: every chapter needs a real, hands-on lab written
from scratch — the 17 files under `labs/archive/` are prose excerpts, not finished
exercises, and none of it counts as coverage. That's the focus of this call, but
it's not the only track. We're also looking for instructional-content reviewers
(chapter text, slides, and quizzes — no coding required), help extending
`cert_mapping/` to cover more certifications (one, Google's "Generative AI Leader,"
already has its exam guide sitting here with zero mapping done), and help
verifying and growing the `videos/` library, where a lot of entries in the later
chapters look populated but don't actually link anywhere.

None of this is urgent-crisis territory — it's the normal state of a project one
person built fast. See our [call for collaborators](CALL_FOR_COLLABORATORS.md) for
the real numbers, what each track involves, and what you get out of it, and
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.
