# `docs/` — GitHub Pages source

This folder is the source of the project website for
[Mastering Agentic AI Systems — Supplementals](../README.md). It is written for
[GitHub Pages](https://docs.github.com/en/pages) (Jekyll, with the
[Just the Docs](https://just-the-docs.com/) theme) and is also readable as plain
Markdown when browsing it on GitHub.

If you are a learner, you do not need this file — start at the site home page
([`index.md`](index.md)) or the repository [README](../README.md).

## Publishing

1. In the repository, go to **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select branch `master` and folder `/docs`, then save.

The site is then served at
`https://gsa.github.io/MasteringAgenticAISystems_supplementals/`. That address is
set by `url` and `baseurl` in [`_config.yml`](_config.yml); change both if the
repository moves or is forked.

**Partially verified against the live site**, not a local Jekyll build: no local Jekyll
toolchain is available where these pages are written (system Ruby is 2.6, too old for
a current `jekyll`/`rouge`). Once GitHub Pages had actually built and served the site,
its rendered HTML was checked directly (`curl`, not just reading source) — this is how
the `{% link %}` baseurl bug above was actually found, since it was invisible from the
Markdown source and from a same-repo-file-exists check alike. The video players
specifically have been confirmed to embed (oEmbed-checked per video), but not watched
in a browser. After any change here, re-check a live rendered page's `href=`
attributes, not just that the source parses. The theme is loaded as
`just-the-docs/just-the-docs` without a version pin; pin it
(`just-the-docs/just-the-docs@vX.Y.Z` in `_config.yml`) so a theme release cannot
change the site unannounced.

## Layout

```text
docs/
├── README.md            This file (excluded from the built site)
├── _config.yml          Jekyll + theme configuration, and the repo link variables
├── Gemfile              Local-preview dependencies (ignored by GitHub Pages)
├── index.md             Site home
├── getting-started.md   Study path and how to use the material
├── prerequisites/
│   ├── index.md          Tier overview, self-assessment, preparation paths
│   └── essential.md, recommended.md, beneficial.md   One page per tier: topics broken into
│                         sub-skills, each with a self-check question and one curated resource
├── curriculum/
│   ├── index.md         The 10 Parts at a glance
│   └── part-01.md … part-10.md   One page per Part: chapter-by-chapter resource table
├── certifications.md    Certification mappings (NCP-AAI, AWS, Databricks, Google, Microsoft)
├── practice.md          Chapter quizzes and full-length practice exams
├── videos.md            Curated third-party videos, per Part
├── video-link-check.md  Record of the video link check/cleanup: what was removed and why (hidden from the menu)
├── slides.md            Slide decks, per chapter
├── labs.md              Lab status and how labs are written
├── code-examples.md     Code snippets and worked examples, per Part
├── ai-tutor.md          The AI study tutor
├── contributing.md      How to help, governance, conduct, security, support
└── about.md             License, citation, contributors
```

## How content is sourced

The pages here are **navigation and catalog pages**. They do not duplicate the
book or the study materials. The canonical files stay where they are in the
repository (`Study_Plan.md`, `slides/`, `figures/`, `cert_mapping/`, and so on) and
every page links to them. The one exception is the short chapter summaries on the
Part pages, which are excerpted from `Study_Plan.md`.

Links to repository files are built from two variables in `_config.yml`, so the
whole site can be re-pointed by editing one line:

```liquid
{{ site.repo_blob }}/slides/Ch3.2_v20MAR26.pdf   {% comment %} file view {% endcomment %}
{{ site.repo_tree }}/figures                     {% comment %} directory view {% endcomment %}
```

| Page | Built from |
|---|---|
| `curriculum/part-NN.md` | Chapter IDs from the five `cert_mapping/*.csv` files and the section headings of `Study_Plan.md` (titles, hours, summaries); resources matched from `slides/`, `figures/`, `code_examples/`, `videos/`, and `quizzes_ver20JUN26.md` (rules below); embedded videos from `videos/` (see "Embedded videos") |
| `certifications.md` | `cert_mapping/README.md` and the five CSVs |
| `practice.md` | `quizzes_ver20JUN26.md`, `simulated_tests_ver20JUN26.md` |
| `videos.md`, `video-link-check.md` | `videos/Part_NN_YoutubeVideos.md`, plus a live check of each YouTube link (see below) |
| `slides.md` | `slides/*.pdf` (titles read from each deck's first page) |
| `labs.md` | `labs/README.md`, `labs/LAB_TEMPLATE.md`, `labs/8.2B_circuit_breaker/` |
| `code-examples.md` | `code_examples/`, `more_examples/` |
| `ai-tutor.md` | `ai_tutor/README.md` |
| `prerequisites/*.md` | `Prerequisite_Knowledge.md`, cross-checked against `Study_Plan.md`'s per-chapter Key Concepts, plus reused videos from `videos/*.md` where a good match exists (see "Prerequisite sub-topics and embedded videos" below) |
| `contributing.md`, `about.md` | `CONTRIBUTING.md`, `CALL_FOR_COLLABORATORS.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `LICENSE.md`, `CITATION.cff`, `.all-contributorsrc` |

The pages were produced by a one-off script and are maintained by hand from here on;
there is no generator in the repository. They are snapshots, so they will drift as
the sources change. When you change a source file, update the matching page in the
same pull request. The chapter counts, deck counts, and video-link counts are the
values most likely to go stale.

### How resources are matched to chapters

The Part pages attach each resource to a chapter by **exact chapter ID**, with these
exceptions, which are marked **†** on the page:

- A resource numbered for a chapter family is attached to every chapter in the family:
  the `1.5` in slide deck `Chapter1.4_1.5_…` (1.5A, 1.5B) and slide deck `Ch6.1` (6.1A,
  6.1C); figure folders `Ch6.1` and `Ch10.3`; quiz `6.1` (6.1A, 6.1C), `6.2` (6.2A, 6.2B), `6.3` (6.3A, 6.3B); the `6.1`
  section of the Part 6 video file.
- Manual quiz overrides, matched by title: quizzes `4.1A` and `4.1B` → chapter 4.1;
  `7.2A` and `7.2B` → 7.2; `10.3` → 10.3A only.

**Quizzes are attached only where the quiz title and chapter title agree on review.**
Six chapters have a quiz with the same number but a different topic, so they get no
quiz link (1.8, 3.6, 4.4, 5.11, 7.2A, 10.6), and thirteen quizzes have no matching
chapter (4.2A, 4.2B, 5.14, 5.15, 6.7, 6.9B, 9.6A, 9.6B, 9.8A, 9.8B, 9.9, 9.10, 9.11).
All quizzes, matched or not, appear on `practice.md`.

Chapter titles come from `Study_Plan.md` section headings. Seven chapters have no
section there (1.8, 3.1C, 6.1C, 6.3C, 6.6C, 9.16, 9.17): 1.8, 3.1C, and 6.1C use the
Study Plan's table of contents; 6.3C and 6.6C use a cross-reference in the Study Plan;
9.16 and 9.17 use the quiz list. They are marked **§** on the Part pages.

### Embedded videos

Each chapter summary on a Part page has a collapsed **Videos** section, built like the code-example sections, with one
lazy-loaded `youtube-nocookie.com` player per video and a caption. Captions use the title and channel that YouTube
reports, not the wording in `videos/`, because the two often differ.

Not every link in `videos/` is embeddable, and the source files are cleaned periodically as a result. Each YouTube link
is checked with YouTube's public oEmbed endpoint (`https://www.youtube.com/oembed?url=…`), which returns the real title
or an error. On 2026-09-23, that check found 93 entries whose link was dead (36 unique links) or pointed at a real but
different video from the one the entry described, and those 93 entries were **removed from `videos/`** rather than left
in and merely un-embedded; four chapters lost their only video this way and now carry a short "Note:" line instead (the
existing convention in these files for a chapter with no direct link), in the same style as the pre-existing notes on,
for example, chapter 6.3A. `video-link-check.md` records what was removed and why, for salvage and so the same dead link
does not get proposed again.

Of what remains, a video is embedded when it is available **and** either its title shares at least half its meaningful
words with the entry's title, or it was accepted by hand after comparing the two titles (the exceptions are listed in the
cleanup and generator scripts, not in the repository). The two links whose uploader has disabled embedding are kept as
plain links rather than players. The three `nngroup.com` links are not YouTube, so they were never checked and are
always shown as plain links.

This is a title check only. It cannot tell whether a video is a good explanation, and a few embedded videos are only
loosely related to the chapter they sit under. It is also a snapshot — new links added later, or videos that go dead
after the check date, will not be caught until the check is re-run. Family-numbered sections (the `6.1` section of
Part 6) are embedded once, under the first chapter of the family.

### Prerequisite sub-topics and embedded videos

`prerequisites/essential.md`, `recommended.md`, and `beneficial.md` break each of the 15 prerequisite topics from
`Prerequisite_Knowledge.md` into specific sub-skills (a "What is a neural network" question is not the same
self-check as "Can you explain temperature", even though both sit under "LLM Fundamentals"). Two overlapping topics —
Essential's "RESTful APIs and HTTP Fundamentals" and Beneficial's "API Design and REST Principles" — were merged into
one Essential topic with a core group and a "going deeper" group, rather than duplicated across tiers.

Every sub-skill gets one curated resource, chosen for that specific sub-skill rather than reused across several: a
short video where a good one exists, otherwise a course, book, or official doc. **Videos are reused from this
repository's own `videos/*.md` library wherever a good match exists**, not sourced fresh — those files are already
oEmbed-verified (see "Embedded videos" above), so a title like "But what is a neural network?" already known to work
for [Part 5]({{ site.repo_blob }}/videos/Part_05_YoutubeVideos.md) is exactly as trustworthy when reused for the LLM
Fundamentals prerequisite. 14 videos are embedded this way across the three tier pages, using the identical
`youtube-nocookie.com` embed markup as the curriculum pages, each used exactly once (no sub-skill reuses another
sub-skill's video). Where no good match exists in the library — Python, Linux CLI, SQL, most of Database
Fundamentals — the sub-skill keeps a text resource instead of forcing a weak video fit.

`supplemental_course/` (a dataset of AI-literacy and governance courses for public-sector professionals) was
evaluated against every sub-topic and found to genuinely fit almost none of them: it's non-technical and
policy/governance-focused, with nothing on Python, Docker, Kubernetes, SQL, async programming, CI/CD, or the
technical depth of LLM/NLP/ML internals. It is not cited anywhere in the prerequisite pages.

The three tier pages cross-link each other by heading anchor (for example, Recommended's Kubernetes sub-topic points
back to `essential/#docker-and-containerization-basics`). Because kramdown's auto-generated heading IDs include any
leading text, tier-page topic headings (`## `) deliberately carry **no** leading number ("Kubernetes fundamentals",
not "3. Kubernetes fundamentals") — a numbered heading would slug to `#3-kubernetes-fundamentals` and break every
cross-reference to it.

### Deliberately not published

These are gitignored working notes and are not part of the site: `_cfc/`,
`guides/`, `workflows/`. The vendored third-party documentation under
`References/` (NeMo Agent Toolkit, Triton Inference Server) is also not
republished. The site does not link to it.

The site also avoids restating the maintainer's name and contact details found in
`GOVERNANCE.md` and `CALL_FOR_COLLABORATORS.md`; it links to those files instead.

## Known inconsistencies in the source files

Building the catalog turned up disagreements between files in the repository. The
site works around each one rather than hiding it; fixing them at the source would
simplify the pages.

| What | Detail | Effect on the site |
|---|---|---|
| Chapter lists differ | `nvidia_NCP-AAI.csv` has 94 chapters, including 1.8, 9.16, 9.17 but not 4.1 or 10.6; the other four CSVs have 93 with the reverse. `Study_Plan.md` has 89 sections. The union is 96. The Study Plan says "86 theory chapters"; the READMEs say 94. | Part pages use the union of the CSVs and the Study Plan; a chapter absent from a CSV shows no tag for that certification |
| README coverage table vs CSVs | In `cert_mapping/README.md`, 30 cells show 🟢 where the CSV has no **H** rating, and 5 show `-` where it does (of 460 cells compared) | Tags are computed from the CSVs |
| Quiz numbering | From Part 4 onward the quiz list uses older numbering: for example quiz 5.11 is *Procedural Memory* but chapter 5.11 is *Rule-Based Decision Making* | See matching rules above |
| Study Plan table of contents | Lists 1.8, 3.1C, and 6.1C, which have no section, and omits sections that exist (for example 6.4B, 6.5B, 6.6A, 7.2A); its introduction ends "Each chapter includes:" with nothing after it | Titles taken from headings first |
| Study Plan described as chapter text | The root `README.md` calls `Study_Plan.md` "the full chapter text". It is a study guide with per-chapter summaries, key concepts, and questions; the book's chapter text is not in the repository | The site calls it the reading guide |
| Exam title | `cert_mapping/README.md` labels AIP-C01 "AWS Certified: AI Practitioner", but the guide PDF beside it does not state a title | The site uses the bare code AIP-C01 |
| Prerequisite tiers | `Prerequisite_Knowledge.md` lists machine learning fundamentals as essential in its detailed section and as recommended in its checklist | The site follows the detailed section and notes the difference |
| AI tutor scope | `ai_tutor/README.md` says both that coverage is "certain part chapters" shared separately and that it is limited to "Parts 1–2" | The site states the Parts 1–2 wording and that coverage is limited |
| Slide files | The Book Club decks carry inconsistent titles ("Session 1", "Session x", "Week 9"), and one file is named `Chapter1.7B_1.8_v1.0_2026_03_01 (1).pdf` (a duplicate-download suffix) | The site lists their chapters instead of the deck titles |
| Videos | Parts 7–10 list far more entries than direct links; the rest are search suggestions. A 2026-09-23 link check found 93 entries whose link was dead or pointed at a real but different video (for example an entry titled "LangGraph Checkpointer - Game-Changer for AI Agents" linked to an unrelated video about marriage) | Those 93 entries were removed from `videos/`, not just left un-embedded; `video-link-check.md` has the full account |
| Review time | `CONTRIBUTING.md` and `SUPPORT.md` promise a first response in 3 business days; `CALL_FOR_COLLABORATORS.md` now says 3–5 | The site says "a few business days" |

## Local preview

You need a current Ruby (3.x); the Ruby 2.6 that ships with macOS is too old.

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000/MasteringAgenticAISystems_supplementals/>.

## Conventions

- Every page has front matter with `title`, and `nav_order` for top-level pages.
  Part pages use `parent: Curriculum`.
- Link to other pages with `{{ site.baseurl }}/target-permalink/`, matching the target
  page's own `permalink:` exactly — **not** `{% link page.md %}`. On this site (GitHub
  Pages, a project site served under `/MasteringAgenticAISystems_supplementals/`, with
  the `just-the-docs` theme), `{% link %}` resolves to the bare `page.url`, which does
  **not** include `baseurl` — every such link 404s on the live site (confirmed: e.g.
  `{% link curriculum/index.md %}` rendered as `https://gsa.github.io/curriculum/`
  instead of `https://gsa.github.io/MasteringAgenticAISystems_supplementals/curriculum/`
  — a bug that shipped in the first version of this site and reached every internal
  link in the body content of every page, ~122 of them, until caught from the live
  site and fixed). The theme's own auto-generated sidebar nav is unaffected — it
  correctly includes baseurl — so a broken body link and a working sidebar link can sit
  on the same page, which is what made this easy to miss from source alone. Link to
  repository files with `{{ site.repo_blob }}` / `{{ site.repo_tree }}`, which do not
  have this problem since they're absolute GitHub URLs, not baseurl-relative ones.
  Never use a relative path that leaves `docs/`: Jekyll cannot serve files outside this
  folder, and such links would break on the published site.
- **Verifying internal links only from source is not enough** — `{% link %}` looks
  exactly as correct as `{{ site.baseurl }}/…/` in the raw Markdown, and a check that
  only confirms the target file exists (as opposed to fetching the actual built page)
  cannot tell them apart. Check the live site directly after a Jekyll-affecting change:
  `curl` a rendered page and grep its `href=` attributes for anything under `/curriculum/`,
  `/videos/`, etc. that is *not* prefixed with `/MasteringAgenticAISystems_supplementals/`.
- Percent-encode spaces and `&` in file names (for example `Chapter1.1A%26B_...`).
- External links (Google Forms quizzes, YouTube) are copied from the source files
  as-is; this project does not verify that they resolve.
