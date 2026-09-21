# `docs/` — GitHub Pages source

This folder is the source of the project website for
[Mastering Agentic AI Systems — Supplementals](../README.md). It is written for
[GitHub Pages](https://docs.github.com/en/pages) (Jekyll, with the
[Just the Docs](https://just-the-docs.com/) theme) and is also readable as plain
Markdown when browsing it on GitHub.

If you are a learner, you do not need this file — start at the site home page
(`index.md`) or the repository [README](../README.md).

## Publishing

1. In the repository, go to **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select branch `master` and folder `/docs`, then save.

The site is then served at
`https://gsa.github.io/MasteringAgenticAISystems_supplementals/`. That address is
set by `url` and `baseurl` in [`_config.yml`](_config.yml); change both if the
repository moves or is forked.

## Layout

```text
docs/
├── README.md            This file (excluded from the built site)
├── _config.yml          Jekyll + theme configuration, and the repo link variables
├── Gemfile              Local-preview dependencies (ignored by GitHub Pages)
├── index.md             Site home
├── getting-started.md   Study path and how to use the material
├── prerequisites.md     Summary of Prerequisite_Knowledge.md
├── curriculum/
│   ├── index.md         The 10 Parts at a glance
│   └── part-01.md … part-10.md   One page per Part: chapter-by-chapter resource table
├── certifications.md    Certification mappings (NVIDIA, AWS, Databricks, Google, Microsoft)
├── practice.md          Chapter quizzes and full-length practice exams
├── videos.md            Curated third-party videos, per Part
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
every page links to them.

Links to repository files are built from two variables in `_config.yml`, so the
whole site can be re-pointed by editing one line:

```liquid
{{ site.repo_blob }}/slides/Ch3.2_v20MAR26.pdf   {% comment %} file view {% endcomment %}
{{ site.repo_tree }}/figures                     {% comment %} directory view {% endcomment %}
```

| Page | Built from |
|---|---|
| `curriculum/part-NN.md` | Chapter IDs from `cert_mapping/nvidia_NCP-AAI.csv` and the other four CSVs; chapter titles from the section headings in `Study_Plan.md`; resources matched by file name in `slides/`, `figures/`, `code_examples/`, and by row in `quizzes_ver20JUN26.md` |
| `certifications.md` | `cert_mapping/README.md` and the five CSVs |
| `practice.md` | `quizzes_ver20JUN26.md`, `simulated_tests_ver20JUN26.md` |
| `videos.md` | `videos/Part_NN_YoutubeVideos.md` |
| `labs.md` | `labs/README.md`, `labs/LAB_TEMPLATE.md` |
| `ai-tutor.md` | `ai_tutor/README.md` |
| `prerequisites.md` | `Prerequisite_Knowledge.md` |
| `contributing.md`, `about.md` | `CONTRIBUTING.md`, `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `LICENSE.md`, `CITATION.cff`, `.all-contributorsrc` |

Because these are snapshots of the sources at the time they were written, they
will drift as the sources change. When you change a source file, update the
matching page in the same pull request.

### Deliberately not published

These are gitignored working notes and are not part of the site: `_cfc/`,
`guides/`, `workflows/`. The vendored third-party documentation under
`References/` (NeMo Agent Toolkit, Triton Inference Server) is also not
republished; the site links to the folder only.

## Local preview

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000/MasteringAgenticAISystems_supplementals/>.

## Conventions

- Every page has front matter with `title`, and `nav_order` for top-level pages.
  Part pages use `parent: Curriculum`.
- Link to repository files with `{{ site.repo_blob }}` / `{{ site.repo_tree }}`,
  never with a relative path that leaves `docs/` — Jekyll cannot serve files
  outside this folder, and such links would break on the published site.
- Percent-encode spaces and `&` in file names (for example `Chapter1.1A%26B_...`).
- External links (Google Forms quizzes, YouTube) are copied from the source files
  as-is; this project does not verify that they resolve.
