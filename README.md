# Mastering Agentic AI Systems — Supplemental Materials

This repository contains the study plan, labs, illustrative code, figures, slide decks, quizzes, certification mappings, reference snapshots, and curated video resources that accompany **Mastering Agentic AI Systems**.

The repository is an educational collection rather than a single installable application. Many code files are intentionally small excerpts that are meant to be read in sequence with the corresponding chapter. Start with the [study plan](Study_Plan.md), then use the chapter-matched labs, figures, videos, and examples.

## Repository map

| Path | Contents |
|---|---|
| `Study_Plan.md` | Primary narrative study plan and chapter objectives |
| `labs/` | Guided exercises, practice questions, and lab briefs |
| `code_examples/` | Illustrative implementation excerpts and runnable examples |
| `figures/` | Chapter figures and diagrams |
| `slides/` | Published slide decks currently available for Parts 1–6 |
| `videos/` | Curated video and learning resources, catalog, and review queue |
| `quizzes_ver20JUN26.md` | Quiz index and Google Forms links |
| `simulated_tests_ver20JUN26.md` | Simulated assessment links |
| `cert_mapping/` | Chapter-to-certification mapping matrices and source guides |
| `References/` | Third-party research and documentation snapshots |
| `ai_tutor/` | Design guidance and example prompts for a tutor implementation |

## Validate the repository

The validators use only the Python standard library and check first-party local links and anchors, Markdown fences and Python/JSON code blocks, structured files, Python syntax, text-file hygiene, accidental private paths, repository metadata, the study-plan table of contents, the video status registry, and whether generated video catalogs are current.

```bash
make validate
```

Optional pre-commit checks can be installed with:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The code examples do not share one dependency set. Review [`code_examples/README.md`](code_examples/README.md) and [`code_examples/COMPATIBILITY.md`](code_examples/COMPATIBILITY.md) before executing an example.

## Known editorial and technical caveats

- The study plan, labs, quizzes, slide filenames, and certification matrices contain overlapping curriculum versions. The study-plan chapter headings are the primary narrative source until the numbering is reconciled.
- Some framework examples demonstrate historical APIs. The compatibility register identifies examples that require migration before use with current releases.
- Most PDFs in this snapshot require accessibility remediation, including document tags, reading-order review, and alternative text. Prefer an equivalent Markdown or HTML source when one is available.
- External links and certification blueprints change over time. Verify current provider documentation before relying on a link, exam objective, image tag, price, or performance claim.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md). Run `make validate` before submitting a pull request.

## Licensing and third-party material

Original project material is released under the terms in [LICENSE.md](LICENSE.md). Files under `References/`, certification source documents, and other attributed third-party assets retain their original rights and terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
