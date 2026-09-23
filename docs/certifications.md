---
title: Certifications
nav_order: 5
permalink: /certifications/
---

# Certification mappings
{: .no_toc }

The book was written for **NVIDIA NCP-AAI**. Its coverage of agentic AI, retrieval, deployment, evaluation, safety,
and governance also overlaps four other certifications. The [`cert_mapping/`]({{ site.repo_tree }}/cert_mapping) folder rates every
chapter against every knowledge item of each exam.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## The rating scale

| Level | Meaning |
|---|---|
| **H** — highly relevant | Mastering the chapter directly prepares you for questions on this knowledge item; the chapter covers it comprehensively. |
| **M** — moderate | The chapter is supplementary; combined with others it contributes meaningfully. |
| **L** — low | The content is tangential or minimal. |
| **N** — not relevant | The chapter does not address the item. |

## Mapped certifications

| Certification | Mapping (CSV) | Exam guide / knowledge items | Chapters mapped | Items rated | Chapters with ≥1 **H** |
|---|---|---|---:|---:|---:|
| **NVIDIA NCP-AAI**<br>The book's primary target exam | [`nvidia_NCP-AAI.csv`]({{ site.repo_blob }}/cert_mapping/nvidia_NCP-AAI.csv) | [`nvt-study-guide-new-agentic-ai-cert-exam-4230000.pdf`]({{ site.repo_blob }}/cert_mapping/nvt-study-guide-new-agentic-ai-cert-exam-4230000.pdf) | 94 | 53 | 94 |
| **AWS AIP-C01** | [`aws_AIP-C01.csv`]({{ site.repo_blob }}/cert_mapping/aws_AIP-C01.csv) | [`aws_AIP-C01.pdf`]({{ site.repo_blob }}/cert_mapping/aws_AIP-C01.pdf) | 93 | 98 | 92 |
| **Databricks Generative AI Engineer Associate** | [`databricks_genAI_EngAsc.csv`]({{ site.repo_blob }}/cert_mapping/databricks_genAI_EngAsc.csv) | [`databricks_genAI_EngAsc.md`]({{ site.repo_blob }}/cert_mapping/databricks_genAI_EngAsc.md) | 93 | 54 | 31 |
| **Google Cloud PMLE**<br>Professional Machine Learning Engineer | [`google_PMLE.csv`]({{ site.repo_blob }}/cert_mapping/google_PMLE.csv) | [`google_PMLE.md`]({{ site.repo_blob }}/cert_mapping/google_PMLE.md) | 93 | 36 | 79 |
| **Microsoft AI-102**<br>Azure AI Engineer Associate | [`microsoft_AI-102.csv`]({{ site.repo_blob }}/cert_mapping/microsoft_AI-102.csv) | [`microsoft_AI‑102.md`]({{ site.repo_blob }}/cert_mapping/microsoft_AI%E2%80%91102.md) | 93 | 106 | 86 |

Each CSV has one row per chapter and one column per knowledge item, holding H, M, L, or N. Open the CSV for the
full grid; the [Part pages]({{ site.baseurl }}/curriculum/) show, for every chapter, which certifications rate it **H**
on at least one item.

**Not yet mapped.** The exam guide for Google's *Generative AI Leader* certification is in the folder
([`google_genAI_leader.pdf`]({{ site.repo_blob }}/cert_mapping/google_genAI_leader.pdf)) but has no mapping yet. Mapping it is an
open [contribution track]({{ site.baseurl }}/contributing/).

## Using the maps

- **NCP-AAI:** the material is designed around this exam, so every chapter contributes.
- **Another certification:** start with the chapters rated **H** for your exam, then use the CSV to find chapters rated
  **M** to fill the remaining knowledge items.

## A caution about chapter lists

The five CSVs do not list exactly the same chapters. The NVIDIA CSV includes chapters 1.8, 9.16, and 9.17 and omits
4.1 and 10.6; the other four include 4.1 and 10.6 and omit those three. The per-chapter summary table in
[`cert_mapping/README.md`]({{ site.repo_blob }}/cert_mapping/README.md) also differs from the CSVs in some cells. The tags on this site
are computed from the CSVs, which are the primary data.
