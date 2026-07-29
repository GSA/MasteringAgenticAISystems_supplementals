# References and source materials

This directory contains third-party papers, documentation snapshots, curriculum PDFs, notebooks, configuration examples, and research notes used while developing *Mastering Agentic AI Systems*.

## Snapshot status

These files are a **partial, date-specific research snapshot**, not a complete mirror of each upstream project. Some copied documentation trees contain links to files that were not included, and some APIs or regulatory materials may have changed since capture. Prefer current upstream documentation for implementation decisions and use the local files for traceability and offline study.

The three historical index files were generated at different stages of the project. Their narrative counts and processing labels may not match the current tree. Generate an authoritative path-and-hash inventory with:

```bash
make inventory
```

The generated `References/reference_inventory.csv` is intentionally ignored by Git so reviewers can regenerate it from the current checkout.

## Current snapshot counts

Counts below were calculated from this repository snapshot and include files recursively within each chapter directory.

| Directory | Files |
|---|---:|
| Chapter 1 - Agent Architecture and Design | 94 |
| Chapter 2 - Agent Development | 141 |
| Chapter 3 - Evaluation and Tuning | 14 |
| Chapter 4 - Deployment and Scaling | 16 |
| Chapter 5 - Cognition, Planning, and Memory | 14 |
| Chapter 6 - Knowledge Integration and Data Handling | 2 |
| Chapter 7 - NVIDIA Platform Implementation | 35 |
| Chapter 8 - Run, Monitor, and Maintain | 16 |
| Chapter 9 - Safety, Ethics, and Compliance | 259 |
| Chapter 10 - Human-AI Interaction and Oversight | 13 |
| **Chapter directories** | **604** |
| Top-level files, including this README and indexes | 7 |
| **References directory total** | **611** |

The largest format groups are 328 Markdown files, 75 PDFs, 73 reStructuredText files, 36 PNGs, 17 notebooks, 10 Python files, 10 YAML files, 10 PlantUML files, and 10 protobuf-text files. The remaining files use other formats or have no extension.

## Indexes

- `reference_file_index_ch1-3.md`
- `reference_file_index_ch4-7.md`
- `reference_file_index_ch8-10.md`

Treat those files as historical research aids. For current counts, paths, sizes, and SHA-256 hashes, use the generated inventory.

## Reuse and attribution

Individual files retain their original rights and terms. Inclusion here does not place a third-party work under the repository's CC0 dedication. Review the source file, adjacent notices, and the original publisher before redistribution or adaptation; also see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
