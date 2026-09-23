---
title: "Part 10 — Human-in-the-Loop & Integration"
parent: Curriculum
nav_order: 10
permalink: /curriculum/part-10/
---

# Part 10 — Human-in-the-Loop & Integration
{: .no_toc }

7 chapters · 38.3 study hours allocated in the Study Plan · 0 slide decks · 2 videos · 4 code example files

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Chapters

Rating tags show which certification knowledge maps rate the chapter **H** (highly relevant) in at least one item: **NV** NCP-AAI · **AWS** AIP-C01 · **DBX** Databricks GenAI Engineer · **GCP** Professional ML Engineer · **MS** AI-102. See [Certifications]({{ site.baseurl }}/certifications/).

| Ch. | Title | Hours | Slides | Quiz | Videos | Figures | Code | H-rated for |
|---|---|---:|---|---|---:|---|---:|---|
| 10.1 | [Conversational UI]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-101-conversational-ui) | 4.4 | — | [Quiz](https://docs.google.com/forms/d/13rN-HUmcylPaElvnlfOfjIFbUKKoY29F5wBykwKGvhY/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-101-conversational-ui) | [10]({{ site.repo_tree }}/figures/Ch10.1_figures_v11MAY26) | — | NV AWS DBX GCP MS |
| 10.2 | [Proactive Agents]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-102-proactive-agents) | 6.2 | — | [Quiz](https://docs.google.com/forms/d/1JfhTZi64odDKNg6bxGdDK6Tzke4LJ6H-l80zreuBmzw/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-102-proactive-agents) | [6]({{ site.repo_tree }}/figures/Ch10.2_figures_v11MAY26) | — | NV AWS DBX GCP MS |
| 10.3A | [RLHF Methodology]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-103a-rlhf-methodology) | 5.3 | — | [Quiz 10.3†](https://docs.google.com/forms/d/1kfKFluEL-PTpI3MlNeYHK6D6zJ2MF44TFe40gaS-EWA/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-103a-rlhf-methodology) | [12†]({{ site.repo_tree }}/figures/Ch10.3_figures_v11MAY26) | — | NV AWS DBX GCP MS |
| 10.3B | [RLHF Pitfalls and Red Teaming]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-103b-rlhf-pitfalls-and-red-teaming) | 3.9 | — | — | [0]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-103b-rlhf-pitfalls) | [12†]({{ site.repo_tree }}/figures/Ch10.3_figures_v11MAY26) | — | NV AWS DBX MS |
| 10.4 | [Human-in-the-Loop]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-104-human-in-the-loop) | 5.5 | — | [Quiz](https://docs.google.com/forms/d/1S9-1L6ueaO3E7ApGEXekHlKI74rXTce7v-P5TtQn_eI/viewform?usp=sharing) | [2]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-104-human-in-the-loop) | [11]({{ site.repo_tree }}/figures/Ch10.4_figures_v11MAY26) | — | NV AWS DBX GCP MS |
| 10.5 | [Human-over-the-Loop]({{ site.repo_blob }}/Study_Plan.md#part-10-chapter-105-human-over-the-loop) | 8.1 | — | [Quiz](https://docs.google.com/forms/d/1ePcVBWqiyScIys77ZiU0F1OQaGRndloJJIZr30tM80Y/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md#chapter-105-human-over-the-loop) | [12]({{ site.repo_tree }}/figures/Ch10.5_figures_v11MAY26) | — | NV AWS DBX GCP MS |
| 10.6 | [Integration (Feedback, Calibration, Explainability, Controllability, Consistency)]({{ site.repo_blob }}/Study_Plan.md#part-10-sections-106-integration-feedback-calibration-explainability-controllability-consistency) | 4.9 | — | — | — | — | — | AWS DBX GCP MS |


**Notes.** The Videos column counts the videos shown under each chapter summary below, out of the unique direct links in [`Part_10_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md) ("3 of 5"). A video is left out when its link is dead, embedding is disabled, or YouTube's title does not match the entry; see the [link check]({{ site.baseurl }}/videos/link-check/). Chapters can also list search suggestions instead of links.

† Linked by chapter-family number, not an exact ID match: the deck, quiz, or figure set is numbered differently from this chapter in the source files (for example a quiz or deck numbered `6.2` for chapters `6.2A` and `6.2B`).

‡ A combined deck that covers more than one chapter.

A chapter that is missing from a certification's mapping file shows no tag for that certification: the NVIDIA file omits 4.1 and 10.6, and the other four omit 1.8, 9.16, and 9.17.


## Chapter summaries


Summaries are excerpted from [`Study_Plan.md`]({{ site.repo_blob }}/Study_Plan.md), which also lists each chapter's key concepts and self-check questions.


### 10.1. Conversational UI

Conversational user interfaces fundamentally shift interaction from constrained navigation menus to natural language dialogue, processing plain language input while maintaining conversation history across multiple turns through hierarchical memory architectures. This transformation dramatically improves accessibility and reduces cognitive load, particularly benefiting users with limited technical literacy or accessibility needs.

_No videos are shown for this chapter: its list has only search suggestions, or its links failed the [link check]({{ site.baseurl }}/videos/link-check/)._

### 10.2. Proactive Agents

Proactive AI agents shift from traditional pull-based user-initiated interaction to push-based systems delivering timely assistance when users need it most by continuously monitoring environments and analyzing patterns across temporal and contextual dimensions. This transformation enables preventative assistance where issues surface before escalation and opportunities materialize before user recognition.

_No videos are shown for this chapter: its list has only search suggestions, or its links failed the [link check]({{ site.baseurl }}/videos/link-check/)._

### 10.3A. RLHF Methodology

Reinforcement Learning from Human Feedback addresses fundamental asymmetry in human cognition where humans excel at recognizing preferences through comparative judgment while struggling to specify desired behavior exhaustively through formal rules. RLHF translates this comparative strength into three-phase training pipeline transforming pre-trained models into systems understanding language and responding in helpful, harmless, and honest ways through preference-based optimization.

_No videos are shown for this chapter: its list has only search suggestions, or its links failed the [link check]({{ site.baseurl }}/videos/link-check/)._

### 10.3B. RLHF Pitfalls and Red Teaming

This chapter exposes twelve critical misconceptions about RLHF that organizations frequently hold, including the dangerous assumption that RLHF solves alignment completely. It provides systematic analysis of preference variation, annotation quality challenges, reward model limitations, and introduces red teaming methodologies for identifying vulnerabilities before production deployment.

_No videos are shown for this chapter: its list has only search suggestions, or its links failed the [link check]({{ site.baseurl }}/videos/link-check/)._

### 10.4. Human-in-the-Loop

This chapter explains Human-in-the-Loop (HITL) approval mechanisms where agent execution halts pending explicit human validation. It covers three-phase approval architectures, graduated autonomy frameworks, state persistence approaches, escalation pathways, and real-world deployment scenarios from healthcare to financial services where organizational oversight maintains authority over consequential outcomes.

<details markdown="block">
<summary>Videos (2)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/7xTGNNLPyMI" title="Deep Dive into LLMs like ChatGPT" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI">Deep Dive into LLMs like ChatGPT</a> &middot; Andrej Karpathy</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Za8CrPqQxpA" title="LangGraph Agents - Human-In-The-Loop Breakpoints" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Za8CrPqQxpA">LangGraph Agents - Human-In-The-Loop Breakpoints</a> &middot; LangChain</div>
</div>

</details>

### 10.5. Human-over-the-Loop

This chapter presents the Human-over-the-Loop (HOvL) governance paradigm that balances agent autonomy with human accountability through policy-based constraints. Rather than requiring approval for every decision, HOvL encodes organizational wisdom into policies that agents respect automatically, eliminating real-time approval bottlenecks while maintaining explicit veto power and continuous learning through RLHF integration.

_No videos are shown for this chapter: its list has only search suggestions, or its links failed the [link check]({{ site.baseurl }}/videos/link-check/)._

### 10.6. Integration (Feedback, Calibration, Explainability, Controllability, Consistency)

This integration chapter synthesizes five foundational capabilities—feedback integration, calibrated confidence, explainability, controllability, and consistent behavior—that work synergistically to create trustworthy human-AI collaborative systems. It covers feedback pipelines incorporating corrections into both immediate context and parametric knowledge, techniques for aligning stated confidence with actual reliability, explainability mechanisms serving diverse stakeholders, controllability infrastructure preserving human authority, and multi-method drift detection ensuring consistent production performance.

### Other code examples

These files are named for a chapter number that is not in the current chapter list (an older numbering), so they are not attached to a chapter above.

<details markdown="block">
<summary>4 files</summary>

- [`Part_10_Chapter_10.3_Human_in_the_Loop_code_01_telemetry.py`]({{ site.repo_blob }}/code_examples/Part_10_Chapter_10.3_Human_in_the_Loop_code_01_telemetry.py)
- [`Part_10_Chapter_10.3_Human_in_the_Loop_code_02_baseline_monitor.py`]({{ site.repo_blob }}/code_examples/Part_10_Chapter_10.3_Human_in_the_Loop_code_02_baseline_monitor.py)
- [`Part_10_Chapter_10.3_Human_in_the_Loop_code_03_alert_manager.py`]({{ site.repo_blob }}/code_examples/Part_10_Chapter_10.3_Human_in_the_Loop_code_03_alert_manager.py)
- [`Part_10_Chapter_10.3_Human_in_the_Loop_code_04_agent_controller.py`]({{ site.repo_blob }}/code_examples/Part_10_Chapter_10.3_Human_in_the_Loop_code_04_agent_controller.py)

</details>

### Additional worked examples

From [`more_examples/part_10/`]({{ site.repo_tree }}/more_examples/part_10):

- [`explanation_generator.py`]({{ site.repo_blob }}/more_examples/part_10/explanation_generator.py)
- [`traceable_agent.py`]({{ site.repo_blob }}/more_examples/part_10/traceable_agent.py)

### Labs

No lab or legacy example exists for this Part yet. See [Labs]({{ site.baseurl }}/labs/) and [Contributing]({{ site.baseurl }}/contributing/).
