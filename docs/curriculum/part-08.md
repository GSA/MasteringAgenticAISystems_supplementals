---
title: "Part 8 — Reliability & Cost Management"
parent: Curriculum
nav_order: 8
permalink: /curriculum/part-08/
---

# Part 8 — Reliability & Cost Management
{: .no_toc }

5 chapters · 10.9 study hours allocated in the Study Plan · 0 slide decks · 2 direct video links · 29 code example files

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## Chapters

Rating tags show which certification knowledge maps rate the chapter **H** (highly relevant) in at least one item: **NV** NCP-AAI · **AWS** AIP-C01 · **DBX** Databricks GenAI Engineer · **GCP** Professional ML Engineer · **MS** AI-102. See [Certifications]({% link certifications.md %}).

| Ch. | Title | Hours | Slides | Quiz | Videos | Figures | Code | H-rated for |
|---|---|---:|---|---|---:|---|---:|---|
| 8.1 | [Latency Fundamentals]({{ site.repo_blob }}/Study_Plan.md#part-8-chapter-81-latency-fundamentals) | 2.2 | — | [Quiz](https://docs.google.com/forms/d/1c2DlRZhVBJvfB5OskMtG6lyOm9PSkcQl-nzg3u3LR4s/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md#chapter-81-latency-fundamentals) | [5]({{ site.repo_tree }}/figures/Ch8.1_figures_v11MAY26) | 8 | NV AWS GCP |
| 8.2A | [Error Taxonomy and SLO]({{ site.repo_blob }}/Study_Plan.md#part-8-chapter-82a-error-taxonomy-and-slo) | 3.5 | — | [Quiz](https://docs.google.com/forms/d/13HhhUa2efm3tWRd5X6l3P1m7ht128IRee5R1C6IF7EI/viewform?usp=sharing) | [2]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md#chapter-82a-error-taxonomy-and-slos) | [10]({{ site.repo_tree }}/figures/Ch8.2A_figures_v11MAY26) | 8 | NV AWS GCP |
| 8.2B | [Circuit Breakers and NeMo Integration]({{ site.repo_blob }}/Study_Plan.md#part-8-chapter-82b-circuit-breakers-and-nemo-integration) | 1.2 | — | [Quiz](https://docs.google.com/forms/d/1fqOZUSK7ZEDr6V_4x7RmpumvkJyGD3DAhLeAvwglA6M/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md#chapter-82b-circuit-breakers-and-nemo-guardrails) | [11]({{ site.repo_tree }}/figures/Ch8.2B_figures_v11MAY26) | 5 | NV AWS DBX GCP MS |
| 8.3 | [Token Economics]({{ site.repo_blob }}/Study_Plan.md#part-8-chapter-83-token-economics) | 3.2 | — | [Quiz](https://docs.google.com/forms/d/1BbZ7sbP0Fh_-NhvEvA0I9bL1gfLweEND5v9zLR8GJ_k/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md#chapter-83-token-economics) | [10]({{ site.repo_tree }}/figures/Ch8.3_figures_v11MAY26) | 8 | NV AWS GCP |
| 8.4 | [Success Metrics]({{ site.repo_blob }}/Study_Plan.md#part-8-chapter-84-success-metrics) | 0.8 | — | [Quiz](https://docs.google.com/forms/d/1xbI_UW3aLlDWQcQSJZ12GTkAz78PP0rMUouJLDfPIdU/viewform?usp=sharing) | [0]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md#chapter-84-success-metrics) | [7]({{ site.repo_tree }}/figures/Ch8.4_figures_v11MAY26) | — | NV AWS GCP MS |


**Notes.** Video counts are unique direct links in [`Part_08_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md); a chapter can list search suggestions instead of links, so a low number does not mean the chapter is skipped.

† Linked by chapter-family number, not an exact ID match: the deck, quiz, or figure set is numbered differently from this chapter in the source files (for example a quiz or deck numbered `6.2` for chapters `6.2A` and `6.2B`).

‡ A combined deck that covers more than one chapter.

A chapter that is missing from a certification's mapping file shows no tag for that certification: the NVIDIA file omits 4.1 and 10.6, and the other four omit 1.8, 9.16, and 9.17.


## Chapter summaries


Summaries are excerpted from [`Study_Plan.md`]({{ site.repo_blob }}/Study_Plan.md), which also lists each chapter's key concepts and self-check questions.


### 8.1. Latency Fundamentals

Agent latency monitoring requires simultaneous tracking of end-to-end metrics and granular per-step measurements to distinguish between average performance that masks outliers and percentile-based metrics revealing true user experience. From diagnosis through distributed tracing to GPU-level observability, this chapter provides the comprehensive measurement framework necessary for production optimization.

<details markdown="block">
<summary>Code examples (8 files)</summary>

- [`agent_tracing_code_04_instrumented_agent_execution.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_agent_tracing_code_04_instrumented_agent_execution.py)
- [`inventory_caching_code_05_caching_optimization.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_inventory_caching_code_05_caching_optimization.py)
- [`metrics_collection_code_08_gpu_metrics_collector.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_metrics_collection_code_08_gpu_metrics_collector.py)
- [`nvidia_dcgm_monitoring_code_06_gpu_metrics_initialization.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_nvidia_dcgm_monitoring_code_06_gpu_metrics_initialization.py)
- [`opentelemetry_imports_code_01_imports_setup.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_opentelemetry_imports_code_01_imports_setup.py)
- [`otlp_exporter_setup_code_03_exporter_configuration.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_otlp_exporter_setup_code_03_exporter_configuration.py)
- [`prometheus_metrics_code_07_prometheus_gauge_setup.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_prometheus_metrics_code_07_prometheus_gauge_setup.py)
- [`tracer_provider_config_code_02_tracer_initialization.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.1_tracer_provider_config_code_02_tracer_initialization.py)

</details>

### 8.2A. Error Taxonomy and SLO

This chapter provides a systematic framework for categorizing AI agent failures into three tiers (planning, execution, verification) and using Service Level Objectives (SLOs) with error budgets and burn rate metrics to make reliability-velocity tradeoffs explicit and measurable. It also covers multi-agent coordination failures and distributed tracing techniques for diagnosing invisible failure patterns in concurrent systems.

<details markdown="block">
<summary>Code examples (8 files)</summary>

- [`agent_execution_with_dependency_tracking_code_04_agent_execution_with_dependency_tracking.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_agent_execution_with_dependency_tracking_code_04_agent_execution_with_dependency_tracking.py)
- [`burn_rate_alerting_configuration_code_02_burn_rate_alerting_configuration.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_burn_rate_alerting_configuration_code_02_burn_rate_alerting_configuration.py)
- [`dependency_graph_with_cycle_detection_code_06_dependency_graph_with_cycle_detection.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_dependency_graph_with_cycle_detection_code_06_dependency_graph_with_cycle_detection.py)
- [`optimistic_locking_with_retry_code_08_optimistic_locking_with_retry.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_optimistic_locking_with_retry_code_08_optimistic_locking_with_retry.py)
- [`ordered_execution_with_barriers_code_07_ordered_execution_with_barriers.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_ordered_execution_with_barriers_code_07_ordered_execution_with_barriers.py)
- [`research_orchestrator_instrumentation_code_03_research_orchestrator_instrumentation.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_research_orchestrator_instrumentation_code_03_research_orchestrator_instrumentation.py)
- [`shared_workflow_state_tracing_code_05_shared_workflow_state_tracing.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_shared_workflow_state_tracing_code_05_shared_workflow_state_tracing.py)
- [`tracer_configuration_initialization_code_01_tracer_configuration_initialization.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2A_tracer_configuration_initialization_code_01_tracer_configuration_initialization.py)

</details>

### 8.2B. Circuit Breakers and NeMo Integration

This chapter addresses how circuit breakers prevent cascading failures in distributed systems through fast-fail behavior, and how to categorize production errors into safety violations versus infrastructure failures for proper team escalation and monitoring. The practical focus includes implementing a three-state circuit breaker automaton and designing separate monitoring pipelines that distinguish NeMo Guardrails safety blocks from execution exceptions.

<details markdown="block">
<summary>Code examples (5 files)</summary>

- [`complete_solution_04_full_circuit_breaker.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2B_complete_solution_04_full_circuit_breaker.py)
- [`moderate_hints_02_call_method_structure.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2B_moderate_hints_02_call_method_structure.py)
- [`nemo_guardrails_integration_05_error_classification.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2B_nemo_guardrails_integration_05_error_classification.py)
- [`scaffolded_code_01_circuit_breaker_setup.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2B_scaffolded_code_01_circuit_breaker_setup.py)
- [`strong_hints_03_half_open_and_closed_logic.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.2B_strong_hints_03_half_open_and_closed_logic.py)

</details>

### 8.3. Token Economics

Token economics fundamentally shape LLM cost optimization strategies through asymmetric pricing, where output tokens cost 4-5× more than input tokens due to computational differences between single-pass encoding and iterative decoding. This chapter establishes a three-tier monitoring architecture and demonstrates how systematic multi-faceted optimizations can achieve significant cost reductions while maintaining quality metrics.

<details markdown="block">
<summary>Code examples (8 files)</summary>

- [`feature_level_analysis_code_03_feature_level_analysis.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_feature_level_analysis_code_03_feature_level_analysis.py)
- [`model_routing_complexity_code_08_model_routing_complexity.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_model_routing_complexity_code_08_model_routing_complexity.py)
- [`output_constraint_code_06_output_constraint.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_output_constraint_code_06_output_constraint.py)
- [`output_format_guidance_code_07_output_format_guidance.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_output_format_guidance_code_07_output_format_guidance.py)
- [`prompt_caching_implementation_code_04_prompt_caching_implementation.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_prompt_caching_implementation_code_04_prompt_caching_implementation.py)
- [`rag_context_retrieval_code_05_rag_context_retrieval.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_rag_context_retrieval_code_05_rag_context_retrieval.py)
- [`request_level_tracking_code_01_request_level_tracking.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_request_level_tracking_code_01_request_level_tracking.py)
- [`token_tracking_infrastructure_code_02_token_tracking_infrastructure.py`]({{ site.repo_blob }}/code_examples/Part_08_Chapter_8.3_token_tracking_infrastructure_code_02_token_tracking_infrastructure.py)

</details>

### 8.4. Success Metrics

This chapter explores multi-dimensional measurement of AI agent success through balanced scorecards that track task completion, user satisfaction, efficiency, and safety metrics simultaneously. Rather than optimizing for single metrics in isolation, production systems must measure across complementary dimensions to prevent optimization pathologies that degrade unmeasured but equally important success factors.

### Additional worked examples

From [`more_examples/part_08/`]({{ site.repo_tree }}/more_examples/part_08):

- [`custom_dashboard.py`]({{ site.repo_blob }}/more_examples/part_08/custom_dashboard.py)
- [`grafana_dashboard.json`]({{ site.repo_blob }}/more_examples/part_08/grafana_dashboard.json)

### Labs

Chapter 8.2B has the project's first lab written to the lab template: [Build a Circuit Breaker for a Flaky Downstream Tool]({{ site.repo_blob }}/labs/8.2B_circuit_breaker/lab.md) (status: draft). See [Labs]({% link labs.md %}).
