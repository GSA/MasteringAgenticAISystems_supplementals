---
title: "Part 4 — Production Deployment & Scaling"
parent: Curriculum
nav_order: 4
permalink: /curriculum/part-04/
---

# Part 4 — Production Deployment & Scaling
{: .no_toc }

7 chapters · 24.4 study hours allocated in the Study Plan · 7 slide decks · 45 direct video links · 45 code example files

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
| 4.1 | [AI Agent Deployment and Scaling]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-41-ai-agent-deployment-and-scaling) | 3.2 | [PDF]({{ site.repo_blob }}/slides/Ch4.1_v20MAR26.pdf) | [Quiz 4.1A†](https://docs.google.com/forms/d/1if2c8Qu-gzjVf7NkPtEV6hPmgCpQcGauEoaFp-zCoGs/viewform?usp=sharing), [Quiz 4.1B†](https://docs.google.com/forms/d/1AGZ2H6dIePFVzdv0nsS_wM0x6BjH6UhC8IdAh-Sp49A/viewform?usp=sharing) | [11]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-41---ai-agent-deployment-and-scaling) | [6]({{ site.repo_tree }}/figures/Ch4.1_figures_v11MAY26) | — | AWS GCP MS |
| 4.2 | [Deployment & Scaling]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-42-deployment--scaling) | 5.8 | [PDF]({{ site.repo_blob }}/slides/Ch4.2_v20MAR26.pdf) | — | [8]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-42---deployment--scaling-architecture) | [12]({{ site.repo_tree }}/figures/Ch4.2_figures_v11MAY26) | 4 | NV AWS GCP |
| 4.3 | [Container Orchestration and Edge Deployment]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-43-container-orchestration-and-edge-deployment) | 2.1 | [PDF]({{ site.repo_blob }}/slides/Ch4.3_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1zf3fb2qlxLbs6Z6ARpbrcUdp-T5OE6hmEP9wMZC3bjw/viewform?usp=sharing) | [9]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-43---container-orchestration-and-edge-deployment) | [12]({{ site.repo_tree }}/figures/Ch4.3_figures_v11MAY26) | 7 | NV AWS GCP MS |
| 4.4 | [Performance Profiling and Optimization]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-44-performance-profiling-and-optimization) | 6.0 | [PDF]({{ site.repo_blob }}/slides/Ch4.4_v20MAR26.pdf) | — | [7]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-44---performance-profiling-and-optimization) | [12]({{ site.repo_tree }}/figures/Ch4.4_figures_v11MAY26) | 12 | NV AWS GCP MS |
| 4.5 | [NVIDIA NIM and Triton Inference Server]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-45-nvidia-nim-and-triton-inference-server) | 2.1 | [PDF]({{ site.repo_blob }}/slides/Ch4.5_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1Z1xLfMAdzT0IJibUl0cGqfRD7Hh1Pb2CY2SCUcGJL6Q/viewform?usp=sharing) | [6]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-45---nvidia-nim-and-triton-inference-server) | [6]({{ site.repo_tree }}/figures/Ch4.5_figures_v11MAY26) | — | NV AWS GCP MS |
| 4.6 | [TensorRT-LLM and NVIDIA Fleet Command]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-46-tensorrt-llm-and-nvidia-fleet-command) | 1.5 | [PDF]({{ site.repo_blob }}/slides/Ch4.6_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1ak-7FD_KYAcAfbrWxUm_y2n3pyBXM0vvrnu5ZjYEkVE/viewform?usp=sharing) | [9]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-46---tensorrt-llm-and-nvidia-fleet-command) | [9]({{ site.repo_tree }}/figures/Ch4.6_figures_v11MAY26) | — | NV AWS GCP MS |
| 4.7 | [Scaling Strategies]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-47-scaling-strategies) | 3.7 | [PDF]({{ site.repo_blob }}/slides/Ch4.7_v20MAR26.pdf) | — | [4]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-47---scaling-strategies) | [6]({{ site.repo_tree }}/figures/Ch4.7_figures_v11MAY26) | — | NV AWS GCP MS |


**Notes.** Video counts are unique direct links in [`Part_04_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md); a chapter can list search suggestions instead of links, so a low number does not mean the chapter is skipped.

† Linked by chapter-family number, not an exact ID match: the deck, quiz, or figure set is numbered differently from this chapter in the source files (for example a quiz or deck numbered `6.2` for chapters `6.2A` and `6.2B`).

‡ A combined deck that covers more than one chapter.

A chapter that is missing from a certification's mapping file shows no tag for that certification: the NVIDIA file omits 4.1 and 10.6, and the other four omit 1.8, 9.16, and 9.17.


## Chapter summaries


Summaries are excerpted from [`Study_Plan.md`]({{ site.repo_blob }}/Study_Plan.md), which also lists each chapter's key concepts and self-check questions.


### 4.1. AI Agent Deployment and Scaling

This chapter introduces the essential infrastructure and operational practices for deploying and scaling multi-agent systems in production, covering message queue architectures, vector database selection, observability patterns, API gateway implementations, MLOps for agentic systems, and CI/CD pipeline automation.

### 4.2. Deployment & Scaling

Chapter 4.2 details deployment patterns for agentic systems, examining microservices and serverless approaches, message queue architecture selection, vector database deployment options, observability implementation, and CI/CD pipeline construction. The chapter provides production-ready guidance for scaling systems while maintaining reliability through progressive deployment and comprehensive monitoring.

<details markdown="block">
<summary>Code examples (4 files)</summary>

- [`01_quantization_optimization.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2_01_quantization_optimization.py)
- [`02_model_pruning.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2_02_model_pruning.py)
- [`03_tensorrt_object_detector.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2_03_tensorrt_object_detector.py)
- [`04_jetson_deployment_runtime.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2_04_jetson_deployment_runtime.py)

</details>

### 4.3. Container Orchestration and Edge Deployment

Chapter 4.3 covers Kubernetes orchestration for production multi-agent deployments and edge model optimization strategies. The chapter explains how Kubernetes automates deployment, scaling, and healing of containerized agents while covering model optimization techniques (quantization, pruning, distillation) that enable efficient edge deployment on resource-constrained devices.

<details markdown="block">
<summary>Code examples (7 files)</summary>

- [`01_autoscaling_metrics_monitoring.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_01_autoscaling_metrics_monitoring.py)
- [`02_weighted_load_balancer.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_02_weighted_load_balancer.py)
- [`03_rag_caching_configuration.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_03_rag_caching_configuration.py)
- [`jetson_deployment_runtime_code_04_jetson_deployment_runtime.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_jetson_deployment_runtime_code_04_jetson_deployment_runtime.py)
- [`model_pruning_code_02_model_pruning.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_model_pruning_code_02_model_pruning.py)
- [`quantization_optimization_code_01_quantization_optimization.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_quantization_optimization_code_01_quantization_optimization.py)
- [`tensorrt_object_detector_code_03_tensorrt_object_detector.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.3_tensorrt_object_detector_code_03_tensorrt_object_detector.py)

</details>

### 4.4. Performance Profiling and Optimization

Performance profiling represents a critical but often-neglected step between deployment and production stability. AI agent systems introduce unique challenges compared to traditional inference workloads because their multi-stage execution pattern creates bottlenecks distributed across components that simple metrics cannot reveal. Measurement-driven optimization transforms deployment from one-time event into continuous cycle of improvement, replacing assumptions with data to guide effort toward high-impact optimizations.

<details markdown="block">
<summary>Code examples (12 files)</summary>

- [`08_react_agent_inference.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_08_react_agent_inference.py)
- [`11_quantization_baseline_fp16.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_11_quantization_baseline_fp16.py)
- [`12_int8_quantization_evaluation.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_12_int8_quantization_evaluation.py)
- [`13_accuracy_validation_quantization.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_13_accuracy_validation_quantization.py)
- [`14_attention_profiling_baseline.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_14_attention_profiling_baseline.py)
- [`15_flash_attention_optimization.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_15_flash_attention_optimization.py)
- [`16_paged_attention_batch_scaling.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_16_paged_attention_batch_scaling.py)
- [`17_optimized_throughput_measurement.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_17_optimized_throughput_measurement.py)
- [`18_predictability_analysis_speculative_decoding.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_18_predictability_analysis_speculative_decoding.py)
- [`19_speculative_decoding_deployment.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_19_speculative_decoding_deployment.py)
- [`20_speculative_output_quality_validation.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_20_speculative_output_quality_validation.py)
- [`21_mlflow_registration_artifacts.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.4_21_mlflow_registration_artifacts.py)

</details>

### 4.5. NVIDIA NIM and Triton Inference Server

NVIDIA NIM represents a paradigm shift in LLM deployment by collapsing the months-long gap between "agent works locally" and "agent serves production traffic" through pre-optimized containerized microservices. NIM bundles a complete, enterprise-grade inference stack while Triton serves as a unified multi-framework serving platform, enabling production-quality deployments without extensive optimization expertise.

### 4.6. TensorRT-LLM and NVIDIA Fleet Command

TensorRT-LLM addresses fundamental inference challenges through optimization pipeline orchestrating multiple complementary optimizations achieving 3-8x speedup while reducing memory 50-75%. Fleet Command enables orchestration of edge AI deployments at scale through hybrid-cloud architecture, one-touch provisioning, and zero-trust security, transforming edge deployment from operational burden to managed platform.

### 4.7. Scaling Strategies

Horizontal scaling addresses capacity expansion through creating multiple agent instances operating in parallel, enabling nearly linear capacity improvements. Strategic scaling requires effective load balancing, sophisticated batching decisions, multi-tier caching architectures, and cost optimization while maintaining high availability across distributed infrastructure.

### Other code examples

These files are named for a chapter number that is not in the current chapter list (an older numbering), so they are not attached to a chapter above.

<details markdown="block">
<summary>22 files</summary>

- [`Part_04_Chapter_4.1C_01_cuda_apt_installation.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_01_cuda_apt_installation.sh)
- [`Part_04_Chapter_4.1C_02_nsys_version_verification.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_02_nsys_version_verification.sh)
- [`Part_04_Chapter_4.1C_03_nsight_docker_pull.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_03_nsight_docker_pull.sh)
- [`Part_04_Chapter_4.1C_04_nsight_workspace_setup.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_04_nsight_workspace_setup.sh)
- [`Part_04_Chapter_4.1C_05_docker_run_nsight_container.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_05_docker_run_nsight_container.sh)
- [`Part_04_Chapter_4.1C_06_nvidia_smi_gpu_verification.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_06_nvidia_smi_gpu_verification.sh)
- [`Part_04_Chapter_4.1C_07_nsys_profile_command.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_07_nsys_profile_command.sh)
- [`Part_04_Chapter_4.1C_09_react_profiling_execution.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_09_react_profiling_execution.sh)
- [`Part_04_Chapter_4.1C_10_nsys_gui_launch.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_10_nsys_gui_launch.sh)
- [`Part_04_Chapter_4.1C_22_mlflow_production_transition.sh`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_22_mlflow_production_transition.sh)
- [`Part_04_Chapter_4.1C_23_kustomization_staging_overlay.yaml`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_23_kustomization_staging_overlay.yaml)
- [`Part_04_Chapter_4.1C_24_kustomization_production_overlay.yaml`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_24_kustomization_production_overlay.yaml)
- [`Part_04_Chapter_4.1C_25_argo_rollout_canary_deployment.yaml`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_25_argo_rollout_canary_deployment.yaml)
- [`Part_04_Chapter_4.1C_26_analysis_template_success_rate_check.yaml`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.1C_26_analysis_template_success_rate_check.yaml)
- [`Part_04_Chapter_4.2A_01_langchain_nim_configuration.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2A_01_langchain_nim_configuration.py)
- [`Part_04_Chapter_4.2A_02_bert_torchscript_conversion.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2A_02_bert_torchscript_conversion.py)
- [`Part_04_Chapter_4.2A_03_triton_sentiment_realtime_client.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2A_03_triton_sentiment_realtime_client.py)
- [`Part_04_Chapter_4.2A_04_triton_sentiment_batch_processing.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2A_04_triton_sentiment_batch_processing.py)
- [`Part_04_Chapter_4.2B_01_gpt2_baseline_performance_measurement.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2B_01_gpt2_baseline_performance_measurement.py)
- [`Part_04_Chapter_4.2B_02_tensorrt_llm_fp16_optimization.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2B_02_tensorrt_llm_fp16_optimization.py)
- [`Part_04_Chapter_4.2B_03_int8_quantization_calibration.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2B_03_int8_quantization_calibration.py)
- [`Part_04_Chapter_4.2B_04_production_kv_cache_configuration.py`]({{ site.repo_blob }}/code_examples/Part_04_Chapter_4.2B_04_production_kv_cache_configuration.py)

</details>

### Labs

No finished lab exists for this Part yet. These legacy example files are prose excerpts with embedded code, kept as source material; they do not count as lab coverage. See [Labs]({% link labs.md %}).

- [`Part_04_Chapter_4.1_Labs1.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs1.md)
- [`Part_04_Chapter_4.1_Labs2.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs2.md)
- [`Part_04_Chapter_4.6.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.6.md)
