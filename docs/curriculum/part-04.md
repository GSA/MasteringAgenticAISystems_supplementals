---
title: "Part 4 — Production Deployment & Scaling"
parent: Curriculum
nav_order: 4
permalink: /curriculum/part-04/
---

# Part 4 — Production Deployment & Scaling
{: .no_toc }

7 chapters · 24.4 study hours allocated in the Study Plan · 7 slide decks · 39 videos · 45 code example files

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
| 4.1 | [AI Agent Deployment and Scaling]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-41-ai-agent-deployment-and-scaling) | 3.2 | [PDF]({{ site.repo_blob }}/slides/Ch4.1_v20MAR26.pdf) | [Quiz 4.1A†](https://docs.google.com/forms/d/1if2c8Qu-gzjVf7NkPtEV6hPmgCpQcGauEoaFp-zCoGs/viewform?usp=sharing), [Quiz 4.1B†](https://docs.google.com/forms/d/1AGZ2H6dIePFVzdv0nsS_wM0x6BjH6UhC8IdAh-Sp49A/viewform?usp=sharing) | [9]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-41---ai-agent-deployment-and-scaling) | [6]({{ site.repo_tree }}/figures/Ch4.1_figures_v11MAY26) | — | AWS GCP MS |
| 4.2 | [Deployment & Scaling]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-42-deployment--scaling) | 5.8 | [PDF]({{ site.repo_blob }}/slides/Ch4.2_v20MAR26.pdf) | — | [1]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-42---deployment--scaling-architecture) | [12]({{ site.repo_tree }}/figures/Ch4.2_figures_v11MAY26) | 4 | NV AWS GCP |
| 4.3 | [Container Orchestration and Edge Deployment]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-43-container-orchestration-and-edge-deployment) | 2.1 | [PDF]({{ site.repo_blob }}/slides/Ch4.3_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1zf3fb2qlxLbs6Z6ARpbrcUdp-T5OE6hmEP9wMZC3bjw/viewform?usp=sharing) | [7]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-43---container-orchestration-and-edge-deployment) | [12]({{ site.repo_tree }}/figures/Ch4.3_figures_v11MAY26) | 7 | NV AWS GCP MS |
| 4.4 | [Performance Profiling and Optimization]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-44-performance-profiling-and-optimization) | 6.0 | [PDF]({{ site.repo_blob }}/slides/Ch4.4_v20MAR26.pdf) | — | [7]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-44---performance-profiling-and-optimization) | [12]({{ site.repo_tree }}/figures/Ch4.4_figures_v11MAY26) | 12 | NV AWS GCP MS |
| 4.5 | [NVIDIA NIM and Triton Inference Server]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-45-nvidia-nim-and-triton-inference-server) | 2.1 | [PDF]({{ site.repo_blob }}/slides/Ch4.5_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1Z1xLfMAdzT0IJibUl0cGqfRD7Hh1Pb2CY2SCUcGJL6Q/viewform?usp=sharing) | [5]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-45---nvidia-nim-and-triton-inference-server) | [6]({{ site.repo_tree }}/figures/Ch4.5_figures_v11MAY26) | — | NV AWS GCP MS |
| 4.6 | [TensorRT-LLM and NVIDIA Fleet Command]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-46-tensorrt-llm-and-nvidia-fleet-command) | 1.5 | [PDF]({{ site.repo_blob }}/slides/Ch4.6_v20MAR26.pdf) | [Quiz](https://docs.google.com/forms/d/1ak-7FD_KYAcAfbrWxUm_y2n3pyBXM0vvrnu5ZjYEkVE/viewform?usp=sharing) | [7]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-46---tensorrt-llm-and-nvidia-fleet-command) | [9]({{ site.repo_tree }}/figures/Ch4.6_figures_v11MAY26) | — | NV AWS GCP MS |
| 4.7 | [Scaling Strategies]({{ site.repo_blob }}/Study_Plan.md#part-4-chapter-47-scaling-strategies) | 3.7 | [PDF]({{ site.repo_blob }}/slides/Ch4.7_v20MAR26.pdf) | — | [3]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md#chapter-47---scaling-strategies) | [6]({{ site.repo_tree }}/figures/Ch4.7_figures_v11MAY26) | — | NV AWS GCP MS |


**Notes.** The Videos column counts the videos shown under each chapter summary below, out of the unique direct links in [`Part_04_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md) ("3 of 5"). A video is left out when its link is dead, embedding is disabled, or YouTube's title does not match the entry; see the [link check]({{ site.baseurl }}/videos/link-check/). Chapters can also list search suggestions instead of links.

† Linked by chapter-family number, not an exact ID match: the deck, quiz, or figure set is numbered differently from this chapter in the source files (for example a quiz or deck numbered `6.2` for chapters `6.2A` and `6.2B`).

‡ A combined deck that covers more than one chapter.

A chapter that is missing from a certification's mapping file shows no tag for that certification: the NVIDIA file omits 4.1 and 10.6, and the other four omit 1.8, 9.16, and 9.17.


## Chapter summaries


Summaries are excerpted from [`Study_Plan.md`]({{ site.repo_blob }}/Study_Plan.md), which also lists each chapter's key concepts and self-check questions.


### 4.1. AI Agent Deployment and Scaling

This chapter introduces the essential infrastructure and operational practices for deploying and scaling multi-agent systems in production, covering message queue architectures, vector database selection, observability patterns, API gateway implementations, MLOps for agentic systems, and CI/CD pipeline automation.

<details markdown="block">
<summary>Videos (9)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/20rOdqag4Dw" title="Kong Gateway Tutorial | API Gateway For Beginners" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=20rOdqag4Dw">Kong Gateway Tutorial | API Gateway For Beginners</a> &middot; Redhwan Nacef</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/qJYWpRuoBx8" title="Kong Gateway Microservice Architecture" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=qJYWpRuoBx8">Kong Gateway Microservice Architecture</a> &middot; Kong</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/xjBavX0JFGk" title="Kong Your Way into the New Year: An Introduction to API Gateway" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=xjBavX0JFGk">Kong Your Way into the New Year: An Introduction to API Gateway</a> &middot; Kong</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Ox26-mRSsvg" title="RAG Explained (Retrieval Augmented Generation) with n8n Chatbot Demo" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Ox26-mRSsvg">RAG Explained (Retrieval Augmented Generation) with n8n Chatbot Demo</a> &middot; The AI Explorer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/e_13aTsxNoY" title="Du Blue-Green au Canary-Release avec Kubernetes (Mathieu Herbert)" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=e_13aTsxNoY">Du Blue-Green au Canary-Release avec Kubernetes (Mathieu Herbert)</a> &middot; BreizhCamp</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/aZzV6X7XhyI" title="Automate your Docker Build/Test/Deploy pipelines!" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=aZzV6X7XhyI">Automate your Docker Build/Test/Deploy pipelines!</a> &middot; Bret Fisher</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/X48VuDVv0do" title="Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=X48VuDVv0do">Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]</a> &middot; TechWorld with Nana</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/s_o8dwzRlu4" title="Kubernetes Crash Course for Absolute Beginners [NEW]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=s_o8dwzRlu4">Kubernetes Crash Course for Absolute Beginners [NEW]</a> &middot; TechWorld with Nana</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/kTp5xUtcalw" title="Docker Containers and Kubernetes Fundamentals – Full Hands-On Course" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=kTp5xUtcalw">Docker Containers and Kubernetes Fundamentals – Full Hands-On Course</a> &middot; freeCodeCamp.org</div>
</div>

</details>

### 4.2. Deployment & Scaling

Chapter 4.2 details deployment patterns for agentic systems, examining microservices and serverless approaches, message queue architecture selection, vector database deployment options, observability implementation, and CI/CD pipeline construction. The chapter provides production-ready guidance for scaling systems while maintaining reliability through progressive deployment and comprehensive monitoring.

<details markdown="block">
<summary>Videos (1)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/QoDqxm7ybLc" title="Setup Prometheus Monitoring on Kubernetes using Helm and Prometheus Operator | Part 1" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=QoDqxm7ybLc">Setup Prometheus Monitoring on Kubernetes using Helm and Prometheus Operator | Part 1</a> &middot; TechWorld with Nana</div>
</div>

</details>

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
<summary>Videos (7)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/X48VuDVv0do" title="Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=X48VuDVv0do">Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]</a> &middot; TechWorld with Nana</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Wf2eSG3owoA" title="Docker and Kubernetes - Full Course for Beginners" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Wf2eSG3owoA">Docker and Kubernetes - Full Course for Beginners</a> &middot; freeCodeCamp.org</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/s_o8dwzRlu4" title="Kubernetes Crash Course for Absolute Beginners [NEW]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=s_o8dwzRlu4">Kubernetes Crash Course for Absolute Beginners [NEW]</a> &middot; TechWorld with Nana</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/km0yT99eVTY" title="$99 Jetson Nano - Intro, Setup and Demo" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=km0yT99eVTY">$99 Jetson Nano - Intro, Setup and Demo</a> &middot; JetsonHacks</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/6zDrLvpfCK4" title="Istio Service Mesh Explained" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=6zDrLvpfCK4">Istio Service Mesh Explained</a> &middot; IBM Technology</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/jM36M39MA3I" title="Kubernetes cluster autoscaling for beginners" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=jM36M39MA3I">Kubernetes cluster autoscaling for beginners</a> &middot; That DevOps Guy</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Vrxr-7rjkvM" title="Kubernetes Tutorial: Why Do You Need StatefulSets in Kubernetes?" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Vrxr-7rjkvM">Kubernetes Tutorial: Why Do You Need StatefulSets in Kubernetes?</a> &middot; KodeKloud</div>
</div>

</details>

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
<summary>Videos (7)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/5Gxx59Q0g6o" title="Performance Tuning the NVIDIA Grace CPU with NVIDIA Nsight Tools" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=5Gxx59Q0g6o">Performance Tuning the NVIDIA Grace CPU with NVIDIA Nsight Tools</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/5ftMMBj6xj0" title="DeepSeek R1 performance optimization to push the throughput performance boundary" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=5ftMMBj6xj0">DeepSeek R1 performance optimization to push the throughput performance boundary</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/J4-qZ6KBalk" title="FlashAttention-2: Making Transformers 800% faster AND exact" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=J4-qZ6KBalk">FlashAttention-2: Making Transformers 800% faster AND exact</a> &middot; Latent Space</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/pxk1Fr33-L4" title="End To End MLOPS Data Science Project Implementation With Deployment" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=pxk1Fr33-L4">End To End MLOPS Data Science Project Implementation With Deployment</a> &middot; Krish Naik</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/0XEWn4VmiDE" title="Load Testing Argo CD at Scale with vCluster and GitOps" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=0XEWn4VmiDE">Load Testing Argo CD at Scale with vCluster and GitOps</a> &middot; vCluster</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/MeU5_k9ssrs" title="ArgoCD Tutorial for Beginners | GitOps CD for Kubernetes" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=MeU5_k9ssrs">ArgoCD Tutorial for Beginners | GitOps CD for Kubernetes</a> &middot; TechWorld with Nana</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/NQDtfSi5QF4" title="Getting Started with NVIDIA Triton Inference Server" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=NQDtfSi5QF4">Getting Started with NVIDIA Triton Inference Server</a> &middot; NVIDIA Developer</div>
</div>

</details>

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

<details markdown="block">
<summary>Videos (5)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/NQDtfSi5QF4" title="Getting Started with NVIDIA Triton Inference Server" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=NQDtfSi5QF4">Getting Started with NVIDIA Triton Inference Server</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/5ftMMBj6xj0" title="DeepSeek R1 performance optimization to push the throughput performance boundary" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=5ftMMBj6xj0">DeepSeek R1 performance optimization to push the throughput performance boundary</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/NaT5Eo97_I0" title="Building Multimodal AI RAG with LlamaIndex, NVIDIA NIM, and Milvus | LLM App Development" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=NaT5Eo97_I0">Building Multimodal AI RAG with LlamaIndex, NVIDIA NIM, and Milvus | LLM App Development</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/lw0c1Ah-c-E" title="運用 NVIDIA Clara 與 Kubernetes 平行醫療 AI 模型訓練與專案管理" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=lw0c1Ah-c-E">運用 NVIDIA Clara 與 Kubernetes 平行醫療 AI 模型訓練與專案管理</a> &middot; NVIDIA Taiwan</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/wtJ09xDuSx0" title="Kubernetes HPA Not Scaling? Here&#x27;s Why – With Live Demo" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=wtJ09xDuSx0">Kubernetes HPA Not Scaling? Here&#x27;s Why – With Live Demo</a> &middot; sarfatech</div>
</div>

</details>

### 4.6. TensorRT-LLM and NVIDIA Fleet Command

TensorRT-LLM addresses fundamental inference challenges through optimization pipeline orchestrating multiple complementary optimizations achieving 3-8x speedup while reducing memory 50-75%. Fleet Command enables orchestration of edge AI deployments at scale through hybrid-cloud architecture, one-touch provisioning, and zero-trust security, transforming edge deployment from operational burden to managed platform.

<details markdown="block">
<summary>Videos (7)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/5ftMMBj6xj0" title="DeepSeek R1 performance optimization to push the throughput performance boundary" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=5ftMMBj6xj0">DeepSeek R1 performance optimization to push the throughput performance boundary</a> &middot; NVIDIA Developer</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/7xTGNNLPyMI" title="Deep Dive into LLMs like ChatGPT" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=7xTGNNLPyMI">Deep Dive into LLMs like ChatGPT</a> &middot; Andrej Karpathy</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/zjkBMFhNj_g" title="[1hr Talk] Intro to Large Language Models" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a> &middot; Andrej Karpathy</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/Oq2SN7uutbQ" title="E07 | Fast LLM Serving with vLLM and PagedAttention" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=Oq2SN7uutbQ">E07 | Fast LLM Serving with vLLM and PagedAttention</a> &middot; MLSys Singapore</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/LuhJEEJQgUM" title="Lecture 1 How to profile CUDA kernels in PyTorch" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=LuhJEEJQgUM">Lecture 1 How to profile CUDA kernels in PyTorch</a> &middot; GPU MODE</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/kCc8FmEb1nY" title="Let&#x27;s build GPT: from scratch, in code, spelled out." loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let&#x27;s build GPT: from scratch, in code, spelled out.</a> &middot; Andrej Karpathy</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/aircAruvnKk" title="But what is a neural network? | Deep learning chapter 1" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=aircAruvnKk">But what is a neural network? | Deep learning chapter 1</a> &middot; 3Blue1Brown</div>
</div>

</details>

### 4.7. Scaling Strategies

Horizontal scaling addresses capacity expansion through creating multiple agent instances operating in parallel, enabling nearly linear capacity improvements. Strategic scaling requires effective load balancing, sophisticated batching decisions, multi-tier caching architectures, and cost optimization while maintaining high availability across distributed infrastructure.

<details markdown="block">
<summary>Videos (3)</summary>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/wtJ09xDuSx0" title="Kubernetes HPA Not Scaling? Here&#x27;s Why – With Live Demo" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=wtJ09xDuSx0">Kubernetes HPA Not Scaling? Here&#x27;s Why – With Live Demo</a> &middot; sarfatech</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/3OP-q55hOUI" title="Cloud Run QuickStart - Docker to Serverless" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=3OP-q55hOUI">Cloud Run QuickStart - Docker to Serverless</a> &middot; Fireship</div>
</div>

<div style="margin:0 0 1.5rem 0">
<iframe src="https://www.youtube-nocookie.com/embed/X48VuDVv0do" title="Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]" loading="lazy" style="width:100%;max-width:560px;aspect-ratio:16/9;border:0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
<div><a href="https://www.youtube.com/watch?v=X48VuDVv0do">Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]</a> &middot; TechWorld with Nana</div>
</div>

</details>

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

No finished lab exists for this Part yet. These legacy example files are prose excerpts with embedded code, kept as source material; they do not count as lab coverage. See [Labs]({{ site.baseurl }}/labs/).

- [`Part_04_Chapter_4.1_Labs1.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs1.md)
- [`Part_04_Chapter_4.1_Labs2.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.1_Labs2.md)
- [`Part_04_Chapter_4.6.md`]({{ site.repo_blob }}/labs/archive/Part_04_Chapter_4.6.md)
