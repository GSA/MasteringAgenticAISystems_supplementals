# Part 04: Production Deployment and Scaling - Video and Learning Resources

> **Catalog note:** A hyperlink means a resource is currently assigned; it does not by itself guarantee that every title, runtime, or scope claim has been independently verified. Entries without a hyperlink are unverified discovery candidates. See [README.md](README.md) and [video_resource_status.csv](video_resource_status.csv) for status and provenance.

## Table of Contents

- [Chapter 4.1 - AI Agent Deployment and Scaling](#chapter-41---ai-agent-deployment-and-scaling)
- [Chapter 4.2 - Deployment & Scaling Architecture](#chapter-42---deployment--scaling-architecture)
- [Chapter 4.3 - Container Orchestration and Edge Deployment](#chapter-43---container-orchestration-and-edge-deployment)
- [Chapter 4.4 - Performance Profiling and Optimization](#chapter-44---performance-profiling-and-optimization)
- [Chapter 4.5 - NVIDIA NIM and Triton Inference Server](#chapter-45---nvidia-nim-and-triton-inference-server)
- [Chapter 4.6 - TensorRT-LLM and NVIDIA Fleet Command](#chapter-46---tensorrt-llm-and-nvidia-fleet-command)
- [Chapter 4.7 - Scaling Strategies](#chapter-47---scaling-strategies)

---

<a name="chapter-41---ai-agent-deployment-and-scaling"></a>
## Chapter 4.1 - AI Agent Deployment and Scaling

**Topics:** Message Queues, Vector Databases, Observability, API Gateways, MLOps, CI/CD Pipelines

### Kong Gateway Tutorial | API Gateway For Beginners
- [https://www.youtube.com/watch?v=20rOdqag4Dw](https://www.youtube.com/watch?v=20rOdqag4Dw) ~10 minutes
- Covers: Kong Gateway setup and configuration, services and routes architecture, plugin architecture for extensibility, API management basics

### Kong Gateway Microservice Architecture
- [https://www.youtube.com/watch?v=qJYWpRuoBx8](https://www.youtube.com/watch?v=qJYWpRuoBx8) ~1 hour 2 minutes
- Covers: Advanced Kong routing patterns, plugin pipeline architecture, authentication and rate limiting, traffic control strategies

### Kong Your Way into the New Year: An Introduction to API Gateway
- [https://www.youtube.com/watch?v=xjBavX0JFGk](https://www.youtube.com/watch?v=xjBavX0JFGk) ~1 hour 7 minutes
- Covers: API gateway fundamentals, centralized authentication, request routing and traffic management, cross-cutting concerns

### RAG Explained – n8n Chatbot Demo with Sources
- [https://www.youtube.com/watch?v=Ox26-mRSsvg](https://www.youtube.com/watch?v=Ox26-mRSsvg) Variable
- Covers: Retrieval-Augmented Generation (RAG), vector databases for semantic search, chatbot implementation with source citations

### Du Blue-Green au Canary Release avec Kubernetes - Mathieu Herbert
- [https://www.youtube.com/watch?v=e_13aTsxNoY](https://www.youtube.com/watch?v=e_13aTsxNoY) (conference talk; runtime not independently verified)
- Covers: Blue-green deployment strategies, canary deployment patterns, progressive rollout with Kubernetes

### GitHub Actions Docker Build Summary: Unlock Insights and Fixes
- [https://www.docker.com/resources/github-actions-docker-build-summary-unlock-insights-and-fixes-webinar/](https://www.docker.com/resources/github-actions-docker-build-summary-unlock-insights-and-fixes-webinar/) (official on-demand webinar)
- Covers: Using Docker build summaries and Docker Desktop insights to diagnose and improve GitHub Actions build workflows

### Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]
- [https://www.youtube.com/watch?v=X48VuDVv0do](https://www.youtube.com/watch?v=X48VuDVv0do) ~4 hours
- Covers: Kubernetes fundamentals, pods, deployments, services, configuration and orchestration, production deployment patterns

### Kubernetes Crash Course for Absolute Beginners [NEW]
- [https://www.youtube.com/watch?v=s_o8dwzRlu4](https://www.youtube.com/watch?v=s_o8dwzRlu4) ~1 hour
- Covers: Kubernetes components and architecture, deployments, services, and practical Minikube demonstrations

### Kubernetes Fundamentals
- [https://www.youtube.com/watch?v=kTp5xUtcalw](https://www.youtube.com/watch?v=kTp5xUtcalw) Variable
- Covers: Docker and Kubernetes fundamentals, containerization, orchestration workflows

### RabbitMQ Crash Course - Hussein Nasser
- [https://www.youtube.com/watch?v=Cie5v59mrTg](https://www.youtube.com/watch?v=Cie5v59mrTg) ~43 minutes
- Covers: AMQP, connections and channels, exchanges and queues, publishers and consumers, acknowledgments, Docker, and Node.js

### Continuous Delivery for Machine Learning - Danilo Sato and Arif Wider
- [https://www.youtube.com/watch?v=um6Sq5EhW6A](https://www.youtube.com/watch?v=um6Sq5EhW6A) ~1 hour
- Covers: Continuous delivery for ML, reproducible pipelines, model and data versioning, testing, deployment, and monitoring

---

<a name="chapter-42---deployment--scaling-architecture"></a>
## Chapter 4.2 - Deployment & Scaling Architecture

**Topics:** Microservices, Serverless, Message Queues, Observability, API Gateways, MLOps

### Microservice Architecture and System Design with Python and Kubernetes - Full Course
- [https://www.youtube.com/watch?v=hmkF77F9TLw](https://www.youtube.com/watch?v=hmkF77F9TLw) ~5 hours
- Covers: Python microservices, Kubernetes, RabbitMQ, MongoDB, MySQL, service communication, and distributed system design

### Your Journey to a Serverless World: An Introduction to Serverless - Red Hat
- [https://www.youtube.com/watch?v=03LeehvuOJE](https://www.youtube.com/watch?v=03LeehvuOJE)
- Covers: Serverless architecture, developer implications, Apache OpenWhisk, Kubernetes, and deployment of serverless functions

### Setup Prometheus Monitoring on Kubernetes using Helm and Prometheus Operator
- [https://www.youtube.com/watch?v=QoDqxm7ybLc](https://www.youtube.com/watch?v=QoDqxm7ybLc) ~30 minutes
- Covers: Prometheus architecture, Kubernetes operators, Helm charts, Grafana integration, metrics collection and exporters

### Microservices Integration: Kafka vs RabbitMQ - Gabriele Santomaggio
- [https://vimeo.com/351826121](https://vimeo.com/351826121) (conference video)
- Covers: Kafka and RabbitMQ architecture, delivery models, integration trade-offs, and workload selection

### RabbitMQ Crash Course - Hussein Nasser
- [https://www.youtube.com/watch?v=Cie5v59mrTg](https://www.youtube.com/watch?v=Cie5v59mrTg) ~43 minutes
- Covers: AMQP, connections and channels, exchanges and queues, routing, acknowledgments, Docker, and Node.js

### MLOps Course: Build Production-Grade Machine Learning Projects
- [https://www.youtube.com/watch?v=-dJPoLm_gtE](https://www.youtube.com/watch?v=-dJPoLm_gtE) ~3 hours
- Covers: MLOps foundations, ZenML, MLflow, production pipelines, deployment, testing, monitoring, and versioning

### What Is an API Gateway? - IBM Technology
- [https://www.youtube.com/watch?v=hWRRdICvMNs](https://www.youtube.com/watch?v=hWRRdICvMNs) ~8 minutes
- Covers: API gateways in microservice systems, routing, security, rate limiting, monitoring, protocol handling, and backend-for-frontend patterns

### Du Blue-Green au Canary Release avec Kubernetes - Mathieu Herbert
- [https://www.youtube.com/watch?v=e_13aTsxNoY](https://www.youtube.com/watch?v=e_13aTsxNoY)
- Covers: Rolling updates, blue-green deployment, canary releases, progressive delivery, risk reduction, and rollback concepts

---

<a name="chapter-43---container-orchestration-and-edge-deployment"></a>
## Chapter 4.3 - Container Orchestration and Edge Deployment

**Topics:** Kubernetes, Model Quantization, Edge Deployment, HPA, Service Mesh

### Kubernetes Tutorial for Beginners [FULL COURSE in 4 Hours]
- [https://www.youtube.com/watch?v=X48VuDVv0do](https://www.youtube.com/watch?v=X48VuDVv0do) ~4 hours
- Covers: Kubernetes architecture, pods, services, deployments, namespaces and Ingress, Helm and StatefulSets, persistent volumes

### Docker and Kubernetes - Full Course for Beginners
- [https://www.youtube.com/watch?v=Wf2eSG3owoA](https://www.youtube.com/watch?v=Wf2eSG3owoA) ~6 hours
- Covers: Docker fundamentals, Kubernetes architecture, config maps and deployments, container orchestration

### Kubernetes Crash Course for Absolute Beginners [NEW]
- [https://www.youtube.com/watch?v=s_o8dwzRlu4](https://www.youtube.com/watch?v=s_o8dwzRlu4) ~1 hour
- Covers: Core Kubernetes concepts, components and architecture, practical Minikube demos

### Quantization: Optimize AI Models to Run Everywhere
- [https://www.youtube.com/watch?v=0VdNflU08yA](https://www.youtube.com/watch?v=0VdNflU08yA) ~10 minutes
- Covers: Post-training quantization (PTQ), quantization-aware training (QAT), model compression techniques

### MIT 6.S965 Lecture 04: Pruning and Sparsity, Part II
- [https://www.youtube.com/watch?v=sDJymyfAOKY](https://www.youtube.com/watch?v=sDJymyfAOKY) ~1 hour 8 minutes
- Covers: Layer-wise pruning ratios, sparse-network fine-tuning, lottery tickets, automatic pruning, and system support for sparsity

### Getting Started with the NVIDIA Jetson Nano
- [https://www.youtube.com/watch?v=km0yT99eVTY](https://www.youtube.com/watch?v=km0yT99eVTY) ~25 minutes
- Covers: Jetson Nano setup, JetPack SDK, CUDA parallel processing, edge AI hardware

### Istio Service Mesh Explained
- [https://www.youtube.com/watch?v=6zDrLvpfCK4](https://www.youtube.com/watch?v=6zDrLvpfCK4) ~8 minutes
- Covers: Service mesh architecture, Istio components, traffic management, sidecar proxies

### Scaling Explained Through Kubernetes HPA, VPA, KEDA & Cluster Autoscaler
- [https://www.youtube.com/watch?v=HQY2jgSN6pA](https://www.youtube.com/watch?v=HQY2jgSN6pA) ~25 minutes
- Covers: Pod and node scaling with HPA, VPA, KEDA, and Cluster Autoscaler, including guidance on when to use each approach

### StatefulSets vs Deployments - Kubernetes
- [https://www.youtube.com/watch?v=Vrxr-7rjkvM](https://www.youtube.com/watch?v=Vrxr-7rjkvM) ~15 minutes
- Covers: StatefulSets for stateful applications, persistent identity, ordered initialization, persistent volume claims

---

<a name="chapter-44---performance-profiling-and-optimization"></a>
## Chapter 4.4 - Performance Profiling and Optimization

**Topics:** NVIDIA Nsight, GPU Profiling, TensorRT-LLM, Flash Attention, MLflow, GitOps

### Performance Tuning the NVIDIA Grace CPU with NVIDIA Nsight Tools
- [https://www.youtube.com/watch?v=5Gxx59Q0g6o](https://www.youtube.com/watch?v=5Gxx59Q0g6o) ~60 minutes
- Covers: Nsight Systems profiling workflow, system-wide performance analysis, CPU-GPU timeline visualization, bottleneck identification

### TensorRT-LLM Livestream: DeepSeek R1 Performance Optimization
- [https://www.youtube.com/watch?v=5ftMMBj6xj0](https://www.youtube.com/watch?v=5ftMMBj6xj0) ~90 minutes
- Covers: TensorRT-LLM performance optimization, throughput boundary optimization, quantization strategies, practical workflows

### FlashAttention with Author Tri Dao - Interview
- [https://www.youtube.com/watch?v=J4-qZ6KBalk](https://www.youtube.com/watch?v=J4-qZ6KBalk)
- Covers: The motivation, algorithmic ideas, IO-aware attention, implementation trade-offs, and development of FlashAttention

### MLflow End to End Tutorial with Deployment
- [https://www.youtube.com/watch?v=pxk1Fr33-L4](https://www.youtube.com/watch?v=pxk1Fr33-L4) ~90 minutes
- Covers: MLflow model registry, artifact tracking and versioning, model deployment workflows, experiment tracking

### Load Testing Argo CD at Scale with vCluster and GitOps
- [https://www.vcluster.com/events/load-testing-argo-cd-at-scale-with-vcluster-and-gitops](https://www.vcluster.com/events/load-testing-argo-cd-at-scale-with-vcluster-and-gitops) (official event video)
- Covers: Load testing Argo CD with GitOps and up to 1,000 virtual clusters using vCluster

### ArgoCD Tutorial for Beginners - GitOps CD for Kubernetes
- [https://www.youtube.com/watch?v=MeU5_k9ssrs](https://www.youtube.com/watch?v=MeU5_k9ssrs) ~60 minutes
- Covers: ArgoCD fundamentals, GitOps continuous delivery, application sync and health, rollback procedures

### NVIDIA Triton Inference Server - Getting Started
- [https://www.youtube.com/watch?v=NQDtfSi5QF4](https://www.youtube.com/watch?v=NQDtfSi5QF4) ~45 minutes
- Covers: Triton fundamentals, model deployment and serving, configuration optimization, Triton Model Analyzer, batching strategies

---

<a name="chapter-45---nvidia-nim-and-triton-inference-server"></a>
## Chapter 4.5 - NVIDIA NIM and Triton Inference Server

**Topics:** NIM Microservices, Triton Server, Dynamic Batching, Tensor Parallelism, Kubernetes GPU

### Triton Inference Server API Endpoints Deep Dive
- [https://www.youtube.com/watch?v=NQDtfSi5QF4](https://www.youtube.com/watch?v=NQDtfSi5QF4) ~45 minutes
- Covers: Triton API endpoints, REST/gRPC protocols, model serving

### TensorRT-LLM Livestream - DeepSeek R1
- [https://www.youtube.com/watch?v=5ftMMBj6xj0](https://www.youtube.com/watch?v=5ftMMBj6xj0) ~90 minutes
- Covers: TensorRT-LLM optimization, kernel fusion, quantization

### Deploying vLLM with Hugging Face Inference Endpoints
- [https://huggingface.co/docs/inference-endpoints/engines/vllm](https://huggingface.co/docs/inference-endpoints/engines/vllm)
- Official Hugging Face documentation
- Covers: vLLM endpoint configuration, PagedAttention, continuous batching, KV-cache settings, and tensor/data parallel scaling

### NVIDIA NIM Multimodal RAG
- [https://www.youtube.com/watch?v=NaT5Eo97_I0](https://www.youtube.com/watch?v=NaT5Eo97_I0) Variable
- Covers: NIM deployment, multimodal RAG, inference microservices

### Clara with Kubernetes - GPU Multi-Node
- [https://www.youtube.com/watch?v=lw0c1Ah-c-E](https://www.youtube.com/watch?v=lw0c1Ah-c-E) Variable
- Covers: GPU Operator, Kubernetes GPU deployment, multi-node scheduling

### Kubernetes HPA Not Scaling? The Complete Troubleshooting Guide on HPA
- [https://www.youtube.com/watch?v=wtJ09xDuSx0](https://www.youtube.com/watch?v=wtJ09xDuSx0)
- Covers: Horizontal Pod Autoscaler troubleshooting, metrics-server, resource requests and limits, scaling policies, and a practical demo

---

<a name="chapter-46---tensorrt-llm-and-nvidia-fleet-command"></a>
## Chapter 4.6 - TensorRT-LLM and NVIDIA Fleet Command

**Topics:** TensorRT-LLM, INT8 Quantization, KV Cache, Tensor Parallelism, Fleet Command

### TensorRT-LLM Livestream: DeepSeek R1
- [https://www.youtube.com/watch?v=5ftMMBj6xj0](https://www.youtube.com/watch?v=5ftMMBj6xj0) ~60 minutes
- Covers: TensorRT-LLM optimization, high-throughput performance, Blackwell architecture

### Deep Dive into LLMs like ChatGPT
- [https://www.youtube.com/watch?v=7xTGNNLPyMI](https://www.youtube.com/watch?v=7xTGNNLPyMI) ~3h31m
- Covers: LLM pretraining, architecture, inference, complete pipeline

### Intro to Large Language Models
- [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g) ~1 hour
- Covers: LLM inference, mental models, optimization foundations

### Efficient Memory Management for LLM Serving with PagedAttention
- [https://www.youtube.com/watch?v=Oq2SN7uutbQ](https://www.youtube.com/watch?v=Oq2SN7uutbQ) ~45 minutes
- Covers: vLLM, PagedAttention, KV cache optimization, SOSP'23

### Fast LLM Serving with vLLM and PagedAttention
- [https://www.youtube.com/watch?v=5ZlavKF_98U](https://www.youtube.com/watch?v=5ZlavKF_98U) ~32 minutes
- Covers: vLLM architecture, PagedAttention, KV-cache memory management, continuous batching, GPU utilization, and serving throughput

### CUDA Mode Lecture 1 - Getting Started with CUDA Optimization
- [https://www.youtube.com/watch?v=LuhJEEJQgUM](https://www.youtube.com/watch?v=LuhJEEJQgUM) ~60 minutes
- Covers: CUDA fundamentals, kernel optimization, memory optimization

### Let's Build GPT: From Scratch, in Code, Spelled Out
- [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) ~2 hours
- Covers: GPT architecture, attention mechanisms, transformer implementation

### But what is a Neural Network?
- [https://www.youtube.com/watch?v=aircAruvnKk](https://www.youtube.com/watch?v=aircAruvnKk) ~20 minutes
- Covers: Neural network fundamentals, mathematical foundations

### PyTorch Distributed Data Parallel (DDP)
- [https://www.youtube.com/watch?v=TibQO_xv1zc](https://www.youtube.com/watch?v=TibQO_xv1zc) ~45 minutes
- Covers: Distributed training, multi-GPU coordination

---

<a name="chapter-47---scaling-strategies"></a>
## Chapter 4.7 - Scaling Strategies

**Topics:** Horizontal Scaling, Load Balancing, Batching, Caching, Cost Optimization

### Kubernetes HPA Not Scaling? The Complete Troubleshooting Guide on HPA
- [https://www.youtube.com/watch?v=wtJ09xDuSx0](https://www.youtube.com/watch?v=wtJ09xDuSx0) ~30 minutes
- Covers: Kubernetes HPA troubleshooting, metrics-server configuration, resource requests/limits, scaling policies

### Cloud Run QuickStart: Docker to Serverless
- [https://www.youtube.com/watch?v=3OP-q55hOUI](https://www.youtube.com/watch?v=3OP-q55hOUI) ~10 minutes
- Covers: Packaging a container and deploying it to Google Cloud Run as a managed serverless service

### Kubernetes Tutorial for Beginners (4 Hours)
- [https://www.youtube.com/watch?v=X48VuDVv0do](https://www.youtube.com/watch?v=X48VuDVv0do) ~4 hours
- Covers: Kubernetes architecture, deployments, services, scaling, load balancing with Services

### Complete Kubernetes Tutorial (Playlist)
- [https://www.youtube.com/watch?v=VnvRFRk_51k](https://www.youtube.com/watch?v=VnvRFRk_51k) Variable (22-video playlist)
- Covers: Kubernetes components, container orchestration, scaling applications
