# Google Professional Machine Learning Engineer – Knowledge Items

## 1. Architecting low‑code AI solutions (~13%)

- **1.1** Develop ML models using BigQuery ML (selecting appropriate model type: linear/binary classification, regression, time‑series, matrix factorization, boosted trees, autoencoders).
- **1.2** Select low‑code Google Cloud services (BigQuery ML, AutoML, Vertex AI Studio, Model Garden) to build ML and generative AI solutions.
- **1.3** Design low‑code solutions that integrate pre‑trained and foundation models (including generative models) into applications.
- **1.4** Evaluate when to use low‑code tools versus custom model development based on requirements, constraints, and team skills.

***

## 2. Collaborating within and across teams to manage data and models (~14%)

- **2.1** Explore and preprocess organization‑wide data stored in Cloud Storage, BigQuery, and other Google Cloud data services.
- **2.2** Select appropriate data representations and storage (structured, semi‑structured, unstructured) for ML workloads.
- **2.3** Apply data governance, access control, and privacy best practices when sharing data for ML across teams.
- **2.4** Collaborate with data engineers and stakeholders to define data requirements, quality checks, and SLAs for ML use cases.
- **2.5** Manage and share models, datasets, and artifacts through Vertex AI, repositories, and registries.
- **2.6** Evaluate generative AI solutions collaboratively, including prompt design reviews and safety considerations.

***

## 3. Scaling prototypes into ML models (~18%)

- **3.1** Build models by choosing ML frameworks (TensorFlow, PyTorch, scikit‑learn, XGBoost) and model architectures that meet requirements.
- **3.2** Apply modeling techniques that satisfy interpretability needs (e.g., linear models vs. complex deep models).
- **3.3** Implement feature engineering, regularization, and hyperparameter tuning to improve model performance.
- **3.4** Implement generative AI solutions using Model Garden and Vertex AI (including foundation and generative models).
- **3.5** Perform distributed training with TPUs and GPUs (e.g., Vertex AI Reduction Server, Horovod) for large‑scale models.
- **3.6** Evaluate and compare candidate models using appropriate metrics, validation schemes, and baselines.

***

## 4. Serving and scaling models (~20%)

- **4.1** Select appropriate serving options (online/batch) for ML and generative AI models on Google Cloud.
- **4.2** Deploy models to Vertex AI (endpoints, batch prediction, prediction routines) and configure autoscaling.
- **4.3** Integrate models into applications and services (REST APIs, gRPC, event‑driven architectures).
- **4.4** Optimize serving performance and latency using hardware accelerators, model optimization, and caching.
- **4.5** Implement A/B testing, canary releases, and traffic splitting for model rollouts.
- **4.6** Ensure security and compliance for model endpoints (authentication, authorization, network controls).

***

## 5. Automating and orchestrating ML pipelines (~22%)

- **5.1** Develop end‑to‑end ML pipelines with data ingestion, validation, training, evaluation, and deployment stages.
- **5.2** Ensure consistent data preprocessing and feature transformations between training and serving.
- **5.3** Host or integrate third‑party ML pipelines on Google Cloud (e.g., MLflow) where appropriate.
- **5.4** Identify pipeline components, parameters, triggers, and compute needs using services like Cloud Build and Cloud Run.
- **5.5** Select and configure orchestration frameworks (Kubeflow Pipelines, Vertex AI Pipelines, Cloud Composer, Jenkins).
- **5.6** Implement CI/CD practices for ML (training, evaluation, deployment, rollback).
- **5.7** Track and audit metadata for datasets, models, and experiments (Vertex AI Experiments, Vertex ML Metadata).
- **5.8** Establish model and data lineage for reproducibility and governance.

***

## 6. Monitoring AI solutions (~13%)

- **6.1** Monitor data quality and data drift for production ML systems.
- **6.2** Monitor model performance against baselines, simpler models, and across time, including accuracy and business KPIs.
- **6.3** Detect and troubleshoot common training and serving errors in ML and generative AI systems.
- **6.4** Apply model explainability tools on Vertex AI (e.g., feature attributions) to understand predictions.
- **6.5** Monitor and enforce responsible AI practices (bias, fairness, safety) in production models.
- **6.6** Configure alerts, logging, and dashboards for ML infrastructure and model health.