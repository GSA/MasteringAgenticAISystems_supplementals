# Databricks Generative AI Engineer Associate – Knowledge Items

## 1. Design Applications

- **1.1** Design a prompt that elicits a specifically formatted response.
- **1.2** Select model tasks to accomplish a given business requirement.
- **1.3** Select chain components for a desired model input and output.
- **1.4** Translate business use case goals into a description of the desired inputs and outputs for the AI pipeline.
- **1.5** Define and order tools that gather knowledge or take actions for multi‑stage reasoning.

***

## 2. Data Preparation

- **2.1** Apply a chunking strategy for a given document structure and model constraints.
- **2.2** Filter extraneous content in source documents that degrades quality of a RAG application.
- **2.3** Choose the appropriate Python package to extract document content from provided source data and format.
- **2.4** Define operations and sequence to write given chunked text into Delta Lake tables in Unity Catalog.
- **2.5** Identify needed source documents that provide necessary knowledge and quality for a given RAG application.
- **2.6** Identify prompt/response pairs that align with a given model task.
- **2.7** Use tools and metrics to evaluate retrieval performance.
- **2.8** Design retrieval systems using advanced chunking strategies.
- **2.9** Explain the role of re‑ranking in the information retrieval process.

***

## 3. Application Development

- **3.1** Create tools needed to extract data for a given data retrieval need.
- **3.2** Select LangChain or similar tools for use in a Generative AI application.
- **3.3** Identify how prompt formats can change model outputs and results.
- **3.4** Qualitatively assess responses to identify common issues such as quality and safety.
- **3.5** Select chunking strategy based on model and retrieval evaluation.
- **3.6** Augment a prompt with additional context from a user’s input based on key fields, terms, and intents.
- **3.7** Create a prompt that adjusts an LLM’s response from a baseline to a desired output.
- **3.8** Implement LLM guardrails to prevent negative outcomes.
- **3.9** Write metaprompts that minimize hallucinations or leaking private data.
- **3.10** Build agent prompt templates exposing available functions.
- **3.11** Select the best LLM based on the attributes of the application to be developed.
- **3.12** Select an embedding model context length based on source documents, expected queries, and optimization strategy.
- **3.13** Select a model from a model hub or marketplace for a task based on model metadata/model cards.
- **3.14** Select the best model for a given task based on common metrics generated in experiments.
- **3.15** Utilize Agent Framework for developing agentic systems.

***

## 4. Assembling and Deploying Applications

- **4.1** Code a chain using a pyfunc model with pre‑ and post‑processing.
- **4.2** Control access to resources from model serving endpoints.
- **4.3** Code a simple chain according to requirements.
- **4.4** Code a simple chain using LangChain.
- **4.5** Choose the basic elements needed to create a RAG application (model flavor, embedding model, retriever, dependencies, input examples, model signature).
- **4.6** Register the model to Unity Catalog using MLflow.
- **4.7** Sequence the steps needed to deploy an endpoint for a basic RAG application.
- **4.8** Create and query a Vector Search index.
- **4.9** Identify how to serve an LLM application that leverages Foundation Model APIs.
- **4.10** Identify resources needed to serve features for a RAG application.
- **4.11** Explain the key concepts and components of Mosaic AI Vector Search.
- **4.12** Identify batch inference workloads and apply `ai_query()` appropriately.

***

## 5. Governance

- **5.1** Use masking techniques as guard rails to meet a performance objective.
- **5.2** Select guardrail techniques to protect against malicious user inputs to a GenAI application.
- **5.3** Recommend an alternative for problematic text mitigation in a data source feeding a RAG application.
- **5.4** Use legal/licensing requirements for data sources to avoid legal risk.
- **5.5** Recommend an alternative for problematic text mitigation in a data source feeding a GenAI application.

***

## 6. Evaluation and Monitoring

- **6.1** Select an LLM choice (size and architecture) based on a set of quantitative evaluation metrics.
- **6.2** Select key metrics to monitor for a specific LLM deployment scenario.
- **6.3** Evaluate model performance in a RAG application using MLflow.
- **6.4** Use inference logging to assess deployed RAG application performance.
- **6.5** Use Databricks features to control LLM costs for RAG applications.
- **6.6** Use inference tables and Agent Monitoring to track a live LLM endpoint.
- **6.7** Identify evaluation judges that require ground truth.
- **6.8** Compare the evaluation and monitoring phases of the GenAI application life cycle.


