# Building RAG Agents with NVIDIA Nemotron

**Source:** https://developer.nvidia.com/blog/build-a-rag-agent-with-nvidia-nemotron/

**Framework:** LangChain + LangGraph with NVIDIA Nemotron
**Focus:** Agentic RAG systems with autonomous decision-making
**Audience:** Developers building production RAG agents

## Overview

Traditional RAG systems automatically retrieve documents for every query. **Agentic RAG** combines retrieval-augmented generation with autonomous decision-making—the agent decides whether to retrieve information via tool calling or respond directly.

This approach enables:
- **Adaptive behavior** - Retrieve only when necessary
- **Complex reasoning** - Multi-step decision-making
- **Cost efficiency** - Avoid unnecessary retrievals
- **Better responses** - Informed by selective retrieval

## Architecture Components

### Core Models

**Primary LLM: Nemotron Nano 9B V2**
- Role: Main reasoning and response generation
- Size: 9 billion parameters
- Specialty: Instruction following and tool calling
- Speed: Fast inference suitable for agents
- Quality: High quality responses

**Embedding Model: Llama 3.2 EmbedQA 1B V2**
- Role: Convert documents to vector embeddings
- Size: 1 billion parameters
- Specialty: Question-answering focused embeddings
- Use: Semantic search in knowledge base
- Speed: Extremely fast embedding

**Reranking Model: Llama 3.2 RerankQA 1B V2**
- Role: Rerank retrieved documents by relevance
- Size: 1 billion parameters
- Specialty: Relevance assessment
- Use: Improve retrieval quality
- Benefit: Filters out irrelevant results

### ReAct Agent Architecture

**ReAct Pattern:** Reasoning + Acting

**Cycle:**
1. **Reasoning Phase** - Agent thinks about what to do
2. **Action Phase** - Agent decides to act (call tool) or respond
3. **Observation Phase** - Agent receives result
4. **Loop** - Continue until task complete

**Decision Point:** For each query, agent decides:
- **Retrieve** - Call retrieval tool (if additional context needed)
- **Respond** - Answer directly from training knowledge

## Implementation Steps

### Step 1: Data Ingestion

**Purpose:** Load knowledge base documents

**Tools:** LangChain's DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="./knowledge_base",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
```

**Considerations:**
- Load various document formats (PDF, TXT, Markdown)
- Handle encoding properly
- Track document metadata (source, date, author)

### Step 2: Text Splitting

**Purpose:** Break documents into manageable chunks

**Tool:** RecursiveCharacterTextSplitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Tokens per chunk
    chunk_overlap=120,   # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
```

**Parameters:**
- **chunk_size:** 500-1000 tokens typical
- **chunk_overlap:** 10-20% overlap recommended
- **separators:** Preserve document structure

**Rationale:**
- Chunks too small → fragmented context
- Chunks too large → slower retrieval
- Overlap prevents context loss at boundaries

### Step 3: Vector Database Population

**Purpose:** Create searchable semantic index

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save for reuse
vector_store.save_local("./faiss_index")
```

**Database Options:**
- **FAISS:** Fast local similarity search (CPU/GPU)
- **Milvus:** Scalable vector database
- **Weaviate:** GraphQL-based vector DB
- **Pinecone:** Managed vector search service

### Step 4: Retrieval Chain

**Purpose:** Retrieve and rerank relevant documents

```python
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Semantic retriever (vector search)
semantic_retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(
    documents=chunks
)

# Ensemble for hybrid search
ensemble = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# Add reranking
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListwiseRerank

compressor = LLMListwiseRerank(
    llm=llm,
    top_n=5
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble
)
```

**Quality Optimization:**
- Semantic search for meaning
- BM25 for keyword matching
- Reranking to surface best results

### Step 5: Tool Creation

**Purpose:** Wrap retriever as agent tool

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description="Search for relevant information in the knowledge base"
)
```

**Tool Specification:**
- **Name:** Used by agent to identify tool
- **Description:** Tells agent when to use tool
- **Return Type:** Document chunks with metadata

### Step 6: Agent Configuration

**Purpose:** Build ReAct agent with LLM and tools

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Get ReAct prompt template
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=[retriever_tool],
    prompt=prompt
)

# Create executor
executor = AgentExecutor(
    agent=agent,
    tools=[retriever_tool],
    verbose=True,
    max_iterations=10
)

# Run agent
result = executor.invoke({
    "input": "What is the company's vacation policy?"
})
```

**Agent Loop:**
1. Receive user query
2. LLM decides: retrieve or respond
3. If retrieve: call tool, process results
4. LLM generates response
5. Return to user

## Advanced Implementation with LangGraph

### Explicit State Management

```python
from langgraph.graph import Graph, END
from langchain.schema import BaseMessage

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list

# Build graph
workflow = Graph()

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("end", END)

# Add edges with conditions
workflow.add_conditional_edges(
    "retrieve",
    should_retrieve,
    {True: "generate", False: "end"}
)

workflow.add_edge("generate", "end")

# Compile graph
agent_graph = workflow.compile()
```

## Best Practices

### System Prompts

**Comprehensive Instructions:**
```
You are a helpful AI assistant specializing in company policies.

When answering questions:
1. First determine if you need additional information from the knowledge base
2. If needed, use the search tool to find relevant documents
3. Ground your response in retrieved documents when available
4. Always cite sources for factual claims

When using the search tool:
- Be specific about what information you need
- Combine multiple searches if necessary
- Review results carefully before using

Format responses clearly with:
- Main answer
- Supporting evidence from documents
- Source citations
```

### Retrieval Quality

**Techniques:**
- **Query Expansion** - Rephrase user queries to improve search
- **Semantic Chunking** - Split documents by meaning, not size
- **Metadata Filtering** - Restrict search to relevant document categories
- **Result Validation** - Check that retrieved documents match query intent

### Performance Optimization

**Caching:**
```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

**Batch Processing:**
```python
queries = ["Query 1", "Query 2", "Query 3"]
results = [executor.invoke({"input": q}) for q in queries]
```

**Async Execution:**
```python
import asyncio

async def process_async(query):
    return await executor.ainvoke({"input": query})

results = await asyncio.gather(*[
    process_async(q) for q in queries
])
```

## Production Deployment

### Migration from Cloud APIs to NIM

**Development (Cloud):**
```python
from langchain.llms import OpenAI

llm = OpenAI(api_key="sk-...")
```

**Production (NIM):**
```python
from langchain.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8000/v1",
    model_id="nemotron-nano-9b-v2"
)
```

### Scaling Considerations

**Vector Database Scaling:**
- Single node: FAISS (up to 1M documents)
- Multi-node: Milvus, Weaviate (unlimited)
- Managed: Pinecone for simplicity

**Agent Orchestration:**
- LangSmith for tracing and monitoring
- Persistent state storage
- Request queuing for throughput

**Cost Management:**
- Cache frequent queries
- Batch similar requests
- Use smaller models when possible
- Monitor API usage

## Evaluation & Testing

### Query Test Set

```python
test_queries = [
    {
        "query": "What is the vacation policy?",
        "expected": ["policy", "days", "approval"]
    },
    {
        "query": "How many days off do employees get?",
        "expected": ["vacation days", "time off"]
    }
]

for test in test_queries:
    result = executor.invoke({"input": test["query"]})
    # Check if expected keywords in response
    assert any(word in result["output"].lower()
              for word in test["expected"])
```

### Metrics

- **Retrieval Precision** - % of retrieved docs relevant to query
- **Response Accuracy** - % of factually correct responses
- **Latency** - Time to generate response
- **Cost** - Compute cost per query

## Use Cases

### Documentation Systems
- Internal knowledge base
- Customer support
- Technical specifications
- Policy reference

### Research Assistants
- Literature search
- Citation management
- Summary generation
- Fact verification

### Business Intelligence
- Data analysis
- Report generation
- Decision support
- Trend analysis

## Troubleshooting

### Poor Retrieval Quality

**Cause:** Documents not relevant to queries
**Solutions:**
- Improve chunk size/overlap
- Add metadata filtering
- Use reranking
- Expand query formulation

### Slow Agent Response

**Cause:** Too many retrieve/think loops
**Solutions:**
- Set `max_iterations` limit
- Improve prompt clarity
- Pre-compute common queries
- Cache retrieval results

### High Cost

**Cause:** Frequent API calls
**Solutions:**
- Reduce number of retrievals
- Use smaller models
- Implement caching
- Batch queries

## Conclusion

Agentic RAG with Nemotron enables intelligent systems that dynamically decide when external knowledge is needed. By combining:

- **Agentic reasoning** (decide when to retrieve)
- **Efficient retrieval** (semantic search + reranking)
- **Specialized models** (Nemotron for reasoning, Llama for embeddings/reranking)
- **Production framework** (LangChain/LangGraph)

Organizations build responsive, accurate, cost-efficient AI assistants that adapt to diverse queries while maintaining quality and performance at scale.
