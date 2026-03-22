# NeMo Guardrails: Safety Framework for LLM Applications

**Source:** https://docs.nvidia.com/nemo/guardrails/

**Framework:** NVIDIA NeMo Guardrails
**License:** Apache 2.0
**GitHub:** https://github.com/NVIDIA-NeMo/Guardrails
**Focus:** Programmable safety controls for LLM conversational systems

## Overview

NeMo Guardrails is NVIDIA's open-source framework that allows developers to "easily add **programmable guardrails** between the application code and the LLM." This creates a protective layer for LLM-based applications by restricting harmful outputs, guiding dialogue flow, and enforcing safety constraints.

## Core Concepts

### What Are Guardrails ("Rails")?

Guardrails are specific ways of controlling LLM output to:
- Restrict topics (no politics, no hate speech)
- Guide conversation flow (specific dialogue paths)
- Enforce language style (formal, casual, technical)
- Extract structured data
- Prevent jailbreaks and prompt injections
- Implement authentication workflows
- Follow business logic rules

### Design Philosophy

**Runtime Protection:** Guardrails operate at runtime rather than through model training

**LLM-Independent:** Works with multiple LLM providers (OpenAI, Anthropic, local models)

**User-Defined:** Developers define guardrails, not model creators

**Interpretable:** Clear, auditable decision rules

## Core Safety Features

### Input Rail Protection

**Purpose:** Validate and filter user inputs before reaching the LLM

**Mechanisms:**
- Jailbreak detection and blocking
- Prompt injection prevention
- Content filtering
- Input format validation
- Intent detection

**Example:**
```yaml
input_rails:
  - type: "regex"
    pattern: "DROP TABLE.*"
    action: "block"
    message: "SQL injection detected"
```

### Output Rail Protection

**Purpose:** Validate and filter LLM outputs before returning to user

**Mechanisms:**
- Harmfulness detection
- Fact checking
- Consistency validation
- Format compliance
- Sensitive information masking

**Example:**
```colang
output_rail:
  - check_for_harmful_content()
  - check_facts_against_knowledge_base()
  - mask_pii()
```

## Guardrail Types

### Topical Restrictions

**Purpose:** Prevent discussions of specific topics

**Implementation:**
```colang
define restricted_topics:
  - "politics"
  - "religion"
  - "confidential business"

define should_avoid_topic:
  "is_topic_about" in message.intent.topics
  and any(topic in restricted_topics for topic in message.intent.topics)

define output_rail:
  if should_avoid_topic():
    return "I cannot discuss that topic"
```

### Dialogue Path Control

**Purpose:** Enforce specific conversation flows

**Use Cases:**
- Authentication workflows (verify identity first)
- Support procedures (follow escalation path)
- Onboarding flows (collect required information)

**Implementation:**
```colang
define flow authenticate_then_help:
  user_msg: authenticate_user()
  if authenticated:
    show_help(user_msg)
  else:
    return "Please authenticate first"
```

### Information Extraction

**Purpose:** Extract structured data from conversations

**Implementation:**
```colang
define extract_contact_info:
  extract(["name", "email", "phone"])
  from_user_messages()
```

### Tool Integration

**Purpose:** Safely connect LLMs to external services

**Security:** Validate tool calls before execution

**Implementation:**
```colang
define safe_tool_call:
  tool_call = parse_function_call()
  if is_safe(tool_call):
    return execute_tool(tool_call)
  else:
    return "Tool call blocked for security"
```

## Colang Language

### Colang 1.0 (Established)

**Syntax:** User-friendly dialogue specification

**Example:**
```colang
define user_greeting:
  "hello" | "hi" | "hey"

define bot_greeting:
  "Hello! How can I help you today?"

define greeting_flow:
  user: user_greeting()
  bot: bot_greeting()
```

### Colang 2.0 (Modern)

**Improvements:**
- More expressive syntax
- Better code organization
- Easier debugging
- Enhanced performance

**Example:**
```colang
flow greeting
  user: "hello" or "hi"
  bot: "Hello! How can I help?"
  user: ...
```

## Configuration

### LLM Configuration

```python
from nemo_guardrails import LLMRails

rails = LLMRails(
    config_path="config/",
    llm_config={
        "type": "openai",
        "model": "gpt-4",
        "api_key": "${OPENAI_API_KEY}"
    }
)
```

### Knowledge Base Integration

```yaml
knowledge_base:
  type: "vector_db"
  path: "path/to/documents"
  embedder: "sentence-transformers/all-MiniLM-L6-v2"
  retriever_config:
    top_k: 5
    threshold: 0.7
```

### Tracing Configuration

```yaml
tracing:
  enabled: true
  logs_dir: "./rails_logs"
  capture_events: true
```

## Integration Patterns

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from nemo_guardrails import LLMRails

llm = ChatOpenAI()
rails = LLMRails(llm=llm, config_path="config/")

# Use with LangChain
response = rails.invoke(
    messages=[{"role": "user", "content": "Hello"}]
)
```

### LangGraph Integration

```python
from langgraph.graph import Graph
from nemo_guardrails import LLMRails

# Create agentic graph with guardrails
graph = Graph()
rails = LLMRails(config_path="config/")

# Insert guardrails at decision points
graph.add_node("validate_input", rails.validate_input)
graph.add_node("generate_response", rails.generate_response)
```

### Streaming Integration

```python
rails = LLMRails(config_path="config/")

# Stream responses with guardrails
for token in rails.generate_stream("Your message"):
    print(token, end="", flush=True)
```

## Security Guidelines

### Input Validation

- Sanitize all inputs
- Check for prompt injection patterns
- Validate input format and size
- Rate limit requests

### Output Validation

- Check for sensitive information
- Validate factual accuracy
- Check for harmful content
- Ensure response format

### Tool Security

- Whitelist allowed tools
- Validate tool parameters
- Check permissions before execution
- Log all tool calls

### Model Security

- Use secure LLM APIs
- Encrypt API keys
- Implement audit logging
- Regular security updates

## Advanced Features

### Multi-Turn Conversations

Support for extended dialogue with context retention:

```colang
define conversation_flow:
  user: initial_message()
  bot: respond_to_user()

  while user_wants_to_continue():
    user: follow_up()
    bot: respond_to_follow_up()

  bot: closing_message()
```

### Multimodal Support

Handle text, images, and other modalities:

```colang
define handle_image:
  if message.contains_image():
    analyze_image(message.image)
  else:
    return "Please provide an image"
```

### Custom Initialization

```python
from nemo_guardrails import LLMRails

rails = LLMRails(
    config_path="config/",
    custom_init_func=initialize_custom_components
)
```

## Deployment Strategies

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.0-runtime

RUN pip install nemo-guardrails

COPY config/ /app/config/
WORKDIR /app

CMD ["python", "guardrails_app.py"]
```

### Self-Hosted Model

```python
rails = LLMRails(
    config_path="config/",
    llm_config={
        "type": "huggingface",
        "model": "meta-llama/Llama-2-7b-chat",
        "device": "cuda"
    }
)
```

### Production Deployment

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: guardrails-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: guardrails:latest
        resources:
          requests:
            nvidia.com/gpu: 1
```

## Evaluation & Monitoring

### Test Suite

```python
from nemo_guardrails.evaluation import EvaluationRunner

evaluator = EvaluationRunner(
    config_path="config/",
    test_cases=[
        {"input": "jailbreak attempt", "expected": "blocked"},
        {"input": "normal question", "expected": "answered"},
    ]
)
results = evaluator.run()
```

### Metrics

Track guardrail effectiveness:
- Input rejection rate
- Output filtering rate
- False positive rate
- Response latency
- User satisfaction

## Use Cases

### Customer Support Bot

- Authenticate users
- Restrict topics (billing policies, confidential info)
- Escalate complex queries
- Ensure polite tone

### Content Moderation System

- Block harmful content
- Detect misinformation
- Mask personally identifiable information
- Maintain moderation logs

### Enterprise Assistant

- Enforce access controls
- Restrict data access
- Follow compliance rules
- Audit all interactions

### Chatbot Safety

- Prevent jailbreaks
- Block misinformation
- Maintain brand voice
- Reject harmful requests

## Best Practices

### Design

- [ ] Define clear safety requirements
- [ ] Design comprehensive guardrail set
- [ ] Test edge cases
- [ ] Plan for updates

### Implementation

- [ ] Start with strict guardrails
- [ ] Relax incrementally based on testing
- [ ] Implement comprehensive logging
- [ ] Set up monitoring

### Deployment

- [ ] Test in staging environment
- [ ] Monitor guardrail performance
- [ ] Collect user feedback
- [ ] Update guardrails regularly

### Maintenance

- [ ] Review rejection logs
- [ ] Adjust false positive rate
- [ ] Update guardrails for new threats
- [ ] Train team on guardrail configuration

## Conclusion

NeMo Guardrails provides production-ready safety controls for LLM applications. By enabling programmable, interpretable guardrails independent of the underlying model, it allows organizations to deploy LLM applications safely while maintaining flexibility and control.

Whether building customer-facing chatbots, enterprise assistants, or content moderation systems, NeMo Guardrails provides essential protection against jailbreaks, prompt injection, harmful content, and compliance violations—enabling confident deployment of LLM applications in production environments.
