# NVIDIA Agent Intelligence Toolkit FAQ

**Source:** https://docs.nvidia.com/aiqtoolkit/latest/resources/faq.html

## Frequently Asked Questions

### Q: Do I Need to Rewrite All of My Existing Code to Use AIQ Toolkit?

**A:** No. The toolkit is **"100% opt in,"** allowing you to integrate at whatever level suits your needs—**tool, agent, or workflow level**. You can start small and expand where you see the most value.

**Implication:** Low barrier to entry. Organizations can adopt AIQ toolkit incrementally without disrupting existing systems.

---

### Q: Is AIQ Toolkit another LLM or Agentic Framework?

**A:** No. It's designed to **complement existing agentic frameworks** rather than replace them, whether you're using:
- Enterprise frameworks
- Simple Python-based agents
- LangChain
- LlamaIndex
- CrewAI
- Microsoft Semantic Kernel

**Implication:** AIQ toolkit sits alongside your current stack, not against it.

---

### Q: Is AIQ Toolkit an Attempt to Solve Agent-to-Agent Communication?

**A:** No. The toolkit doesn't address agent communication directly. That responsibility is best handled through **established protocols** such as:
- **MCP** (Model Context Protocol)
- **HTTP**
- **gRPC**
- **Sockets**

**Implication:** Use standard communication patterns for agent coordination. AIQ toolkit focuses on other aspects of agent development.

---

### Q: Is AIQ Toolkit an Observability Platform?

**A:** No. While it **collects and transmits telemetry** for optimization and evaluation purposes, it **doesn't replace dedicated observability platforms** and data collection applications.

**Intended Use:** AIQ toolkit is a complementary tool that works alongside your observability stack, not as a replacement.

**Integration Points:**
- Send AIQ metrics to your observability platform
- Use AIQ profiling for agent-specific insights
- Maintain existing monitoring systems

---

## Key Principles

### 1. Composability
AIQ toolkit components are designed to be **composable and reusable** across different contexts and frameworks.

### 2. Framework Agnostic
Work with **any agentic framework**. AIQ toolkit enhances rather than constrains your choice of tools.

### 3. Gradual Adoption
- Start with single tools
- Expand to agents
- Eventually manage entire workflows
- Move at your own pace

### 4. Existing Protocols
Leverage industry-standard communication protocols rather than reinventing the wheel.

### 5. Complementary, Not Competitive
- Works alongside your existing tools
- Enhances them with profiling and evaluation
- Doesn't replace your monitoring strategy
- Plays well with other solutions

## Integration Patterns

### Pattern 1: Tool-Level Integration
Wrap individual tools with AIQ decorators to get profiling without touching agents or workflows.

### Pattern 2: Agent-Level Integration
Integrate at the agent level to track decision-making and tool selection.

### Pattern 3: Workflow-Level Integration
Manage entire orchestration workflows with AIQ toolkit visibility.

## Supported Frameworks

AIQ toolkit integrates with:
- **LangChain** - Fully supported
- **LlamaIndex** - Full integration
- **CrewAI** - Compatible
- **Microsoft Semantic Kernel** - Supported
- **Google ADK** - Recently added
- **Custom Python agents** - Supported
- **Enterprise frameworks** - Generally compatible

## Communication Patterns

For agent-to-agent or service-to-service communication, AIQ toolkit recommends:

| Protocol | Best For | Example |
|----------|----------|---------|
| **HTTP** | REST APIs, microservices | API calls between agents |
| **WebSocket** | Real-time messaging, streaming | Live agent coordination |
| **gRPC** | High-performance services | Service mesh communication |
| **MCP** | Tool discovery and registration | Standard tool interfaces |
| **Sockets** | Direct connections | Low-latency communication |

## Quick Reference

### What AIQ Toolkit DOES
✓ Profile agent and tool performance
✓ Evaluate workflow quality
✓ Provide observability integrations
✓ Work with existing frameworks
✓ Support gradual adoption
✓ Enable workflow optimization

### What AIQ Toolkit DOES NOT
✗ Replace your agentic framework
✗ Handle agent communication directly
✗ Replace dedicated observability platforms
✗ Require system-wide rewrites
✗ Lock you into specific tools
✗ Manage infrastructure

## Getting Started Tips

1. **Start small** - Wrap one tool to see value
2. **Integrate gradually** - Expand as you see benefits
3. **Use existing protocols** - For communication between agents
4. **Combine with monitoring** - Use alongside your observability stack
5. **Leverage built-in evaluators** - Don't reinvent evaluation logic

## Common Misconceptions

### "AIQ toolkit requires switching frameworks"
**False.** It works alongside your current framework.

### "AIQ toolkit replaces observability platforms"
**False.** It's complementary. Use both together.

### "AIQ toolkit handles all agent communication"
**False.** Use MCP, HTTP, gRPC for communication. AIQ toolkit handles profiling and evaluation.

### "Adopting AIQ toolkit is an all-or-nothing decision"
**False.** Start with tools, expand to agents, move to workflows at your pace.

## Best Practices

1. **Evaluate early** - Start with evaluation to understand baseline performance
2. **Profile systematically** - Identify bottlenecks before optimizing
3. **Use standard protocols** - For agent communication, not custom solutions
4. **Integrate monitoring** - Combine AIQ toolkit metrics with your observability platform
5. **Iterate on evaluations** - Continuously refine your evaluation criteria
