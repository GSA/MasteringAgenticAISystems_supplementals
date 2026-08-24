# Section 2.11: Hands-On Lab - Multi-Framework Comparison

# Hands-On Lab 11: Multi-Framework Comparison

## Lab Overview

**Parent Chapter**: Part 2 - Agent Development Frameworks
**Parent Section**: Chapter 2.1-2.6 - Framework Selection and Implementation
**Lab Type**: Guided Project
**Difficulty**: Intermediate
**Estimated Time**: 3 hours (15 min setup + 135 min implementation + 30 min validation)

### Why This Lab Matters

Throughout this chapter, you've explored five distinct agent frameworks—LangChain, LangGraph, AutoGen, CrewAI, and Semantic Kernel—each with different strengths and trade-offs. Reading about these frameworks provides conceptual understanding, but making informed architectural decisions for production systems requires hands-on experience. This lab addresses that gap by having you build the same weather agent in three frameworks: LangChain, LangGraph, and AutoGen.

By implementing identical functionality across frameworks, you'll develop an intuitive sense for when each framework's architecture truly shines versus when it introduces unnecessary complexity. You'll experience firsthand the difference between LangChain's rapid prototyping for sequential workflows, LangGraph's explicit state management for complex routing, and AutoGen's conversational multi-agent patterns. More importantly, you'll build practical evaluation skills—comparing code complexity, developer experience, debugging visibility, and performance—that transfer to any framework selection decision in your career.

This isn't a tutorial to memorize; it's a comparative experiment designed to calibrate your architectural judgment. When you finish, you won't just know these frameworks—you'll understand when to choose each one based on direct implementation experience rather than marketing claims.

### Lab Objectives

By completing this lab, you will:

1. **Implement Identical Functionality Across Frameworks**: Build the same weather agent in LangChain, LangGraph, and AutoGen to understand framework-specific approaches
2. **Compare Implementation Complexity**: Analyze code structure, boilerplate requirements, and developer experience across three major agent frameworks
3. **Evaluate Framework Trade-offs**: Make informed framework selection decisions based on hands-on experience with features, performance, and maintainability
4. **Apply Best Practices**: Implement proper error handling, tool integration, and state management patterns for each framework
5. **Benchmark Performance**: Measure and compare latency, resource usage, and reliability across framework implementations

### Prerequisites

**Required Knowledge:**
- Understanding of agent architectures from Chapter 2.1
- Familiarity with ReAct pattern from Part 1
- Basic proficiency in Python and async programming
- Understanding of REST APIs and JSON

**Required Skills:**
- Can write Python functions and classes independently
- Comfortable with virtual environments and pip
- Can read API documentation
- Basic understanding of agent reasoning loops

**Self-Assessment:**
- [ ] I can explain the ReAct pattern (Reasoning + Acting)
- [ ] I understand what tools/functions are in the context of agents
- [ ] I have completed Part 1 labs or equivalent
- [ ] I can make API calls in Python

**If you checked fewer than all boxes**: Review Part 1 Chapter 1.2-1.4 before proceeding.

---

## Lab Setup

### Why Proper Environment Setup Matters

Before diving into implementation, we need to establish a consistent development environment across all three frameworks. This consistency is critical for fair comparison—if one framework works in Python 3.11 while another requires 3.10, or if dependency conflicts force different virtual environments, you can't accurately compare developer experience or performance. The lab repository provides pre-configured dependencies, starter code templates, and verification scripts to ensure you're comparing frameworks on level ground rather than debugging environment issues.

Additionally, this setup introduces you to a realistic agent development workflow: cloning structured repositories, managing API keys securely through environment variables, and validating your environment before writing code. Production agent systems require this discipline from day one—discovering API key issues or version conflicts during a critical deployment is far costlier than catching them during initial setup.

### Environment Requirements

**Hardware:**
- Minimum: 4GB RAM, dual-core CPU
- Recommended: 8GB RAM, quad-core CPU
- NVIDIA GPU: Not required

**Software:**
- Operating System: Linux, macOS, or Windows with WSL2
- Python: 3.10+
- Internet connection for API calls

**Cloud Alternative:**
- Google Colab (free tier sufficient)
- GitHub Codespaces (2-core instance)

### Setup Instructions

**Step 1: Clone Lab Repository**

```bash
# Clone the lab materials
git clone https://github.com/nvidia-agentic-ai-cert/book-labs
cd book-labs/chapter-02/lab-01-framework-comparison

# Verify structure
ls -la
# You should see:
# - README.md
# - requirements.txt
# - starter_code/
# - tests/
# - data/
# - benchmark_config.yml
```

**Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Verify activation
which python
# Should show path to venv/bin/python
```

**Step 3: Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify key packages
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import langgraph; print(f'LangGraph: {langgraph.__version__}')"
python -c "import autogen; print(f'AutoGen: {autogen.__version__}')"
```

**Step 4: Set Up API Keys**

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
WEATHER_API_KEY=your_weatherapi_key_here
EOF

# Get free Weather API key from: https://www.weatherapi.com/signup.aspx
# Note: Free tier allows 1M calls/month - sufficient for this lab
```

**Step 5: Run Setup Verification**

```bash
# Run verification script
python verify_setup.py

# Expected output:
# ✓ Python version: 3.10+
# ✓ All required packages installed
# ✓ Environment variables loaded
# ✓ API keys configured
# ✓ Weather API accessible
# ✓ Environment ready!
```

**Troubleshooting Setup Issues:**

If you encounter module import errors for langchain, langgraph, or autogen, the most common cause is incomplete installation or version conflicts. Run `pip install --force-reinstall -r requirements.txt` to ensure all packages and their dependencies are properly installed. This forces pip to reinstall everything, resolving version mismatches that can occur when packages were previously installed in different contexts.

When the Weather API returns 401 Unauthorized errors, this indicates an authentication problem with your API key. Verify that the key is correctly copied into your .env file without extra spaces or quotes, then check your account status at weatherapi.com to confirm the key is active and hasn't exceeded rate limits. Remember that API keys are sensitive credentials—never commit them to version control or share them publicly.

---

## Lab Instructions

### Part 1: Understanding the Problem

**Problem Statement**

You are building a weather information agent that can answer natural language queries about weather conditions in various cities. The agent must:
- Understand user queries in natural language
- Determine when to call the weather API
- Retrieve current weather data
- Format responses in a user-friendly manner

**Scenario:**
Your team is evaluating three frameworks (LangChain, LangGraph, AutoGen) for building production agents. Before committing to one framework, you need to implement a proof-of-concept agent in all three frameworks to compare:
- Developer experience and code complexity
- Feature capabilities and extensibility
- Performance and reliability
- Maintenance and debugging ease

**Requirements:**
1. Agent must support conversational queries: "What's the weather in San Francisco?"
2. Agent must call Weather API tool when needed
3. Agent must handle errors gracefully (invalid cities, API failures)
4. Agent must support multi-turn conversations with context
5. Response time should be <3 seconds for simple queries

**Success Criteria:**
- [ ] All three agents successfully answer weather queries
- [ ] Each implementation follows framework best practices
- [ ] Error handling works for invalid inputs
- [ ] Performance benchmarks show <3s latency
- [ ] Code is well-documented and maintainable

**Architecture Overview**

**System Components:**

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│              (Shared across all frameworks)             │
└───────────────────┬─────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   LangChain     │   │   LangGraph     │   │    AutoGen      │
│   Agent         │   │   Agent         │   │    Agent        │
│                 │   │                 │   │                 │
│  - AgentExec    │   │  - StateGraph   │   │  - ConvAgent    │
│  - Tools        │   │  - Nodes/Edges  │   │  - Functions    │
│  - Memory       │   │  - Checkpoints  │   │  - GroupChat    │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Weather API Tool  │
                    │   (Shared service)  │
                    └─────────────────────┘
```

**Component Responsibilities:**
- **User Interface**: Consistent CLI for testing all implementations
- **LangChain Agent**: Uses AgentExecutor with tool binding
- **LangGraph Agent**: State graph with explicit nodes for reasoning/acting
- **AutoGen Agent**: Conversational agent with function calling
- **Weather API Tool**: Shared weather data retrieval service

**Review Key Concepts**

Before implementing, review these concepts from the chapter:

**Concept 1: Tool/Function Calling** (Chapter 2.2)
- **Definition**: Agents extend LLM capabilities by calling external functions
- **Why it's used here**: Weather data retrieval requires API calls beyond LLM knowledge

**Concept 2: ReAct Pattern** (Chapter 2.3)
- **Definition**: Reasoning (decide what to do) + Acting (execute tool calls) loop
- **Why it's used here**: Agent must reason about when to call weather API vs answer directly

**Concept 3: Agent State Management** (Chapter 2.4)
- **Definition**: Maintaining conversation context and intermediate results
- **Why it's used here**: Multi-turn conversations require tracking chat history

---

### Part 2: Starter Code Walkthrough

With your environment configured and the problem scope clear, let's examine the provided starter code structure. The repository organizes code into framework-specific directories while sharing common utilities—a pattern you'll often see in production codebases that support multiple agent frameworks. Understanding this structure before coding helps you focus on framework-specific implementation rather than boilerplate concerns.

Navigate to `starter_code/` directory:

```bash
cd starter_code/
ls -la

# File structure:
# - common/
#   - weather_tool.py       (Shared weather API wrapper)
#   - base_agent.py         (Abstract base class)
#   - utils.py              (Logging, timing utilities)
# - langchain_agent/
#   - agent.py              (TODO: Implement)
#   - config.py             (Configuration)
# - langgraph_agent/
#   - agent.py              (TODO: Implement)
#   - state.py              (State schema)
# - autogen_agent/
#   - agent.py              (TODO: Implement)
#   - config.py             (Configuration)
# - main.py                 (CLI for testing)
# - benchmark.py            (Performance comparison)
```

**File: `common/weather_tool.py` (Provided)**

```python
"""
Shared Weather API Tool

This tool is used by all three agent implementations.
"""

import os
import requests
from typing import Dict, Any

class WeatherTool:
    """Weather API wrapper shared across all frameworks."""

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "http://api.weatherapi.com/v1/current.json"

    def get_weather(self, city: str) -> Dict[str, Any]:
        """
        Get current weather for a city.

        Args:
            city: City name (e.g., "San Francisco", "London, UK")

        Returns:
            Dict with weather data or error
        """
        try:
            response = requests.get(
                self.base_url,
                params={"key": self.api_key, "q": city},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            return {
                "city": data["location"]["name"],
                "country": data["location"]["country"],
                "temperature_c": data["current"]["temp_c"],
                "temperature_f": data["current"]["temp_f"],
                "condition": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "wind_kph": data["current"]["wind_kph"]
            }
        except requests.RequestException as e:
            return {"error": f"Failed to fetch weather: {str(e)}"}
```

**File: `common/base_agent.py` (Provided)**

```python
"""
Abstract base class for all agent implementations.

Ensures consistent interface across frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseWeatherAgent(ABC):
    """Base class for weather agents."""

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Process a weather query.

        Args:
            query: Natural language weather question

        Returns:
            Agent response as string
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state (clear conversation history)."""
        pass
```

---

### Part 3: Implementation Tasks

You're now ready to build three distinct agent implementations. Each task follows the same workflow—implement the agent class, configure framework-specific components, and validate with tests—but the implementation details reveal how frameworks approach agents differently. Pay attention not just to what you're building, but to how much code each framework requires, how explicit or implicit state management is, and how easy it is to understand what the agent is doing.

**Task 1: Implement LangChain Agent (45 minutes)**

**Objective**: Build weather agent using LangChain's AgentExecutor pattern

LangChain's AgentExecutor represents the framework's "batteries included" approach to agent development. You'll wrap your weather tool as a LangChain Tool, configure a prompt template that guides the agent's behavior, and let the AgentExecutor handle the ReAct loop automatically. This abstraction prioritizes developer velocity—you can build a working agent quickly—but as you implement it, notice where the framework makes decisions for you versus where you have explicit control.

**Instructions:**
1. Open `langchain_agent/agent.py`
2. Implement `LangChainWeatherAgent` class
3. Configure tool binding and agent executor
4. Test with sample queries

**Implementation Guide:**

```python
"""
LangChain Weather Agent Implementation

TODO: Complete this implementation
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from common.base_agent import BaseWeatherAgent
from common.weather_tool import WeatherTool

class LangChainWeatherAgent(BaseWeatherAgent):
    """LangChain implementation of weather agent."""

    def __init__(self):
        """
        Initialize LangChain agent.

        TODO: Implement initialization
        Steps:
        1. Create ChatOpenAI LLM instance
        2. Create Tool from WeatherTool
        3. Define prompt template with system message
        4. Create agent with create_openai_tools_agent
        5. Wrap in AgentExecutor
        6. Initialize chat history
        """

        # Initialize weather tool
        weather_tool_instance = WeatherTool()

        # TODO: Create LangChain Tool
        # Hint: Use Tool.from_function()
        # The tool should call weather_tool_instance.get_weather
        self.tools = None  # Replace with your implementation

        # TODO: Create LLM
        # Hint: Use ChatOpenAI with temperature=0 for consistency
        self.llm = None  # Replace with your implementation

        # TODO: Create prompt template
        # Include:
        # - System message: "You are a helpful weather assistant..."
        # - MessagesPlaceholder for chat_history
        # - User input placeholder
        # - MessagesPlaceholder for agent_scratchpad
        self.prompt = None  # Replace with your implementation

        # TODO: Create agent
        # Hint: Use create_openai_tools_agent(llm, tools, prompt)
        agent = None  # Replace with your implementation

        # TODO: Create AgentExecutor
        # Set return_intermediate_steps=True for debugging
        self.agent_executor = None  # Replace with your implementation

        # Chat history for multi-turn conversations
        self.chat_history = []

    def run(self, query: str) -> str:
        """
        Process a weather query.

        TODO: Implement query processing
        Steps:
        1. Invoke agent_executor with query and chat_history
        2. Extract output from result
        3. Update chat_history with user message and AI response
        4. Return response
        """
        try:
            # TODO: Invoke agent
            # Hint: agent_executor.invoke({"input": query, "chat_history": self.chat_history})
            result = None  # Replace with your implementation

            # TODO: Extract response
            response = None  # Replace with your implementation

            # TODO: Update chat history
            # Add HumanMessage(content=query) and AIMessage(content=response)

            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def reset(self):
        """Reset conversation history."""
        self.chat_history = []
```

**Key Concepts to Apply:**
- **Tool Creation**: LangChain's `Tool.from_function()` wraps Python functions
- **Prompt Engineering**: System prompts guide agent behavior
- **Chat History**: Maintains context across multiple turns

**Checkpoint 1:**
```bash
# Test LangChain implementation
python -m pytest tests/test_langchain_agent.py -v

# Manual test
python main.py --framework langchain --query "What's the weather in Tokyo?"

# Expected: Agent calls weather API and returns formatted response
```

**Task 2: Implement LangGraph Agent (45 minutes)**

**Objective**: Build weather agent using LangGraph's state graph pattern

LangGraph takes a fundamentally different approach: instead of hiding the agent loop behind an executor, you'll explicitly define the workflow as a graph with nodes and edges. This task requires more upfront design—you must define a state schema, implement node functions for reasoning and tool calling, and configure routing logic—but in exchange, you gain complete visibility into how the agent makes decisions. As you build this, contrast the explicitness with LangChain's abstraction.

**Instructions:**
1. Open `langgraph_agent/agent.py`
2. Define state schema in `state.py`
3. Implement graph nodes (reasoning, tool calling, response)
4. Configure state graph with edges
5. Compile and test

**State Schema (`langgraph_agent/state.py`):**

```python
"""
LangGraph Agent State Schema

TODO: Complete state definition
"""

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    State for weather agent graph.

    TODO: Define state fields
    Required fields:
    - messages: List of conversation messages
    - next_action: What the agent should do next ("call_tool", "respond", "end")
    - tool_result: Result from weather API (optional)
    """
    # Your implementation here
    pass
```

**Agent Implementation (`langgraph_agent/agent.py`):**

```python
"""
LangGraph Weather Agent Implementation

TODO: Complete this implementation
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from common.base_agent import BaseWeatherAgent
from common.weather_tool import WeatherTool
from .state import AgentState

class LangGraphWeatherAgent(BaseWeatherAgent):
    """LangGraph implementation of weather agent."""

    def __init__(self):
        """
        Initialize LangGraph agent.

        TODO: Implement initialization
        Steps:
        1. Create LLM
        2. Create WeatherTool instance
        3. Define graph nodes (reason, call_tool, respond)
        4. Build StateGraph
        5. Compile graph
        """

        self.llm = None  # TODO: Create ChatOpenAI instance
        self.weather_tool = WeatherTool()

        # TODO: Build state graph
        # Hint: Use StateGraph(AgentState)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the agent state graph.

        TODO: Implement graph construction

        Graph structure:
        START → reason → [call_tool OR respond]
        call_tool → respond
        respond → END
        """

        # TODO: Create StateGraph
        graph = StateGraph(AgentState)

        # TODO: Add nodes
        # graph.add_node("reason", self._reason_node)
        # graph.add_node("call_tool", self._call_tool_node)
        # graph.add_node("respond", self._respond_node)

        # TODO: Add edges
        # graph.set_entry_point("reason")
        # graph.add_conditional_edges("reason", self._should_call_tool)
        # graph.add_edge("call_tool", "respond")
        # graph.add_edge("respond", END)

        # TODO: Compile graph
        return graph.compile()

    def _reason_node(self, state: AgentState) -> AgentState:
        """
        Reasoning node: Decide if we need to call weather API.

        TODO: Implement reasoning logic
        Steps:
        1. Get last user message
        2. Use LLM to determine if weather API is needed
        3. Update state with next_action
        """
        # Your implementation here
        pass

    def _call_tool_node(self, state: AgentState) -> AgentState:
        """
        Tool calling node: Execute weather API call.

        TODO: Implement tool calling
        Steps:
        1. Extract city from last message
        2. Call weather_tool.get_weather(city)
        3. Update state with tool_result
        """
        # Your implementation here
        pass

    def _respond_node(self, state: AgentState) -> AgentState:
        """
        Response node: Generate final response.

        TODO: Implement response generation
        Steps:
        1. If tool_result exists, include in context
        2. Use LLM to generate natural language response
        3. Add response to messages
        """
        # Your implementation here
        pass

    def _should_call_tool(self, state: AgentState) -> str:
        """
        Conditional edge: Determine next node.

        TODO: Implement routing logic
        Returns: "call_tool" or "respond"
        """
        # Your implementation here
        pass

    def run(self, query: str) -> str:
        """
        Process a weather query.

        TODO: Implement query processing
        Steps:
        1. Create initial state with user message
        2. Invoke compiled graph
        3. Extract final response from state
        """
        try:
            # TODO: Create initial state
            initial_state = None  # Replace with your implementation

            # TODO: Invoke graph
            final_state = self.graph.invoke(initial_state)

            # TODO: Extract response
            response = None  # Replace with your implementation

            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def reset(self):
        """Reset agent state."""
        # LangGraph is stateless by design - no reset needed
        pass
```

**Key Concepts to Apply:**
- **State Graph**: Explicit nodes and edges for agent workflow
- **Conditional Routing**: Dynamic flow based on agent decisions
- **Immutable State**: Each node returns updated state

**Checkpoint 2:**
```bash
# Test LangGraph implementation
python -m pytest tests/test_langgraph_agent.py -v

# Manual test
python main.py --framework langgraph --query "How's the weather in Paris?"

# Expected: Agent processes query through graph nodes
```

**Task 3: Implement AutoGen Agent (45 minutes)**

**Objective**: Build weather agent using AutoGen's conversational agent pattern

AutoGen introduces a multi-agent perspective even for single-agent tasks. You'll create an AssistantAgent (the LLM-powered agent) and a UserProxyAgent (which executes functions on the assistant's behalf), then configure them to communicate through function calling. This architecture feels more complex for simple tasks, but as you implement it, consider how this pattern would scale to multi-agent scenarios where multiple assistants collaborate.

**Instructions:**
1. Open `autogen_agent/agent.py`
2. Implement `AutoGenWeatherAgent` class
3. Register weather function
4. Configure agent with function calling
5. Test with sample queries

**Implementation Guide:**

```python
"""
AutoGen Weather Agent Implementation

TODO: Complete this implementation
"""

import autogen
from typing import Dict, Any, Optional, List
from common.base_agent import BaseWeatherAgent
from common.weather_tool import WeatherTool

class AutoGenWeatherAgent(BaseWeatherAgent):
    """AutoGen implementation of weather agent."""

    def __init__(self):
        """
        Initialize AutoGen agent.

        TODO: Implement initialization
        Steps:
        1. Create LLM config for GPT-4
        2. Define weather function schema
        3. Create AssistantAgent with function calling
        4. Create UserProxyAgent for executing functions
        5. Register weather function
        """

        self.weather_tool = WeatherTool()

        # TODO: Create LLM config
        # Hint: Use autogen config format with model, api_key, temperature
        llm_config = None  # Replace with your implementation

        # TODO: Define function schema
        # AutoGen requires OpenAI function calling format
        weather_function_schema = {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., 'San Francisco, CA')"
                    }
                },
                "required": ["city"]
            }
        }

        # TODO: Create AssistantAgent
        # Set system_message, llm_config, and function_map
        self.assistant = None  # Replace with your implementation

        # TODO: Create UserProxyAgent
        # Set human_input_mode="NEVER" for automated execution
        # Set code_execution_config=False (no code execution needed)
        self.user_proxy = None  # Replace with your implementation

        # TODO: Register function
        # self.user_proxy.register_function(
        #     function_map={"get_weather": self._get_weather_wrapper}
        # )

        self.chat_history = []

    def _get_weather_wrapper(self, city: str) -> str:
        """
        Wrapper for weather tool to match AutoGen function signature.

        TODO: Implement wrapper
        Steps:
        1. Call self.weather_tool.get_weather(city)
        2. Format result as JSON string
        3. Handle errors gracefully
        """
        # Your implementation here
        pass

    def run(self, query: str) -> str:
        """
        Process a weather query.

        TODO: Implement query processing
        Steps:
        1. Initiate chat between user_proxy and assistant
        2. Extract final response
        3. Update chat history
        """
        try:
            # TODO: Initiate chat
            # Hint: self.user_proxy.initiate_chat(
            #     self.assistant,
            #     message=query,
            #     max_turns=3
            # )

            # TODO: Extract last message from assistant
            # AutoGen stores messages in chat_messages
            response = None  # Replace with your implementation

            # TODO: Update chat history

            return response

        except Exception as e:
            return f"Error processing query: {str(e)}"

    def reset(self):
        """Reset conversation history."""
        # TODO: Clear chat history for both agents
        self.chat_history = []
```

**Key Concepts to Apply:**
- **Function Calling**: AutoGen uses OpenAI function calling format
- **Multi-Agent Interaction**: UserProxyAgent executes functions for AssistantAgent
- **Conversation Management**: Chat history maintained across agents

**Checkpoint 3:**
```bash
# Test AutoGen implementation
python -m pytest tests/test_autogen_agent.py -v

# Manual test
python main.py --framework autogen --query "Tell me about the weather in Berlin"

# Expected: Agent uses function calling to get weather data
```

---

### Part 4: Framework Comparison

Now that you've implemented the same agent in three different frameworks, you're positioned to make evidence-based comparisons rather than relying on documentation claims. This analysis phase transforms your hands-on experience into actionable insights for framework selection decisions. As you work through these comparisons, focus on patterns that would apply to more complex agent systems—the differences that matter at scale.

**Task 4: Comparative Analysis (30 minutes)**

**Objective**: Analyze and document differences across implementations

**Comparison Dimensions:**

**1. Code Complexity**

Run line count analysis:
```bash
python analyze_complexity.py

# Expected output:
# LangChain: ~120 lines
# LangGraph: ~180 lines (including state schema)
# AutoGen: ~110 lines
```

**Questions to Answer:**
- Which framework requires the most boilerplate?
- Which is most concise for this use case?
- Which is easiest to understand for a new developer?

**2. Feature Comparison**

| Feature | LangChain | LangGraph | AutoGen |
|---------|-----------|-----------|---------|
| Tool Calling | Built-in | Manual in nodes | Function calling |
| State Management | Chat history | Explicit state graph | Agent messages |
| Multi-Turn Conversations | Native | Stateful graph | Native |
| Debugging Visibility | Intermediate steps | Node-level tracing | Chat logs |
| Extensibility | High (many tools) | Very High (custom nodes) | Medium (function-based) |

**3. Performance Benchmarking**

```bash
# Run benchmark suite
python benchmark.py --iterations 10

# Metrics measured:
# - Average latency per query
# - P95 latency
# - Token usage
# - API calls made
# - Memory usage
```

**Expected Results:**
| Metric | LangChain | LangGraph | AutoGen |
|--------|-----------|-----------|---------|
| Avg Latency | ~2.1s | ~2.3s | ~2.0s |
| P95 Latency | ~2.8s | ~3.1s | ~2.6s |
| Tokens Used | ~500 | ~600 | ~450 |
| Memory (MB) | ~120 | ~150 | ~110 |

**4. Error Handling**

Test error cases:
```bash
# Invalid city
python main.py --framework langchain --query "Weather in Atlantis?"

# API failure (simulate by setting wrong API key)
WEATHER_API_KEY=invalid python main.py --framework langgraph --query "Weather in NYC?"

# Malformed query
python main.py --framework autogen --query "asdfghjkl"
```

**Document:**
- How does each framework handle errors?
- Are error messages helpful for debugging?
- Does the agent gracefully degrade?

**5. Developer Experience**

**Subjective Assessment:**
Rate each framework (1-5 stars):

| Aspect | LangChain | LangGraph | AutoGen |
|--------|-----------|-----------|---------|
| Ease of Setup | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Documentation Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Debugging Experience | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Learning Curve | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

---

### Part 5: Testing and Validation

With implementations complete and comparisons documented, you need to verify that all three agents actually work correctly and meet performance requirements. This validation phase ensures your comparative analysis is based on properly functioning implementations rather than buggy code that misrepresents framework capabilities.

**Comprehensive Test Suite**

```bash
# Run all tests
python -m pytest tests/ -v --cov=. --cov-report=html

# Expected coverage: >85%
```

**Test Categories:**

**1. Unit Tests**
- Weather tool functionality
- Error handling
- Input validation

**2. Integration Tests**
- End-to-end query processing
- Multi-turn conversations
- Tool calling accuracy

**3. Comparison Tests**
- All frameworks produce correct answers
- Response times within acceptable range
- Resource usage within limits

**Validation Checklist**

**Functional Requirements:**
- [ ] All three agents answer weather queries correctly
- [ ] Multi-turn conversations work
- [ ] Invalid cities handled gracefully
- [ ] API errors don't crash agents
- [ ] Responses are natural and helpful

**Performance Requirements:**
- [ ] P95 latency <3 seconds
- [ ] Memory usage <200MB per agent
- [ ] Token efficiency (avoid redundant calls)

**Code Quality:**
- [ ] All implementations follow framework best practices
- [ ] Code is well-documented
- [ ] Error handling is comprehensive
- [ ] Tests pass with >85% coverage

---

## Solution Guide

### Solution Structure

Complete solution available in `solutions/` directory:

```
solutions/
├── langchain_agent/
│   └── agent.py           # Complete LangChain implementation
├── langgraph_agent/
│   ├── agent.py           # Complete LangGraph implementation
│   └── state.py           # State schema
├── autogen_agent/
│   └── agent.py           # Complete AutoGen implementation
├── COMPARISON_REPORT.md   # Detailed framework comparison
└── IMPLEMENTATION_NOTES.md # Design decisions and trade-offs
```

### Key Implementation Decisions

**1. Tool Design Choice**: Shared vs Framework-Specific
- **Decision**: Use shared `WeatherTool` class for all frameworks
- **Rationale**: Isolates API concerns from framework logic, enables fair comparison
- **Trade-offs**: May not leverage framework-specific tool optimizations

**2. State Management**: Stateful vs Stateless
- **LangChain**: Stateful (chat_history list)
- **LangGraph**: Stateless graph (state passed through nodes)
- **AutoGen**: Stateful (agent message history)
- **Implications**: LangGraph more scalable for complex workflows, LangChain/AutoGen simpler for linear conversations

**3. Error Handling Strategy**
- **Decision**: Try-except blocks with descriptive error messages
- **Implementation**: Each framework returns user-friendly error strings
- **Extension**: Could add error recovery (retry logic, fallback responses)

---

## Learning Outcomes

### Self-Assessment

After completing this lab, you should be able to:

**Technical Skills:**
- [ ] Implement agents in LangChain, LangGraph, and AutoGen independently
- [ ] Configure tool/function calling in each framework
- [ ] Manage conversation state across multiple frameworks
- [ ] Benchmark and profile agent implementations
- [ ] Debug framework-specific issues

**Conceptual Understanding:**
- [ ] Explain trade-offs between different agent frameworks
- [ ] Justify framework selection for specific use cases
- [ ] Understand tool integration patterns
- [ ] Compare state management approaches
- [ ] Analyze performance characteristics

**Decision-Making Skills:**
- [ ] Choose appropriate framework for new projects
- [ ] Evaluate framework capabilities against requirements
- [ ] Balance complexity vs flexibility in design
- [ ] Assess long-term maintainability

### Confidence Rating

Rate your confidence (1-5 stars):

| Skill | Confidence | Need More Practice? |
|-------|------------|---------------------|
| LangChain Development | ⭐⭐⭐⭐⭐ | Yes / No |
| LangGraph Development | ⭐⭐⭐⭐⭐ | Yes / No |
| AutoGen Development | ⭐⭐⭐⭐⭐ | Yes / No |
| Framework Selection | ⭐⭐⭐⭐⭐ | Yes / No |

**Target**: All skills at 4+ stars

---

## Next Steps

### If You Succeeded:
1. ✓ Mark lab complete in progress tracker
2. ✓ Try the extension challenge (add CrewAI implementation)
3. ✓ Move to Lab 12: Comprehensive Evaluation Pipeline

### If You Struggled:
1. Review Chapter 2.1-2.6 in depth
2. Study one framework's solution carefully
3. Implement that framework again from scratch
4. Seek help in community forum or office hours

### Going Deeper:
- **Extension Challenge**: Add a fourth framework (CrewAI or LlamaIndex)
- **Real-World Extension**: Build a multi-tool agent (weather + news + calendar)
- **Advanced Topic**: Implement streaming responses in all frameworks

---

## Troubleshooting

### Common Issues

When LangChain's AgentExecutor fails with "No tool found" errors during invocation, this typically indicates either the tool wasn't properly registered in the tool list or there's a name mismatch between what the LLM's function call specifies and what tools are available. Verify that your tool list is correctly passed to `create_openai_tools_agent` and that the tool name matches exactly what the LLM expects. Adding logging to confirm tool registration during initialization can help debug these issues before they occur during execution.

If LangGraph state appears stale or unchanged between nodes—for example, tool results aren't visible in the response node—this signals that node functions aren't properly returning updated state dictionaries. Each node function must return a modified `AgentState` dict that includes all changes; LangGraph doesn't mutate state in place. Add print statements in your node functions to verify that state updates are actually being created and returned, not just modified locally and lost.

AutoGen agents sometimes make redundant API calls to the weather service for the same city within a single conversation. This happens because the agent doesn't automatically track previous function calls in its reasoning context. Improve your system message to explicitly instruct the agent to reference existing function results when available, or enhance the conversation context to include function call history. This prevents wasted API calls and improves response latency.

When the Weather API returns 401 Unauthorized errors across all frameworks, this indicates an authentication problem with your API key rather than a framework-specific issue. Verify the `WEATHER_API_KEY` in your .env file is correct and doesn't contain extra spaces or quotes. Check your account status at weatherapi.com to confirm the key is active and hasn't exceeded rate limits. You can test the API key directly using: `curl "http://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q=London"` to isolate whether this is a configuration issue or an actual API problem.

### Getting Help

- **Check**: Solution notes in `solutions/IMPLEMENTATION_NOTES.md`
- **Community**: Course discussion forum
- **Documentation**:
  - LangChain: https://python.langchain.com/docs/modules/agents/
  - LangGraph: https://langchain-ai.github.io/langgraph/
  - AutoGen: https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat
- **Office Hours**: Check course schedule

---

## Lab Completion Checklist

- [ ] Environment set up correctly
- [ ] LangChain agent implemented and tested
- [ ] LangGraph agent implemented and tested
- [ ] AutoGen agent implemented and tested
- [ ] Comparison analysis completed
- [ ] All tests passing (>85% coverage)
- [ ] Performance benchmarks within targets
- [ ] Self-assessment completed
- [ ] Confidence rating 4+ stars
- [ ] Ready for next lab

### What You've Accomplished

You've now completed a comprehensive multi-framework comparison lab that few developers ever undertake. Rather than choosing frameworks based on popularity or documentation, you have direct implementation experience comparing LangChain's rapid prototyping capabilities, LangGraph's explicit state management, and AutoGen's multi-agent architecture. This hands-on evaluation—measuring code complexity, performance characteristics, debugging experience, and developer productivity—gives you the practical judgment to make informed framework decisions for production systems.

More importantly, you've developed a methodology for evaluating agent frameworks that extends beyond these three. When new frameworks emerge or you encounter specialized frameworks like CrewAI or LlamaIndex, you now have a structured approach: implement the same functionality, measure code complexity and performance, assess debugging visibility, and evaluate long-term maintainability. This skill—evidence-based framework evaluation—will serve you throughout your career as agent development tools continue to evolve.

---

## Assessment Rubric (100 points)

**Implementation (60 points)**
- LangChain agent works correctly (20 points)
- LangGraph agent works correctly (20 points)
- AutoGen agent works correctly (20 points)

**Analysis (25 points)**
- Code complexity comparison (5 points)
- Feature comparison table completed (5 points)
- Performance benchmarking (10 points)
- Developer experience assessment (5 points)

**Code Quality (15 points)**
- Follows framework best practices (5 points)
- Proper error handling (5 points)
- Code documentation (5 points)

**Passing Grade**: 70/100 points
