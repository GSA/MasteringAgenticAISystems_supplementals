# Launching the NVIDIA Agent Intelligence Toolkit API Server and User Interface

**Source:** https://docs.nvidia.com/aiqtoolkit/latest/quick-start/launching-ui.html

## Overview

The NVIDIA Agent Intelligence (AIQ) toolkit provides a web-based user interface for interacting with running workflows, featuring chat history, HTTP/WebSocket APIs, and intermediate step management.

## Key UI Features
- **Chat history** - Track conversation context
- **HTTP API** - RESTful workflow interaction
- **WebSocket API** - Real-time bidirectional communication
- **Toggle intermediate steps** - Show/hide execution details
- **Override capabilities** - Manually intervene in workflow execution

## Prerequisites

Before launching, ensure you have:
- **Node.js v18+** installed for the web development server
- **AIQ toolkit** installed (`pip install nvidia-aiq`)
- A **configured workflow** ready to serve
- A **configuration file** with your workflow definition

## Launch Process

### Step 1: Initialize Git Submodules

First, ensure the UI submodule is checked out:
```bash
git submodule update --init --recursive
```

This pulls the web UI code into your project.

### Step 2: Start the AIQ Server

Launch the AIQ toolkit using your configuration file:
```bash
aiq serve --config_file=examples/simple_calculator/configs/config.yml
```

**Expected Output:**
```
Server running on http://localhost:8000
```

The server initializes and confirms it's ready to accept requests.

### Step 3: Verify Server Operation

Test the server with an HTTP request:
```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{"input_message": "Your query", "use_knowledge_base": true}'
```

Successful response confirms the server is operational.

### Step 4: Launch the Web UI

Navigate to the UI directory and start the development server:
```bash
cd external/aiqtoolkit-opensource-ui
npm install
npm run dev
```

**Expected Output:**
```
Local:   http://localhost:3000
```

If port 3000 is in use, the server will suggest port 3001.

### Step 5: Open in Browser

Open your web browser and navigate to:
- **Primary URL:** `http://localhost:3000/`
- **Alternative URL:** `http://localhost:3001/` (if port 3000 is in use)

## Configure Settings

Access settings via the bottom-left icon in the UI to specify:

### Endpoint Configuration
- **HTTP Endpoint:** Choose from available endpoints
- **Recommended:** `/chat/stream` for streaming responses
- **Alternative:** `/generate` for non-streaming

### Connection Settings
- **WebSocket Connection URL:** Specify WebSocket endpoint
- **HTTP Base URL:** Server address and port
- **Schema Type:** REST or GraphQL (if applicable)

### Preference Settings
- **Theme:** Light or dark mode
- **Language:** Interface language selection
- **Default Behavior:** Response streaming preferences

## Available Workflow Transactions

When the AIQ toolkit server is running, four workflow transactions are available via HTTP or WebSocket:

### 1. Generate (Non-Streaming)
- **Endpoint:** `/generate`
- **Protocol:** HTTP POST
- **Behavior:** Returns complete response in single call
- **Use Case:** Simple queries, batch processing

### 2. Generate (Streaming)
- **Endpoint:** `/generate/stream` or `/chat/stream`
- **Protocol:** HTTP POST with streaming response
- **Behavior:** Returns response tokens incrementally
- **Use Case:** Long-running tasks, real-time feedback

### 3. Chat (Non-Streaming)
- **Endpoint:** `/chat`
- **Protocol:** HTTP POST
- **Behavior:** Maintains conversation context, single response
- **Use Case:** Multi-turn conversation tracking

### 4. Chat (Streaming)
- **Endpoint:** `/chat/stream`
- **Protocol:** HTTP POST with streaming response
- **Behavior:** Maintains context with streaming responses
- **Use Case:** Real-time conversations, interactive agents

## HTTP Request Format

### Basic Request Structure
```json
{
  "input_message": "Your query or instruction",
  "use_knowledge_base": true,
  "temperature": 0.7,
  "max_tokens": 512,
  "metadata": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `input_message` | string | The user query or instruction |
| `use_knowledge_base` | boolean | Enable/disable knowledge base retrieval |
| `temperature` | float | Response creativity (0.0-1.0) |
| `max_tokens` | integer | Maximum response length |
| `metadata` | object | Custom tracking metadata |

## WebSocket Connection

### Connection Initiation
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Response chunk:', data);
};

ws.send(JSON.stringify({
  input_message: "Your query",
  use_knowledge_base: true
}));
```

### Advantages
- Real-time bidirectional communication
- Lower latency than repeated HTTP requests
- Persistent connection for multi-turn interactions
- Streaming support built-in

## Troubleshooting

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000  # for server
lsof -i :3000  # for UI

# Kill the process
kill -9 <PID>
```

### Node.js Version Issues
```bash
# Check Node.js version
node --version  # Should be v18+

# If outdated, install latest
nvm install 18
nvm use 18
```

### Submodule Issues
```bash
# If UI submodule is missing
git submodule update --init --recursive
```

### Connection Refused
- Ensure AIQ server is running on port 8000
- Check firewall settings
- Verify configuration file path is correct

## Next Steps

Once the UI is running:
1. **Access the Interface** - Open http://localhost:3000
2. **Configure Settings** - Set endpoint and connection parameters
3. **Test Workflows** - Run sample queries through the UI
4. **Monitor Execution** - View intermediate steps and performance metrics
5. **Iterate** - Refine workflows based on results

## Advanced Configuration

For production deployments:
- Use environment variables for configuration
- Implement authentication/authorization
- Configure SSL/TLS for security
- Set up load balancing for scalability
- Configure logging and monitoring

## Additional Resources

- **Configuration Guide:** NVIDIA AIQ Toolkit documentation
- **API Reference:** Complete endpoint specifications
- **Workflow Examples:** Sample configurations
- **Community Forum:** Support and best practices
