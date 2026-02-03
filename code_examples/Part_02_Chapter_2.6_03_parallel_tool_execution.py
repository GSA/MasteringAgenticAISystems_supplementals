import asyncio
import json
from typing import List, Dict, Any
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Async tool implementations
async def get_weather_async(location: str) -> Dict[str, Any]:
    """Simulate async weather API call"""
    # Simulate network latency (2 seconds)
    await asyncio.sleep(2)
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny",
        "source": "weather_api"
    }

async def get_stock_price_async(symbol: str) -> Dict[str, Any]:
    """Simulate async stock price API call"""
    # Simulate network latency (1.5 seconds)
    await asyncio.sleep(1.5)
    return {
        "symbol": symbol,
        "price": 150.25,
        "change": +2.5,
        "source": "stock_api"
    }

async def get_news_async(topic: str) -> Dict[str, Any]:
    """Simulate async news API call"""
    # Simulate network latency (3 seconds)
    await asyncio.sleep(3)
    return {
        "topic": topic,
        "headlines": [
            "Breaking news about " + topic,
            "Latest developments in " + topic
        ],
        "source": "news_api"
    }

# Tool schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol (e.g., NVDA)"}
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get latest news headlines for a topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "News topic"}
                },
                "required": ["topic"]
            }
        }
    }
]

async def execute_tool_parallel(function_name: str, arguments: Dict) -> Any:
    """Route and execute tool calls"""
    if function_name == "get_weather":
        return await get_weather_async(**arguments)
    elif function_name == "get_stock_price":
        return await get_stock_price_async(**arguments)
    elif function_name == "get_news":
        return await get_news_async(**arguments)
    else:
        return {"error": f"Unknown function: {function_name}"}

async def parallel_agent_execution(user_query: str):
    """
    Demonstrate parallel tool execution:
    1. LLM generates multiple function calls
    2. Execute all concurrently using asyncio.gather
    3. Return all results to LLM for synthesis
    """
    import time
    start_time = time.time()

    print(f"User Query: {user_query}\n")

    # Get LLM response with tool calls
    messages = [{"role": "user", "content": user_query}]

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        print(f"LLM generated {len(response_message.tool_calls)} function calls\n")

        # Parse all function calls
        tool_tasks = []
        tool_call_info = []

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"  - {function_name}({arguments})")

            # Create async task for this tool call
            task = execute_tool_parallel(function_name, arguments)
            tool_tasks.append(task)
            tool_call_info.append((tool_call.id, function_name))

        # Execute all tools in parallel
        print(f"\nExecuting {len(tool_tasks)} tools in parallel...")
        parallel_start = time.time()

        results = await asyncio.gather(*tool_tasks)

        parallel_duration = time.time() - parallel_start
        print(f"Parallel execution completed in {parallel_duration:.2f} seconds\n")

        # Add all results to conversation
        messages.append(response_message)
        for (call_id, func_name), result in zip(tool_call_info, results):
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": json.dumps(result)
            })

        # Get final response from LLM
        final_response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        total_duration = time.time() - start_time
        print(f"Agent Response: {final_response.choices[0].message.content}")
        print(f"\nTotal execution time: {total_duration:.2f} seconds")
        print(f"Time saved vs sequential: ~{(2 + 1.5 + 3) - parallel_duration:.2f} seconds")
    else:
        print(f"Agent Response: {response_message.content}")

# Run the parallel execution example
if __name__ == "__main__":
    asyncio.run(parallel_agent_execution(
        "Give me the weather in San Francisco, NVIDIA stock price, and latest AI technology news"
    ))
