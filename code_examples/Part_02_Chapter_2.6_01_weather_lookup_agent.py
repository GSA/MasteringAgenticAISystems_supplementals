import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Define the weather tool schema
weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a specified city. Use this tool when users ask about weather, temperature, or conditions. Returns temperature in specified units (celsius or fahrenheit), conditions description, humidity percentage, and wind speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state or country in format 'City, State' or 'City, Country' (e.g., 'San Francisco, CA')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit for response",
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    }
}

# Implement the actual weather lookup function
def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    Mock weather lookup function. In production, this would call
    an external weather API like OpenWeather or Weather.com.
    """
    # Mock weather data for demonstration
    mock_data = {
        "San Francisco, CA": {
            "celsius": {"temp": 18, "conditions": "Partly cloudy", "humidity": 65, "wind": 12},
            "fahrenheit": {"temp": 64, "conditions": "Partly cloudy", "humidity": 65, "wind": 12}
        },
        "New York, NY": {
            "celsius": {"temp": 22, "conditions": "Clear", "humidity": 58, "wind": 8},
            "fahrenheit": {"temp": 72, "conditions": "Clear", "humidity": 58, "wind": 8}
        },
        "London, UK": {
            "celsius": {"temp": 15, "conditions": "Rainy", "humidity": 78, "wind": 15},
            "fahrenheit": {"temp": 59, "conditions": "Rainy", "humidity": 78, "wind": 15}
        }
    }

    # Lookup weather data (default to San Francisco if location not in mock data)
    location_data = mock_data.get(location, mock_data["San Francisco, CA"])
    weather_data = location_data.get(unit, location_data["celsius"])

    return {
        "location": location,
        "temperature": weather_data["temp"],
        "unit": unit,
        "conditions": weather_data["conditions"],
        "humidity": weather_data["humidity"],
        "wind_speed": weather_data["wind"]
    }

# Agent conversation with function calling
def run_weather_agent(user_query: str):
    """
    Complete agent workflow demonstrating function calling:
    1. User submits query
    2. LLM decides if tool call is needed
    3. Application executes tool and returns result
    4. LLM generates final response incorporating tool result
    """
    print(f"User: {user_query}\n")

    # Step 1: Send query to LLM with tool schema
    messages = [
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=[weather_tool_schema],
        tool_choice="auto"  # Let LLM decide if tool is needed
    )

    response_message = response.choices[0].message

    # Step 2: Check if LLM wants to call a function
    if response_message.tool_calls:
        # LLM decided a function call is needed
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"LLM Decision: Call function '{function_name}' with args: {function_args}\n")

        # Step 3: Execute the function
        if function_name == "get_weather":
            function_result = get_weather(**function_args)
            print(f"Function Result: {json.dumps(function_result, indent=2)}\n")

            # Step 4: Send function result back to LLM for final response
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_result)
            })

            final_response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )

            final_answer = final_response.choices[0].message.content
            print(f"Agent: {final_answer}\n")

    else:
        # LLM responded directly without function call
        print(f"Agent: {response_message.content}\n")

# Test the agent with different queries
if __name__ == "__main__":
    # Query that triggers function call
    run_weather_agent("What's the weather like in San Francisco?")

    # Query with explicit unit preference
    run_weather_agent("Tell me the temperature in New York in fahrenheit")

    # Query that shouldn't trigger function call
    run_weather_agent("What is the capital of France?")
