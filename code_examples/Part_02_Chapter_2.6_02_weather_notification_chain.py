import json
from typing import Dict, List, Any
from openai import OpenAI

client = OpenAI()

# Tool Schema 1: Weather lookup
weather_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a location. Returns temperature, conditions, alerts.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state (e.g., 'Boston, MA')"
                }
            },
            "required": ["location"]
        }
    }
}

# Tool Schema 2: User preference lookup
preferences_schema = {
    "type": "function",
    "function": {
        "name": "get_user_preferences",
        "description": "Retrieve user notification preferences and alert criteria. Requires weather data to determine if user's conditions are met.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User identifier"
                },
                "current_conditions": {
                    "type": "object",
                    "description": "Current weather conditions to check against preferences"
                }
            },
            "required": ["user_id", "current_conditions"]
        }
    }
}

# Tool Schema 3: Message formatting
format_message_schema = {
    "type": "function",
    "function": {
        "name": "format_alert_message",
        "description": "Format personalized weather alert message based on conditions and user preferences.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "description": "User's name for personalization"
                },
                "weather_data": {
                    "type": "object",
                    "description": "Weather conditions triggering the alert"
                },
                "alert_reason": {
                    "type": "string",
                    "description": "Why this alert was triggered"
                }
            },
            "required": ["user_name", "weather_data", "alert_reason"]
        }
    }
}

# Tool Schema 4: Notification delivery
send_notification_schema = {
    "type": "function",
    "function": {
        "name": "send_notification",
        "description": "Send notification via user's preferred channel (email, SMS, push).",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["email", "sms", "push"],
                    "description": "Notification channel"
                },
                "recipient": {
                    "type": "string",
                    "description": "Recipient address (email, phone, device_id)"
                },
                "message": {
                    "type": "string",
                    "description": "Formatted message content"
                }
            },
            "required": ["channel", "recipient", "message"]
        }
    }
}

# Tool Implementations
def get_weather(location: str) -> Dict[str, Any]:
    """Step 1: Fetch current weather conditions"""
    mock_weather = {
        "location": location,
        "temperature": 35,  # Fahrenheit
        "conditions": "Heavy Rain",
        "wind_speed": 25,
        "alerts": ["Flood Warning", "High Wind Advisory"],
        "humidity": 85
    }
    print(f"[Tool 1] Weather lookup for {location}: {mock_weather}")
    return mock_weather

def get_user_preferences(user_id: str, current_conditions: Dict) -> Dict[str, Any]:
    """Step 2: Check user preferences against current conditions"""
    mock_preferences = {
        "user_id": user_id,
        "name": "Alice Johnson",
        "alert_triggers": {
            "temperature_below": 40,
            "severe_weather": True,
            "high_wind": 20
        },
        "notification_channel": "email",
        "email": "alice@example.com"
    }

    # Check if conditions meet user's alert criteria
    triggers_met = []
    if current_conditions.get("temperature", 100) < mock_preferences["alert_triggers"]["temperature_below"]:
        triggers_met.append("temperature below threshold")
    if current_conditions.get("alerts") and mock_preferences["alert_triggers"]["severe_weather"]:
        triggers_met.append("severe weather alerts")
    if current_conditions.get("wind_speed", 0) > mock_preferences["alert_triggers"]["high_wind"]:
        triggers_met.append("high wind conditions")

    result = {
        **mock_preferences,
        "should_alert": len(triggers_met) > 0,
        "triggers_met": triggers_met
    }
    print(f"[Tool 2] User preferences check: {result}")
    return result

def format_alert_message(user_name: str, weather_data: Dict, alert_reason: str) -> str:
    """Step 3: Format personalized message"""
    message = f"""Weather Alert for {user_name}

Current Conditions:
- Temperature: {weather_data['temperature']}°F
- Conditions: {weather_data['conditions']}
- Wind Speed: {weather_data['wind_speed']} mph

Alert Reason: {alert_reason}

Active Alerts: {', '.join(weather_data.get('alerts', []))}

Stay safe and take appropriate precautions.
"""
    print(f"[Tool 3] Formatted message: {message[:100]}...")
    return message

def send_notification(channel: str, recipient: str, message: str) -> Dict[str, Any]:
    """Step 4: Send notification via preferred channel"""
    result = {
        "status": "sent",
        "channel": channel,
        "recipient": recipient,
        "message_length": len(message),
        "sent_at": "2024-11-09T14:30:00Z"
    }
    print(f"[Tool 4] Notification sent via {channel} to {recipient}")
    return result

# Orchestrate the multi-tool chain
def run_weather_alert_chain(user_id: str, location: str):
    """
    Complete workflow demonstrating tool chaining:
    1. Get weather conditions
    2. Check user preferences against conditions
    3. Format personalized alert message
    4. Send via user's preferred channel
    """
    print(f"\n=== Starting Weather Alert Chain ===")
    print(f"User: {user_id}, Location: {location}\n")

    # Define all available tools
    tools = [weather_schema, preferences_schema, format_message_schema, send_notification_schema]

    # Initialize conversation
    messages = [{
        "role": "system",
        "content": "You are a weather alert assistant. Check weather conditions, verify if they meet user alert criteria, format appropriate messages, and send notifications. Always complete the full workflow in sequence."
    }, {
        "role": "user",
        "content": f"Check weather for {location} and send alert to user {user_id} if their criteria are met."
    }]

    # Execute tool chain (max 10 steps to prevent infinite loops)
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Check if LLM wants to call a function
        if response_message.tool_calls:
            # Append LLM's response to maintain state
            messages.append(response_message)

            # Execute each tool call (could be parallel)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"LLM Decision: Call {function_name}")
                print(f"Arguments: {json.dumps(function_args, indent=2)}")

                # Route to appropriate function
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "get_user_preferences":
                    result = get_user_preferences(**function_args)
                elif function_name == "format_alert_message":
                    result = format_alert_message(**function_args)
                elif function_name == "send_notification":
                    result = send_notification(**function_args)
                else:
                    result = {"error": f"Unknown function: {function_name}"}

                # Add function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })
        else:
            # No more tool calls - workflow complete
            final_response = response_message.content
            print(f"\n=== Workflow Complete ===")
            print(f"Agent Summary: {final_response}")
            break
    else:
        print("\n=== Workflow exceeded maximum steps ===")

# Test the multi-tool chain
if __name__ == "__main__":
    run_weather_alert_chain(
        user_id="user_123",
        location="Boston, MA"
    )
