# Client receives stream of events showing reasoning steps
async for event in client.stream_log({"question": "What's your return policy?"}):
    if event["ops"][0]["path"] == "/logs/ChatOpenAI":
        # Tool call occurred
        print(f"Tool: {event['ops'][0]['value']}")
    elif event["ops"][0]["path"] == "/streamed_output/-":
        # Token generated
        print(event["ops"][0]["value"], end="")
