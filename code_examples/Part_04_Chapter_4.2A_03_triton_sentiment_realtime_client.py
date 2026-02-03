import tritonclient.http as httpclient
import numpy as np

def analyze_sentiment_realtime(text):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128)

    # Prepare Triton inputs
    input_ids = httpclient.InferInput("input_ids", inputs["input_ids"].shape, "INT64")
    input_ids.set_data_from_numpy(inputs["input_ids"])

    attention_mask = httpclient.InferInput("attention_mask", inputs["attention_mask"].shape, "INT64")
    attention_mask.set_data_from_numpy(inputs["attention_mask"])

    # Request real-time model
    response = client.infer("sentiment_realtime", [input_ids, attention_mask])
    logits = response.as_numpy("logits")

    return "positive" if logits[0][1] > logits[0][0] else "negative"
