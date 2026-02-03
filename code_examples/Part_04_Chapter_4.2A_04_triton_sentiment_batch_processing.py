def analyze_sentiment_batch(texts):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    results = []

    # Process in chunks of 256 (matching max batch size)
    for i in range(0, len(texts), 256):
        chunk = texts[i:i+256]
        inputs_batch = tokenizer(chunk, return_tensors="np", padding="max_length",
                                max_length=128, truncation=True)

        input_ids = httpclient.InferInput("input_ids", inputs_batch["input_ids"].shape, "INT64")
        input_ids.set_data_from_numpy(inputs_batch["input_ids"])

        attention_mask = httpclient.InferInput("attention_mask",
                                               inputs_batch["attention_mask"].shape, "INT64")
        attention_mask.set_data_from_numpy(inputs_batch["attention_mask"])

        # Request batch model
        response = client.infer("sentiment_batch", [input_ids, attention_mask])
        logits = response.as_numpy("logits")

        # Process batch results
        results.extend(["positive" if l[1] > l[0] else "negative" for l in logits])

    return results
