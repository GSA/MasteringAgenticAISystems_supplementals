# Evaluate classification accuracy
def compute_accuracy(outputs, ground_truth):
    correct = 0
    for output, expected in zip(outputs, ground_truth):
        predicted_label = extract_label(output.text)
        if predicted_label == expected:
            correct += 1
    return correct / len(outputs)

# Full validation set evaluation (10,000 samples)
validation_full = moderation_samples.select(range(10000))

accuracy_fp16 = compute_accuracy(
    llm_fp16.generate([s["text"] for s in validation_full], max_new_tokens=10),
    [s["label"] for s in validation_full]
)
accuracy_int8 = compute_accuracy(
    llm_int8.generate([s["text"] for s in validation_full], max_new_tokens=10),
    [s["label"] for s in validation_full]
)

print(f"FP16 Accuracy: {accuracy_fp16:.2%}")
print(f"INT8 Accuracy: {accuracy_int8:.2%}")
print(f"Accuracy Degradation: {accuracy_fp16 - accuracy_int8:.2%}")

# Results:
# FP16 Accuracy: 96.8%
# INT8 Accuracy: 95.9%
# Accuracy Degradation: 0.9%
