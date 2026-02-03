import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("fine-tuned-sentiment")
model.eval()

# Create example input for tracing
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
example_text = "This product exceeded my expectations"
inputs = tokenizer(example_text, return_tensors="pt", padding="max_length", max_length=128)

# Trace model to TorchScript
traced_model = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))
traced_model.save("model.pt")
