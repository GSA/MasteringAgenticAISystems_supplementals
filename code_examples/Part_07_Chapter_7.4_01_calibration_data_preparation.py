# calibration_prep.py
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Representative production samples (500-1000 examples)
calibration_texts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci",
    "What are the benefits of serverless architecture?",
    # ... add 497+ more representative examples
]

calibration_data = []
for text in calibration_texts:
    tokens = tokenizer(text, return_tensors='np', max_length=512, truncation=True)
    calibration_data.append(tokens['input_ids'])

np.save('calibration.npy', np.array(calibration_data))
