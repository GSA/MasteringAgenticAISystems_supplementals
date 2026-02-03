# Example: Layer 2 Output Classification
from transformers import pipeline
from typing import Tuple

class ToxicityClassifier:
    """Layer 2: Classify model outputs for toxicity"""

    def __init__(self):
        # Using pre-trained toxicity detection model
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1  # CPU
        )
        self.toxicity_threshold = 0.7

    def is_toxic(self, text: str) -> Tuple[bool, float]:
        """
        Classify text for toxicity.
        Returns (is_toxic, confidence_score)
        """
        result = self.classifier(text)[0]
        score = result['score']
        label = result['label']

        is_toxic = (label == 'toxic' and score > self.toxicity_threshold)
        return is_toxic, score

# Usage
classifier = ToxicityClassifier()
model_output = "I hate you and your stupid ideas"
is_toxic, score = classifier.is_toxic(model_output)
if is_toxic:
    print(f"Output blocked: Toxicity score {score:.2f}")
