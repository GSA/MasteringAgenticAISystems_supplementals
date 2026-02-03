from sklearn.metrics import precision_score, recall_score, f1_score

class FilterMetrics:
    """Evaluate output filter performance"""

    def __init__(self):
        self.predictions = []
        self.ground_truth = []

    def add_prediction(self, predicted_harmful: bool, actually_harmful: bool):
        """Add a prediction for evaluation"""
        self.predictions.append(1 if predicted_harmful else 0)
        self.ground_truth.append(1 if actually_harmful else 0)

    def calculate_metrics(self) -> dict:
        """Calculate precision, recall, F1"""
        precision = precision_score(self.ground_truth, self.predictions)
        recall = recall_score(self.ground_truth, self.predictions)
        f1 = f1_score(self.ground_truth, self.predictions)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': 1 - precision,
            'false_negative_rate': 1 - recall
        }

# Usage example
metrics = FilterMetrics()
# Ground truth: [harmful, safe, harmful, safe, safe]
# Predictions:   [harmful, harmful, harmful, safe, safe]
#                 TP       FP       TP       TN    TN
metrics.add_prediction(True, True)    # True Positive
metrics.add_prediction(True, False)   # False Positive
metrics.add_prediction(True, True)    # True Positive
metrics.add_prediction(False, False)  # True Negative
metrics.add_prediction(False, False)  # True Negative

results = metrics.calculate_metrics()
print(f"Precision: {results['precision']:.2%}")  # 66.67% (2 TP, 1 FP)
print(f"Recall: {results['recall']:.2%}")        # 100% (2 TP, 0 FN)
print(f"F1: {results['f1']:.2%}")                # 80% (harmonic mean)
