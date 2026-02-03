# Adjust K based on running acceptance rate
class AdaptiveSpeculativeDecoding:
    def __init__(self, initial_k=4):
        self.k = initial_k
        self.acceptance_history = []

    def update_k(self, accepted_tokens):
        self.acceptance_history.append(accepted_tokens / self.k)
        avg_acceptance = sum(self.acceptance_history[-100:]) / 100

        if avg_acceptance > 0.8:
            self.k = min(self.k + 1, 8)  # Increase K
        elif avg_acceptance < 0.4:
            self.k = max(self.k - 1, 2)  # Decrease K
