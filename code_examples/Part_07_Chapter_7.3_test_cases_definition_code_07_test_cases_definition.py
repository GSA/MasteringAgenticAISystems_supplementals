from nemo.agent_toolkit import Evaluator

# Define test cases with expected ground truth
test_cases = [
    {
        "query": "What is quantum computing?",
        "expected": "ground_truth_quantum.txt",  # File containing validated answer
        "metrics": ["accuracy", "latency", "cost"]
    },
    {
        "query": "Explain transformer architecture for NLP",
        "expected": "ground_truth_transformers.txt",
        "metrics": ["accuracy", "latency", "cost"]
    },
    {
        "query": "Summarize recent advances in diffusion models",
        "expected": "ground_truth_diffusion.txt",
        "metrics": ["accuracy", "latency", "cost"]
    }
]
