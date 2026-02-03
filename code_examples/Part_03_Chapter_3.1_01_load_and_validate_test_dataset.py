import json
from pathlib import Path
from typing import List, Dict, Any

def load_test_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load test dataset with validation"""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        test_cases = json.load(f)

    # Validate dataset structure
    required_fields = {'query', 'expected_answer', 'intent', 'difficulty'}
    for idx, case in enumerate(test_cases):
        missing = required_fields - set(case.keys())
        if missing:
            raise ValueError(
                f"Test case {idx} missing required fields: {missing}"
            )

    print(f"Loaded {len(test_cases)} test cases from {dataset_path}")
    return test_cases
