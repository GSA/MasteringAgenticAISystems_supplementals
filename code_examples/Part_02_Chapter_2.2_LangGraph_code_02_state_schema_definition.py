from typing import TypedDict, List, Optional
from enum import Enum

class TestStatus(Enum):
    """Test execution outcomes."""
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"

class DebugAgentState(TypedDict):
    """State for code debugging workflow.

    This state flows through all nodes, tracking the complete
    context needed for iterative code refinement.
    """
    # Input specification
    requirements: str              # What to build (function signature + behavior)

    # Code generation tracking
    generated_code: str            # Current code iteration
    code_history: List[str]        # Previous code attempts (for learning)

    # Test execution results
    test_status: TestStatus        # Current test outcome
    test_output: str               # Pytest stdout/stderr
    passed_tests: int              # Count of passing tests
    failed_tests: int              # Count of failing tests

    # Error analysis
    error_analysis: Optional[str]  # What went wrong (if tests failed)
    error_patterns: List[str]      # Recurring error types across iterations

    # Iteration control
    iteration: int                 # Current refinement cycle (0-based)
    max_iterations: int            # Prevent infinite loops

    # Completion tracking
    is_complete: bool              # Workflow finished (success or gave up)
    final_code: Optional[str]      # Successful code (if tests passed)
