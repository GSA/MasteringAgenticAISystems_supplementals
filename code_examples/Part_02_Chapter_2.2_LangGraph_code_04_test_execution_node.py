import subprocess
import tempfile
import os
from pathlib import Path

def run_tests_node(state: DebugAgentState) -> DebugAgentState:
    """Execute tests against generated code and capture results.

    This node runs pytest in an isolated environment and parses
    the output to determine pass/fail status.
    """
    # Create temporary directory for test execution
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write generated code to file
        code_file = tmpdir_path / "solution.py"
        code_file.write_text(state["generated_code"])

        # Write test file (in real implementation, this would come from state)
        test_file = tmpdir_path / "test_solution.py"
        test_content = f"""import pytest
                        from solution import *

                        # Test cases derived from requirements
                        {state.get("test_suite", "# No tests provided")}
                        """
        test_file.write_text(test_content)

        # Run pytest with minimal output
        result = subprocess.run(
            ["pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )

        # Parse pytest output
        test_output = result.stdout + "\n" + result.stderr

        # Count passed/failed from pytest summary line
        # Format: "X passed, Y failed in Z.ZZs"
        passed = 0
        failed = 0
        for line in test_output.split("\n"):
            if "passed" in line.lower():
                try:
                    passed = int(line.split("passed")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass
            if "failed" in line.lower():
                try:
                    failed = int(line.split("failed")[0].strip().split()[-1])
                except (ValueError, IndexError):
                    pass

        # Determine status
        if failed == 0 and passed > 0:
            status = TestStatus.PASSED
        elif failed > 0:
            status = TestStatus.FAILED
        else:
            status = TestStatus.NOT_RUN

        return {
            "test_status": status,
            "test_output": test_output,
            "passed_tests": passed,
            "failed_tests": failed
        }
