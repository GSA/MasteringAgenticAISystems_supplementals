# Define initial state
initial_state = DebugAgentState(
    requirements="""Create a function `fibonacci(n: int) -> int` that returns the nth Fibonacci number.

    Requirements:
    - Handle n=0 returning 0
    - Handle n=1 returning 1
    - For n>1, return fib(n-1) + fib(n-2)
    - Raise ValueError for negative n
    """,
    test_suite="""
    def test_base_cases():
        assert fibonacci(0) == 0
        assert fibonacci(1) == 1

    def test_sequence():
        assert fibonacci(5) == 5
        assert fibonacci(10) == 55

    def test_negative():
        with pytest.raises(ValueError):
            fibonacci(-1)
    """,
    generated_code="",
    code_history=[],
    test_status=TestStatus.NOT_RUN,
    test_output="",
    passed_tests=0,
    failed_tests=0,
    error_analysis=None,
    error_patterns=[],
    iteration=0,
    max_iterations=3,
    is_complete=False,
    final_code=None
)

# Execute workflow
final_state = app.invoke(initial_state)

# Inspect results
if final_state["final_code"]:
    print(f"Success after {final_state['iteration']} iterations!")
    print(f"\nFinal Code:\n{final_state['final_code']}")
    print(f"\nTest Results: {final_state['passed_tests']} passed")
else:
    print(f"Failed to generate working code after {final_state['max_iterations']} iterations")
    print(f"\nLast error: {final_state['error_analysis']}")
