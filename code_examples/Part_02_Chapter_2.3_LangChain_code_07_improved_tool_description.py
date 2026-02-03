calculator_tool = Tool.from_function(
    func=calculate,
    name="calculator",
    description=(
        "Evaluate mathematical expressions involving arithmetic, "
        "algebra, trigonometry, and calculus. "
        "Use this when the user asks for numerical calculations, "
        "solving equations, or mathematical analysis. "
        "Input: A math expression as a string (e.g., '2 + 2', 'sin(pi/4)', 'solve x^2 - 4 = 0'). "
        "Returns: The numerical result or symbolic solution. "
        "Examples of appropriate queries: 'What is 15% of 200?', "
        "'Solve the quadratic equation x^2 - 5x + 6 = 0', "
        "'What is the derivative of x^3?'"
    )
)
