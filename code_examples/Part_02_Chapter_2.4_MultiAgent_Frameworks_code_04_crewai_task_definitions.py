# Task 1: Research the topic comprehensively
research_task = Task(
    description="""Research the topic: 'Retrieval-Augmented Generation (RAG)
    optimization techniques'. Focus on:
    - Recent advancements (2023-2024)
    - Key techniques: chunking strategies, embedding models, retrieval methods
    - Practical implementation challenges
    - Performance benchmarks and evaluation metrics

    Provide structured research notes with citations.""",
    expected_output="""Comprehensive research document (1500-2000 words) covering:
    1. Overview of RAG optimization landscape
    2. Detailed analysis of 5-7 key techniques with citations
    3. Comparison of approaches with benchmark results
    4. Practical implementation guidance
    All technical claims must be cited with paper references or documentation.""",
    agent=researcher
)

# Task 2: Write the blog post based on research
writing_task = Task(
    description="""Using the research findings, write an engaging technical blog
    post on RAG optimization techniques. The post should:
    - Open with a concrete problem scenario that RAG optimization solves
    - Explain 5-7 optimization techniques with practical code examples
    - Include before/after performance comparisons
    - Provide actionable implementation guidance
    - Close with a summary and recommended starting points

    Target length: 2500-3000 words. Include code examples in Python.""",
    expected_output="""Publication-ready blog post with:
    - Engaging opening (200 words)
    - Technical content sections (2000-2500 words)
    - 3-5 code examples with explanations
    - Performance comparison data
    - Actionable conclusion (200-300 words)
    Flesch-Kincaid readability: 50-60 (college level)""",
    agent=writer,
    context=[research_task]  # Writing depends on research output
)

# Task 3: Edit and refine the draft
editing_task = Task(
    description="""Review the draft blog post for technical accuracy, clarity,
    and publication quality. Specifically:
    - Validate all technical claims against research sources
    - Verify code examples are correct and executable
    - Check citations are properly formatted
    - Improve prose clarity and flow
    - Ensure consistent style and formatting

    Provide specific feedback and produce final edited version.""",
    expected_output="""Polished, publication-ready blog post that:
    - Maintains technical accuracy with verified claims
    - Includes tested, working code examples
    - Has proper citations for all claims
    - Reads smoothly with clear section transitions
    - Meets style guide requirements
    Plus: Editorial notes documenting changes and verification""",
    agent=editor,
    context=[writing_task]  # Editing depends on draft output
)
