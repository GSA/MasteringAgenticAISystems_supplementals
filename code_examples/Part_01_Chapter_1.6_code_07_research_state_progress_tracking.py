class ResearchState(TypedDict):
    research_question: str
    sub_questions: List[str]  # All sub-questions from decomposition
    completed_questions: List[str]  # Sub-questions already searched
    search_results: List[dict]  # Accumulated results
    current_question: str  # Sub-question being processed

def get_next_question(state: ResearchState) -> str:
    """Determine which sub-question to search next."""
    for question in state["sub_questions"]:
        if question not in state["completed_questions"]:
            return question
    return None  # All questions complete

def search_handler(state: ResearchState) -> ResearchState:
    """Execute search and track completion."""
    current = state["current_question"]
    results = execute_search(current)

    return {
        **state,
        "search_results": state["search_results"] + [results],
        "completed_questions": state["completed_questions"] + [current],
        # Mark this question as complete to prevent repeating it
    }
