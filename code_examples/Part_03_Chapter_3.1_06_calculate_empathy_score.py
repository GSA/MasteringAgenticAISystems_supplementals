def calculate_empathy_score(response: str, query: str) -> float:
    """Score empathetic language in response"""
    empathy_phrases = [
        "i understand", "i apologize", "i'm sorry",
        "let me help", "i can assist", "i appreciate"
    ]

    response_lower = response.lower()

    # Count empathy signals
    empathy_count = sum(
        1 for phrase in empathy_phrases
        if phrase in response_lower
    )

    # Normalize to 0-1 range (cap at 2 phrases for max score)
    # Rationale: 2+ empathy phrases = genuine concern
    #           1 phrase = basic acknowledgment
    #           0 phrases = robotic/cold
    score = min(empathy_count / 2.0, 1.0)

    return score
