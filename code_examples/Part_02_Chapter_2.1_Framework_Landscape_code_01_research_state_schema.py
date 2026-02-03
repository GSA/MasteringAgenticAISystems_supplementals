from typing import TypedDict, List, Annotated
from operator import add

class ResearchState(TypedDict):
    query: str                              # User's research question
    papers: Annotated[List[Paper], add]     # Accumulate papers across searches
    extractions: dict                       # Map paper_id -> key findings
    synthesis_history: List[str]            # Track synthesis iterations
    current_summary: str                    # Latest summary version
    user_feedback: str                      # Refinement guidance
    iteration_count: int                    # Prevent infinite refinement
