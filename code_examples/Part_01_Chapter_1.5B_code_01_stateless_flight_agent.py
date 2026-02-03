"""
Worked Example 1: Stateless vs Stateful Agent Comparison

Learning Focus:
- Understanding state management necessity
- Recognizing failure modes of stateless agents
- Implementing proper state persistence
"""

# ====================================================================
# STATELESS AGENT (PROBLEMATIC)
# ====================================================================

class StatelessFlightAgent:
    """
    Naive implementation without state management.

    PROBLEMS:
    - Loses context between tool calls
    - Cannot track which flight legs have been searched
    - Repeats searches or misses connections
    """

    def __init__(self, llm):
        self.llm = llm

    def process_query(self, user_query: str) -> str:
        """
        Process user query without maintaining state.
        Each tool call is independent - previous results are lost!
        """
        # Step 1: Extract itinerary
        # Notice: we extract cities from the query, but this information
        # exists only in a local variable. No persistent storage.

        extraction_prompt = f"""
        Extract cities and dates from: {user_query}
        Format: {{"legs": [["City1", "City2", "Date"], ...]}}
        """
        itinerary = self.llm.invoke(extraction_prompt)

        # Step 2: Search flights
        # PROBLEM: The 'itinerary' variable from Step 1 lives in this
        # function's local scope. If the agent needs to make multiple
        # tool calls or handle errors, this data is lost.

        search_prompt = f"""
        Search flights for: {itinerary}
        """
        flights = self.llm.invoke(search_prompt)

        # Step 3: Generate recommendation
        # PROBLEM: If this step fails, we have to restart from Step 1.
        # No way to resume from where we left off.

        return flights

    # Key observation: After this method returns, all intermediate data
    # (extracted itinerary, search results) is garbage collected.
    # The agent has no memory of previous steps!
