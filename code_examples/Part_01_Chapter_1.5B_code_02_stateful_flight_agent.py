# ====================================================================
# STATEFUL AGENT (CORRECT)
# ====================================================================

from typing import TypedDict, List, Annotated
from operator import add

class FlightSearchState(TypedDict):
    """
    Explicit state schema for the flight booking agent.

    This TypedDict defines exactly what information persists
    across the agent's execution.
    """
    user_query: str
    itinerary: List[List[str]]  # [[origin, destination, date], ...]
    flight_results: Annotated[List[dict], add]  # Accumulate results
    pricing_comparisons: List[dict]
    current_step: str  # Track where we are in the workflow
    error_count: int


class StatefulFlightAgent:
    """
    Proper implementation with explicit state management.

    BENEFITS:
    - State persists across all operations
    - Can resume after failures
    - Enables parallel tool calls (multiple flight searches)
    - Provides full audit trail
    """

    def __init__(self, llm):
        self.llm = llm
        self.state: FlightSearchState = {
            "user_query": "",
            "itinerary": [],
            "flight_results": [],
            "pricing_comparisons": [],
            "current_step": "init",
            "error_count": 0
        }

    def extract_itinerary(self, state: FlightSearchState) -> FlightSearchState:
        """
        Extract itinerary and UPDATE STATE.
        Returns new state - original is preserved (immutable update).
        """
        # We need to parse the query and store results in a way that
        # subsequent steps can access them. State provides that persistence.

        extraction_prompt = f"""
        Extract cities and dates from: {state['user_query']}
        Format: {{"legs": [["City1", "City2", "Date"], ...]}}
        """

        result = self.llm.invoke(extraction_prompt)

        # Create new state with extracted itinerary
        new_state = state.copy()
        new_state["itinerary"] = result["legs"]
        new_state["current_step"] = "itinerary_extracted"

        # Checkpoint: Validate extraction
        assert len(new_state["itinerary"]) > 0, "Must have at least one flight leg"

        return new_state

    def search_flights(self, state: FlightSearchState) -> FlightSearchState:
        """
        Search flights for each leg using state data.
        This step can access the itinerary from previous step!
        """
        new_state = state.copy()

        for leg in state["itinerary"]:
            origin, destination, date = leg

            # Call flight search API (simulated)
            search_result = self._search_api(origin, destination, date)

            # Accumulate results in state
            # The Annotated[List[dict], add] type hint tells the system
            # to APPEND rather than REPLACE
            new_state["flight_results"].append(search_result)

        new_state["current_step"] = "flights_searched"
        return new_state

    def compare_pricing(self, state: FlightSearchState) -> FlightSearchState:
        """
        Compare pricing across airlines using accumulated flight results.
        """
        new_state = state.copy()

        # Access all flight results from previous step
        all_flights = state["flight_results"]

        # Generate comparisons
        comparisons = []
        for flight_options in all_flights:
            comparison = {
                "cheapest": min(flight_options, key=lambda x: x["price"]),
                "fastest": min(flight_options, key=lambda x: x["duration"]),
                "best_value": self._calculate_value_score(flight_options)
            }
            comparisons.append(comparison)

        new_state["pricing_comparisons"] = comparisons
        new_state["current_step"] = "pricing_compared"
        return new_state

    def generate_recommendation(self, state: FlightSearchState) -> str:
        """
        Final step: synthesize all state data into recommendation.
        """
        # Access complete workflow state
        itinerary = state["itinerary"]
        comparisons = state["pricing_comparisons"]

        recommendation = f"""
        Flight Recommendations:

        Your itinerary: {len(itinerary)} legs
        """

        for i, (leg, comparison) in enumerate(zip(itinerary, comparisons)):
            recommendation += f"""

            Leg {i+1}: {leg[0]} → {leg[1]} on {leg[2]}
            - Cheapest: {comparison['cheapest']['airline']} (${comparison['cheapest']['price']})
            - Fastest: {comparison['fastest']['airline']} ({comparison['fastest']['duration']}h)
            - Best Value: {comparison['best_value']['airline']}
            """

        return recommendation

    def _search_api(self, origin, destination, date):
        """Simulate flight search API call."""
        # In reality, this calls an external API
        return [
            {"airline": "Delta", "price": 350, "duration": 3.5},
            {"airline": "United", "price": 320, "duration": 4.0},
            {"airline": "Southwest", "price": 280, "duration": 5.0}
        ]

    def _calculate_value_score(self, flights):
        """Calculate value score balancing price and duration."""
        scored = []
        for flight in flights:
            # Value = inverse of (normalized_price + normalized_duration)
            score = 1 / (flight["price"]/1000 + flight["duration"]/10)
            scored.append({**flight, "value_score": score})
        return max(scored, key=lambda x: x["value_score"])
