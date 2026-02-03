"""
Code Example 1.6.1: State Machine Agent Implementation

Purpose: Demonstrate stateful agent orchestration using explicit state management

Concepts Demonstrated:
- State persistence across workflow steps
- TypedDict state schemas for type safety
- Immutable state updates (functional programming pattern)
- Error recovery and resumability

Prerequisites:
- Understanding of Python type hints
- Basic knowledge of state machines
- Familiarity with LangChain/LangGraph concepts

Author: NVIDIA Agentic AI Certification Course
Chapter: 1, Section: 1.6
Exam Skill: 1.6 - Apply logic trees, prompt chains, and stateful orchestration
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

from typing import TypedDict, List, Annotated, Literal, Optional
from operator import add
import logging
import time
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATE SCHEMA DEFINITION
# ============================================================================

class FlightSearchState(TypedDict):
    """
    State schema for multi-city flight booking agent.

    This TypedDict explicitly defines all information that persists
    across the agent's execution. Using TypedDict provides:
    - Type checking at development time
    - Clear documentation of state structure
    - IDE autocomplete support

    Attributes:
        user_query (str): Original user request
        itinerary (List[List[str]]): Flight legs as [[origin, dest, date], ...]
        flight_results (Annotated[List[dict], add]): Accumulated search results
            The 'add' operator tells LangGraph to append rather than replace
        pricing_comparisons (List[dict]): Price analysis for each leg
        current_step (str): Workflow tracking - where we are in execution
        error_count (int): Number of errors encountered (for retry logic)
        metadata (dict): Additional tracking info (timestamps, confidence, etc.)
    """
    user_query: str
    itinerary: List[List[str]]
    flight_results: Annotated[List[dict], add]  # Accumulate results
    pricing_comparisons: List[dict]
    current_step: Literal["init", "itinerary_extracted", "flights_searched",
                          "pricing_compared", "complete", "error"]
    error_count: int
    metadata: dict


# ============================================================================
# MOCK LLM AND SERVICES (Replace with real implementations)
# ============================================================================

@dataclass
class MockLLMResponse:
    """Simulated LLM response structure."""
    content: str
    response_metadata: dict

class MockLLM:
    """
    Mock LLM for demonstration purposes.

    In production, replace with:
    - ChatNVIDIA for NVIDIA NIM endpoints
    - ChatOpenAI for OpenAI
    - Other LangChain-compatible LLM classes
    """

    def invoke(self, prompt: str) -> MockLLMResponse:
        """Simulate LLM inference."""
        # Extract what kind of task this is from the prompt
        if "Extract cities and dates" in prompt:
            # Simulate itinerary extraction
            return MockLLMResponse(
                content=json.dumps({
                    "legs": [
                        ["New York", "Los Angeles", "2025-11-15"],
                        ["Los Angeles", "Tokyo", "2025-11-20"],
                        ["Tokyo", "New York", "2025-11-28"]
                    ]
                }),
                response_metadata={"confidence": 0.92}
            )
        elif "synthesize" in prompt.lower():
            # Simulate report generation
            return MockLLMResponse(
                content="""
                Flight Recommendations Report:

                Your itinerary includes 3 legs with optimized routing.
                Total estimated cost: $1,450 (best value option)
                Total travel time: 24.5 hours

                Detailed recommendations per leg are provided below.
                """,
                response_metadata={}
            )
        else:
            return MockLLMResponse(
                content="Default response",
                response_metadata={}
            )

# Initialize mock LLM
llm = MockLLM()


def perform_search_api(origin: str, destination: str, date: str) -> List[dict]:
    """
    Simulate flight search API call.

    In production, this would call:
    - Amadeus API
    - Skyscanner API
    - Direct airline APIs

    Args:
        origin: Departure city
        destination: Arrival city
        date: Flight date (YYYY-MM-DD)

    Returns:
        List of flight options with airline, price, duration
    """
    # Simulate API latency
    time.sleep(0.1)

    # Return mock results
    return [
        {
            "airline": "Delta",
            "price": 450,
            "duration": 5.5,
            "departure": "08:00",
            "arrival": "11:30",
            "flight_number": "DL1234"
        },
        {
            "airline": "United",
            "price": 420,
            "duration": 6.0,
            "departure": "10:00",
            "arrival": "14:00",
            "flight_number": "UA5678"
        },
        {
            "airline": "Southwest",
            "price": 380,
            "duration": 7.0,
            "departure": "12:00",
            "arrival": "17:00",
            "flight_number": "SW9012"
        }
    ]


# ============================================================================
# STATEFUL AGENT IMPLEMENTATION
# ============================================================================

class StatefulFlightAgent:
    """
    Production-grade stateful flight booking agent.

    Key Design Principles:
    1. Explicit State Management: All context stored in state object
    2. Immutable Updates: Functions return new state, don't modify input
    3. Resumability: Can restart from any step using saved state
    4. Observability: Current step always tracked for debugging
    5. Error Handling: Errors logged in state, don't lose progress

    This design pattern enables:
    - Fault tolerance (resume after crashes)
    - Parallel execution (state is the only shared data)
    - Debugging (full state history available)
    - Testing (deterministic given same initial state)
    """

    def __init__(self, llm):
        """
        Initialize the stateful agent.

        Args:
            llm: Language model instance (ChatNVIDIA, ChatOpenAI, etc.)
        """
        self.llm = llm
        self.state_history: List[FlightSearchState] = []

    def extract_itinerary(self, state: FlightSearchState) -> FlightSearchState:
        """
        Extract flight itinerary from user query.

        This is Step 1 of the workflow. It parses the natural language
        query into structured data (list of flight legs).

        Args:
            state: Current workflow state

        Returns:
            Updated state with extracted itinerary

        Raises:
            ValueError: If no valid itinerary can be extracted
        """
        logger.info("Step 1: Extracting itinerary from query")

        query = state["user_query"]

        # Construct extraction prompt
        extraction_prompt = f"""
        Extract cities and dates from this flight booking request:
        {query}

        Output format (JSON):
        {{
            "legs": [
                ["Origin City", "Destination City", "YYYY-MM-DD"],
                ...
            ]
        }}

        Rules:
        - Each leg must have origin, destination, and date
        - Use standard city names (e.g., "New York" not "NYC")
        - Dates in ISO format (YYYY-MM-DD)
        """

        # Call LLM
        response = self.llm.invoke(extraction_prompt)

        # Parse response
        try:
            result = json.loads(response.content)
            legs = result["legs"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Return error state
            new_state = state.copy()
            new_state["current_step"] = "error"
            new_state["error_count"] += 1
            new_state["metadata"]["last_error"] = str(e)
            return new_state

        # Validate extraction
        if not legs or len(legs) == 0:
            raise ValueError("No flight legs extracted from query")

        # Create new state with results
        new_state = state.copy()
        new_state["itinerary"] = legs
        new_state["current_step"] = "itinerary_extracted"
        new_state["metadata"]["extraction_time"] = time.time()
        new_state["metadata"]["extraction_confidence"] = response.response_metadata.get("confidence", 0.0)

        # Save state snapshot for debugging
        self.state_history.append(new_state)

        logger.info(f"✅ Extracted {len(legs)} flight legs")

        return new_state

    def search_flights(self, state: FlightSearchState) -> FlightSearchState:
        """
        Search for flight options for each leg.

        This is Step 2 of the workflow. It queries flight search APIs
        for each leg in the itinerary.

        Key Feature: Results are accumulated in state using Annotated[List, add]
        which tells the system to APPEND rather than REPLACE.

        Args:
            state: State with extracted itinerary

        Returns:
            Updated state with flight search results
        """
        logger.info("Step 2: Searching flights for each leg")

        new_state = state.copy()

        for i, leg in enumerate(state["itinerary"]):
            origin, destination, date = leg

            logger.info(f"  Searching: {origin} → {destination} on {date}")

            try:
                # Call search API
                results = perform_search_api(origin, destination, date)

                # Package results with metadata
                leg_results = {
                    "leg_number": i + 1,
                    "origin": origin,
                    "destination": destination,
                    "date": date,
                    "options": results,
                    "search_time": time.time()
                }

                # Accumulate in state
                # Because flight_results is Annotated[List[dict], add],
                # this appends rather than replaces
                new_state["flight_results"].append(leg_results)

                logger.info(f"    Found {len(results)} flight options")

            except Exception as e:
                logger.error(f"    ❌ Search failed for leg {i+1}: {e}")
                new_state["error_count"] += 1
                new_state["metadata"][f"leg_{i+1}_error"] = str(e)

        new_state["current_step"] = "flights_searched"
        self.state_history.append(new_state)

        return new_state

    def compare_pricing(self, state: FlightSearchState) -> FlightSearchState:
        """
        Analyze and compare pricing across options.

        This is Step 3 of the workflow. It evaluates all flight options
        for each leg and identifies:
        - Cheapest option
        - Fastest option
        - Best value (balancing price and time)

        Args:
            state: State with flight search results

        Returns:
            Updated state with pricing comparisons
        """
        logger.info("Step 3: Comparing pricing and options")

        new_state = state.copy()
        comparisons = []

        for leg_results in state["flight_results"]:
            options = leg_results["options"]

            if not options:
                logger.warning(f"No options for leg {leg_results['leg_number']}")
                continue

            # Find cheapest
            cheapest = min(options, key=lambda x: x["price"])

            # Find fastest
            fastest = min(options, key=lambda x: x["duration"])

            # Calculate best value (normalized price + duration score)
            def value_score(flight):
                # Normalize price (0-1 scale, lower is better)
                price_norm = flight["price"] / max(o["price"] for o in options)
                # Normalize duration (0-1 scale, lower is better)
                duration_norm = flight["duration"] / max(o["duration"] for o in options)
                # Combined score (lower is better)
                return price_norm * 0.6 + duration_norm * 0.4

            best_value = min(options, key=value_score)

            # Package comparison
            comparison = {
                "leg_number": leg_results["leg_number"],
                "route": f"{leg_results['origin']} → {leg_results['destination']}",
                "cheapest": {
                    "airline": cheapest["airline"],
                    "price": cheapest["price"],
                    "duration": cheapest["duration"]
                },
                "fastest": {
                    "airline": fastest["airline"],
                    "price": fastest["price"],
                    "duration": fastest["duration"]
                },
                "best_value": {
                    "airline": best_value["airline"],
                    "price": best_value["price"],
                    "duration": best_value["duration"],
                    "value_score": value_score(best_value)
                }
            }

            comparisons.append(comparison)

            logger.info(f"  Leg {comparison['leg_number']}: "
                       f"Cheapest ${comparison['cheapest']['price']} ({comparison['cheapest']['airline']}), "
                       f"Fastest {comparison['fastest']['duration']}h ({comparison['fastest']['airline']})")

        new_state["pricing_comparisons"] = comparisons
        new_state["current_step"] = "pricing_compared"
        self.state_history.append(new_state)

        return new_state

    def generate_recommendation(self, state: FlightSearchState) -> str:
        """
        Generate final recommendation report.

        This is Step 4 (final) of the workflow. It synthesizes all
        accumulated state data into a human-readable recommendation.

        Args:
            state: Complete workflow state

        Returns:
            Formatted recommendation text
        """
        logger.info("Step 4: Generating final recommendation")

        itinerary = state["itinerary"]
        comparisons = state["pricing_comparisons"]

        # Build context for LLM
        findings = []
        for comp in comparisons:
            findings.append(f"""
            Leg {comp['leg_number']}: {comp['route']}
            - Cheapest: {comp['cheapest']['airline']} (${comp['cheapest']['price']})
            - Fastest: {comp['fastest']['airline']} ({comp['fastest']['duration']}h)
            - Best Value: {comp['best_value']['airline']} (${comp['best_value']['price']}, {comp['best_value']['duration']}h)
            """)

        findings_text = "\n".join(findings)

        # Generate synthesis
        synthesis_prompt = f"""
        Create a flight recommendation report based on these findings:

        Original Query: {state['user_query']}

        Itinerary: {len(itinerary)} legs

        Findings per leg:
        {findings_text}

        Provide:
        1. Executive summary (2-3 sentences)
        2. Detailed recommendations per leg
        3. Total cost estimate (best value option)
        4. Total travel time
        5. Booking tips
        """

        response = self.llm.invoke(synthesis_prompt)

        recommendation = response.content

        # Update final state
        final_state = state.copy()
        final_state["current_step"] = "complete"
        final_state["metadata"]["completion_time"] = time.time()
        self.state_history.append(final_state)

        logger.info("✅ Workflow complete")

        return recommendation

    def execute_workflow(self, user_query: str) -> tuple[str, FlightSearchState]:
        """
        Execute the complete workflow.

        This orchestrates all steps in sequence, maintaining state
        throughout. If any step fails, the state up to that point
        is preserved and can be inspected or resumed.

        Args:
            user_query: User's flight booking request

        Returns:
            Tuple of (recommendation text, final state)
        """
        # Initialize state
        initial_state: FlightSearchState = {
            "user_query": user_query,
            "itinerary": [],
            "flight_results": [],
            "pricing_comparisons": [],
            "current_step": "init",
            "error_count": 0,
            "metadata": {
                "start_time": time.time(),
                "workflow_id": f"flight_{int(time.time())}"
            }
        }

        self.state_history = [initial_state]

        try:
            # Execute workflow steps
            state = initial_state

            state = self.extract_itinerary(state)
            if state["current_step"] == "error":
                raise ValueError("Itinerary extraction failed")

            state = self.search_flights(state)

            state = self.compare_pricing(state)

            recommendation = self.generate_recommendation(state)

            return recommendation, state

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            # Return partial state for debugging
            return f"Error: {e}", state


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_stateful_workflow():
    """
    Demonstrate the stateful agent in action.

    This shows:
    1. Complete successful workflow
    2. State persistence across steps
    3. Error recovery capabilities
    """
    print("\n" + "="*70)
    print("Stateful Flight Booking Agent Demo")
    print("="*70)

    # Create agent
    agent = StatefulFlightAgent(llm=llm)

    # Test query
    query = "Book flights: NYC to LA on Nov 15, then LA to Tokyo on Nov 20, returning to NYC on Nov 28"

    print(f"\nUser Query: {query}\n")

    # Execute workflow
    recommendation, final_state = agent.execute_workflow(query)

    # Display results
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(recommendation)

    print("\n" + "="*70)
    print("FINAL STATE SUMMARY")
    print("="*70)
    print(f"Workflow Status: {final_state['current_step']}")
    print(f"Legs Processed: {len(final_state['itinerary'])}")
    print(f"Flight Options Found: {len(final_state['flight_results'])}")
    print(f"Comparisons Generated: {len(final_state['pricing_comparisons'])}")
    print(f"Errors Encountered: {final_state['error_count']}")

    print("\n" + "="*70)
    print("STATE HISTORY (Workflow Trace)")
    print("="*70)
    for i, state in enumerate(agent.state_history):
        print(f"Step {i}: {state['current_step']}")

    print("\n✅ Demonstration complete!")


def demonstrate_resumability():
    """
    Demonstrate workflow resumability after failure.

    This shows how stateful design enables resuming from
    the last successful step without re-executing previous work.
    """
    print("\n" + "="*70)
    print("Resumability Demo: Simulating Failure and Recovery")
    print("="*70)

    agent = StatefulFlightAgent(llm=llm)

    # Initialize state
    state: FlightSearchState = {
        "user_query": "NYC to LA to Tokyo",
        "itinerary": [],
        "flight_results": [],
        "pricing_comparisons": [],
        "current_step": "init",
        "error_count": 0,
        "metadata": {}
    }

    # Step 1: Extract itinerary (succeeds)
    print("\n▶️  Step 1: Extract itinerary")
    state = agent.extract_itinerary(state)
    print(f"   Status: {state['current_step']}")
    print(f"   Itinerary: {state['itinerary']}")

    # Step 2: Search flights (succeeds)
    print("\n▶️  Step 2: Search flights")
    state = agent.search_flights(state)
    print(f"   Status: {state['current_step']}")
    print(f"   Results: {len(state['flight_results'])} legs searched")

    # Simulate failure at Step 3
    print("\n▶️  Step 3: Compare pricing")
    print("   ❌ SIMULATED FAILURE: System crash during pricing comparison")

    # Save state (simulating persistence)
    saved_state = state.copy()
    print(f"   💾 State saved at: {saved_state['current_step']}")

    # Recovery: Resume from saved state
    print("\n▶️  Recovery: Resuming workflow from saved state")
    print(f"   📂 Loading state from: {saved_state['current_step']}")
    print(f"   ✅ Itinerary preserved: {len(saved_state['itinerary'])} legs")
    print(f"   ✅ Search results preserved: {len(saved_state['flight_results'])} searches")

    # Continue from where we left off
    recovered_state = agent.compare_pricing(saved_state)
    print(f"   ✅ Step 3 completed: {recovered_state['current_step']}")

    # Complete workflow
    final_rec = agent.generate_recommendation(recovered_state)
    print(f"   ✅ Step 4 completed: {recovered_state['current_step']}")

    print("\n" + "="*70)
    print("KEY INSIGHT: Stateful Design Enables Resumability")
    print("="*70)
    print("- Steps 1-2 were NOT re-executed after failure")
    print("- State from successful steps was preserved")
    print("- Workflow resumed from Step 3 onward")
    print("- Total re-work: Only 2 steps instead of 4")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run demonstrations of stateful agent capabilities.
    """
    # Demo 1: Complete workflow
    demonstrate_stateful_workflow()

    # Demo 2: Resumability after failure
    demonstrate_resumability()

    print("\n" + "="*70)
    print("All demonstrations completed! ✅")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. State persists across all workflow steps")
    print("2. Failures don't lose prior work")
    print("3. Workflows can resume from any step")
    print("4. Full audit trail available via state history")
    print("5. Type safety via TypedDict state schema")
    print("="*70)
