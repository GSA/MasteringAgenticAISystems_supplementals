# ====================================================================
# COMPARISON DEMONSTRATION
# ====================================================================

def demonstrate_difference():
    """Show the critical difference between stateless and stateful."""

    print("="*70)
    print("Stateless Agent Test")
    print("="*70)

    # Stateless agent
    stateless = StatelessFlightAgent(llm=None)

    # Simulate a failure mid-execution
    try:
        result = stateless.process_query("Book flights: NYC → LA → Tokyo")
        # If ANY step fails, we must restart from scratch
        # No way to recover intermediate results
    except Exception as e:
        print(f"❌ Failure: {e}")
        print("❌ All progress lost - must restart from beginning")

    print("\n" + "="*70)
    print("Stateful Agent Test")
    print("="*70)

    # Stateful agent
    stateful = StatefulFlightAgent(llm=None)

    # Initialize state
    stateful.state["user_query"] = "Book flights: NYC → LA → Tokyo"

    # Execute step by step, state persists
    state = stateful.extract_itinerary(stateful.state)
    print(f"✅ Step 1 complete. State: {state['current_step']}")

    state = stateful.search_flights(state)
    print(f"✅ Step 2 complete. State: {state['current_step']}")
    print(f"   Found {len(state['flight_results'])} flight options")

    # Simulate failure here
    try:
        # Even if this fails, we still have state from previous steps
        state = stateful.compare_pricing(state)
    except Exception as e:
        print(f"⚠️  Step 3 failed: {e}")
        print(f"✅ But we can resume! State preserved:")
        print(f"   - Itinerary: {state['itinerary']}")
        print(f"   - Flight results: {len(state['flight_results'])} options")
        print(f"   - Can retry Step 3 without re-executing Steps 1-2")
