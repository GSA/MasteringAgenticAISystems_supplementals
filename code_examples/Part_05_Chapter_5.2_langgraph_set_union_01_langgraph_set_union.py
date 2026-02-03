from typing import TypedDict, List, Set, Dict
from langgraph.graph import StateGraph, END

class SetUnionState(TypedDict):
    """State for parallel set union computation"""
    input_sets: List[Set[int]]           # Original sets to union
    partial_unions: Dict[str, Set[int]]  # Intermediate merge results
    merge_queue: List[tuple]             # Pairs ready for merging
    final_union: Set[int]                # Complete result
    merge_depth: int                     # Current layer depth

def pairwise_merge_node(state: SetUnionState) -> SetUnionState:
    """Node: Merge pairs of sets from merge queue"""
    if not state["merge_queue"]:
        return state

    # Execute all pairwise merges in current layer
    next_queue = []
    partial_unions = state["partial_unions"].copy()

    for pair_id, (set_a_key, set_b_key) in enumerate(state["merge_queue"]):
        set_a = partial_unions[set_a_key]
        set_b = partial_unions[set_b_key]

        # Perform union (this is where LLM would be invoked for semantic merging)
        merged = set_a.union(set_b)
        result_key = f"layer{state['merge_depth']}_merge{pair_id}"
        partial_unions[result_key] = merged

        # Queue result for next layer aggregation
        next_queue.append(result_key)

    # Pair up results for next layer
    next_pairs = [(next_queue[i], next_queue[i+1])
                  for i in range(0, len(next_queue)-1, 2)]

    return {
        **state,
        "partial_unions": partial_unions,
        "merge_queue": next_pairs,
        "merge_depth": state["merge_depth"] + 1
    }

def route_merge(state: SetUnionState) -> str:
    """Conditional edge: Continue merging or finalize"""
    if len(state["merge_queue"]) == 0:
        # Single result remaining - extract final union
        return "finalize"
    else:
        # More layers needed
        return "merge"

# Build graph
graph = StateGraph(SetUnionState)
graph.add_node("merge", pairwise_merge_node)
graph.add_node("finalize", lambda s: {**s, "final_union": list(s["partial_unions"].values())[0]})

graph.set_entry_point("merge")
graph.add_conditional_edges("merge", route_merge, {"merge": "merge", "finalize": "finalize"})
graph.add_edge("finalize", END)

compiled_graph = graph.compile()
