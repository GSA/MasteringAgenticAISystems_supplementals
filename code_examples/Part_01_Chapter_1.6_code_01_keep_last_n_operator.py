from typing import TypedDict, List, Annotated

def keep_last_n(n: int):
    """Custom operator that maintains only last n items."""
    def reducer(existing: List, new: List) -> List:
        combined = existing + new
        return combined[-n:] if len(combined) > n else combined
    return reducer

class PrunedState(TypedDict):
    messages: Annotated[List[Message], keep_last_n(15)]
    # Automatically keeps only last 15 messages, pruning older ones
    # This prevents unbounded growth while maintaining recent context
