def robust_web_search(query: str) -> str:
    """Web search with comprehensive error handling."""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.run(query)

        if not results or len(results) < 10:
            return "Search completed but returned no relevant results. Try a different query."

        return results

    except requests.exceptions.Timeout:
        return "Search timed out. The search service may be temporarily unavailable. Please try again."

    except requests.exceptions.RequestException as e:
        return f"Search failed due to network error: {str(e)}. Please try again."

    except Exception as e:
        return f"Unexpected error during search: {str(e)}. Please rephrase your query."
