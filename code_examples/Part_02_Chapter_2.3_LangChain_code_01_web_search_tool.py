from langchain.tools import Tool
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# Initialize the search utility
# DuckDuckGoSearchAPIWrapper provides a free web search API
# Alternative: SerpAPI, Google Search API (require API keys)
search = DuckDDuckGoSearchAPIWrapper()

# Wrap the search function with LangChain Tool interface
search_tool = Tool.from_function(
    func=search.run,
    name="web_search",
    description=(
        "Search the web for current information. "
        "Use this when you need recent data, news, breaking events, "
        "or information not in your training data. "
        "Input should be a focused search query string. "
        "Returns: A summary of top search results as text."
    )
)
