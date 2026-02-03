# First query: Requires web search for current information
response = agent_executor.invoke({
    "input": "What are the latest developments in quantum computing?"
})

print(response["output"])
# Output might be:
# "Recent developments in quantum computing include IBM's announcement of
# a 1,000+ qubit processor, Google's achievement of quantum error correction
# milestones, and new quantum algorithms for drug discovery. These advances
# suggest we're approaching practical quantum advantage for specific applications."

# Follow-up query: Uses memory to resolve "those advances"
response = agent_executor.invoke({
    "input": "Which of those advances is most significant for drug discovery?"
})

print(response["output"])
# Output leverages context from previous response:
# "The most significant advance for drug discovery is the development of new
# quantum algorithms specifically designed for molecular simulation. These
# algorithms can model protein folding and drug interactions with higher
# accuracy than classical approaches, potentially accelerating the discovery
# of new therapeutic compounds."
