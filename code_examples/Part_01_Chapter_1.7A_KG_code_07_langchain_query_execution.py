question = """
Which companies have received investments from both Google and Microsoft?
What were the investment amounts?
"""

response = cypher_chain.invoke({"query": question})

print("Generated Cypher Query:")
print(response["intermediate_steps"][0]["query"])

print("\nAnswer:")
print(response["result"])
