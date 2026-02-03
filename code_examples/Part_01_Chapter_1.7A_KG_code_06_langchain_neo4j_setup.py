from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
import os

# Initialize Neo4j graph connection
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password=os.getenv("NEO4J_PASSWORD")
)

# Introspect schema for LLM context
graph.refresh_schema()

# Initialize LLM with temperature=0 for deterministic query generation
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# Create Cypher QA chain
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,  # Shows generated Cypher queries
    return_intermediate_steps=True  # Returns query + results
)
