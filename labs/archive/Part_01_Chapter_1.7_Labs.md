## 1.7.6 Hands-On Lab: Build an Agent with Knowledge Graph Memory

Understanding hybrid architectures conceptually prepares you for implementation, but building a complete system cements the skills through practice. This lab guides you through creating a conversational agent that maintains conversation history in a knowledge graph, enabling it to recall past interactions and make connections across conversations—a capability that simple vector-based memory cannot provide.

### Lab Objective and Learning Outcomes

By completing this lab, you'll gain hands-on experience with knowledge graph integration patterns that translate directly to production systems. The agent you build will store entities mentioned in conversations as graph nodes, link those entities to specific conversation turns, and retrieve relevant past context by traversing the graph when users reference previously discussed topics. This mirrors how production agents maintain long-term memory across sessions—user preferences, past interactions, and accumulated knowledge all benefit from graph-structured storage that enables relational queries.

The learning outcomes extend beyond this specific implementation. You'll internalize how graph schemas evolve to support new features (adding node types, relationship types, and properties without breaking existing structure), understand entity linking challenges when the same concept appears in varied phrasings across conversations, and experience the trade-offs between graph complexity and query expressiveness. These skills transfer to any agent system requiring structured memory beyond simple chat history.

### Lab Setup and Prerequisites

Before building the agent, ensure your environment meets the prerequisites. You need Neo4j running locally (Docker provides the simplest setup), Python with the neo4j driver and LangChain libraries, and spaCy for entity extraction. The setup commands establish this environment:

```bash
# Start Neo4j in Docker (if not already running)
docker run -d \
    --name neo4j-memory-lab \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/lab_password \
    neo4j:latest

# Install Python dependencies
pip install neo4j langchain langchain-openai langchain-community spacy

# Download spaCy model for entity extraction
python -m spacy download en_core_web_sm

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="lab_password"
```

These prerequisites establish the infrastructure your agent needs. Neo4j stores the conversation graph, spaCy extracts entities from user messages, and OpenAI's LLM generates responses augmented with graph-retrieved context. The environment variables make configuration flexible without hardcoding credentials.

### Implementation: Knowledge Graph Memory Agent

Building the agent requires orchestrating multiple capabilities: extracting entities from conversations, storing interactions in the graph, retrieving relevant past context through graph queries, and augmenting LLM prompts with that context. The implementation builds these capabilities incrementally, starting with the core structure:

```python
"""
Hands-On Lab: Knowledge Graph Memory Agent

Demonstrates:
- Storing conversation entities in knowledge graph
- Retrieving past context through graph traversal
- Making connections across conversations
- Evolution of graph schema as agent capabilities grow
"""

from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
import spacy
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime

class KnowledgeGraphMemoryAgent:
    """
    Conversational agent with knowledge graph-based memory.

    Unlike simple chat history that stores message sequences,
    this agent builds a knowledge graph connecting entities
    mentioned across conversations. This enables:
    - Recalling past discussions about specific entities
    - Finding connections between topics across sessions
    - Answering "When did we discuss X?" relational queries
    """

    def __init__(self):
        """
        Initialize agent with Neo4j graph, NER, and LLM.
        """
        # Initialize Neo4j for graph-based memory
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USER"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

        # Initialize entity extraction
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize LLM for response generation
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7
        )
```

This initialization mirrors production agent patterns: graph storage for structured memory, NER for entity extraction, and LLM for generation. The temperature setting (0.7) balances creativity and consistency—higher than query generation (which needs determinism) but lower than creative writing (which tolerates more randomness).

The chat method orchestrates the conversation flow, demonstrating how graph memory integrates with the standard perception-reasoning-action loop:

```python
    def chat(self, user_input: str) -> str:
        """
        Process user input and return response with graph memory.

        Flow:
        1. Extract entities from user's message
        2. Retrieve relevant past context from graph
        3. Generate response using LLM + past context
        4. Store this interaction in graph for future recall

        Args:
            user_input: User's message

        Returns:
            Agent's response incorporating past context
        """
        # Perception: Extract entities from user input
        # These become graph query starting points
        entities = self._extract_entities(user_input)

        # Memory retrieval: Find relevant past context via graph
        # Traverses from mentioned entities to past interactions
        past_context = self._retrieve_graph_context(entities)

        # Reasoning: Generate response using LLM + context
        # LLM has both current input and relevant past discussions
        response = self._generate_response(
            user_input,
            past_context
        )

        # Memory storage: Store interaction in graph
        # Future conversations can reference this exchange
        self._store_interaction(user_input, response, entities)

        return response
```

This flow reveals the graph's role in the agent loop. During perception, entity extraction identifies which concepts the user mentioned—these become graph query targets. During memory retrieval, graph traversal finds past interactions mentioning those same entities. During reasoning, the LLM receives both the current question and relevant historical context. During memory storage, this interaction joins the graph, available for future retrieval.

Entity extraction provides the bridge between unstructured conversation and structured graph queries:

```python
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text for graph linking.

        Args:
            text: User's message

        Returns:
            List of (entity_text, entity_type) tuples

        Example:
            "I'm interested in Tesla and SpaceX"
            Returns: [("Tesla", "ORG"), ("SpaceX", "ORG")]
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter to entity types we want to track in memory
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                entities.append((ent.text, ent.label_))

        return entities
```

The filtering decision matters: we only track entities meaningful for conversation memory (people, organizations, places, products), not every recognized span (dates, quantities, cardinal numbers). This keeps the graph focused on recallable concepts rather than cluttered with temporal or numeric data that doesn't benefit from relationship modeling.

Graph context retrieval demonstrates multi-hop reasoning in practice: given an entity mention, find past interactions where that entity appeared, retrieving conversation context that might inform the current discussion:

```python
    def _retrieve_graph_context(
        self,
        entities: List[Tuple[str, str]]
    ) -> str:
        """
        Retrieve relevant past interactions from knowledge graph.

        For each entity user mentioned, traverse graph to find
        past conversation turns that discussed that entity.

        Args:
            entities: Entities extracted from current input

        Returns:
            Formatted string describing relevant past context
        """
        if not entities:
            return ""

        context_items = []

        with self.driver.session() as session:
            for entity_text, entity_type in entities:
                # Graph traversal: Entity -> MENTIONED_IN -> Interaction
                # Finds past conversations about this entity
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})-[:MENTIONED_IN]->(i:Interaction)
                    RETURN i.timestamp AS timestamp,
                           i.user_input AS user_input,
                           i.agent_response AS response
                    ORDER BY i.timestamp DESC
                    LIMIT 3
                    """,
                    name=entity_text
                )

                for record in result:
                    context_items.append(
                        f"Past discussion about {entity_text} "
                        f"({record['timestamp']}): {record['user_input']}"
                    )

        if context_items:
            return "Relevant past context:\n" + "\n".join(context_items)
        else:
            return ""
```

This query pattern (Entity → MENTIONED_IN → Interaction) implements the core graph memory capability: finding past discussions about specific topics. The LIMIT 3 constraint prevents overwhelming the LLM with every past mention—we want recent relevant context, not the complete history. The timestamp ordering ensures most recent discussions appear first, reflecting recency bias in memory retrieval.

Response generation augments the LLM's prompt with retrieved graph context, giving it awareness of past conversations without maintaining prohibitively long chat histories:

```python
    def _generate_response(
        self,
        user_input: str,
        past_context: str
    ) -> str:
        """
        Generate response using LLM with past context from graph.

        Args:
            user_input: Current user message
            past_context: Retrieved graph context string

        Returns:
            Agent's response incorporating past discussions
        """
        if past_context:
            # Augment prompt with graph-retrieved context
            augmented_input = f"""
            {past_context}

            Current question: {user_input}

            Instructions: Use the past context if relevant to provide
            continuity and recall from previous conversations. Reference
            past discussions naturally when they inform your response.
            """
        else:
            # No relevant past context found
            augmented_input = user_input

        response = self.llm.invoke(augmented_input)
        return response.content
```

The prompt engineering here matters: we instruct the LLM to use past context "if relevant," avoiding forced references to unrelated past discussions. This allows the graph to provide context when it adds value without contaminating responses with irrelevant historical mentions.

Storing interactions in the graph creates the memory foundation for future retrievals, implementing the write-path that complements the read-path we just covered:

```python
    def _store_interaction(
        self,
        user_input: str,
        response: str,
        entities: List[Tuple[str, str]]
    ) -> None:
        """
        Store conversation interaction in knowledge graph.

        Creates:
        - Interaction node with conversation details
        - Entity nodes for mentioned concepts
        - MENTIONED_IN edges linking entities to interaction

        Args:
            user_input: User's message
            response: Agent's response
            entities: Entities extracted from conversation
        """
        timestamp = datetime.now().isoformat()

        with self.driver.session() as session:
            # Create interaction node capturing this conversation turn
            interaction_id = session.run(
                """
                CREATE (i:Interaction {
                    timestamp: $timestamp,
                    user_input: $user_input,
                    agent_response: $response
                })
                RETURN id(i) AS interaction_id
                """,
                timestamp=timestamp,
                user_input=user_input,
                response=response
            ).single()["interaction_id"]

            # Link entities to this interaction
            # MERGE ensures entity nodes aren't duplicated
            for entity_text, entity_type in entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type
                    WITH e
                    MATCH (i:Interaction)
                    WHERE id(i) = $interaction_id
                    MERGE (e)-[:MENTIONED_IN]->(i)
                    """,
                    name=entity_text,
                    type=entity_type,
                    interaction_id=interaction_id
                )
```

This storage pattern demonstrates knowledge graph evolution in practice. Each conversation creates new Interaction nodes, but Entity nodes use MERGE—if "Tesla" was mentioned in previous conversations, we reuse that entity node and create another MENTIONED_IN edge rather than duplicating the entity. Over time, frequently discussed entities accumulate many MENTIONED_IN edges, making them easy to find when relevant context is needed.

The cleanup method handles resource lifecycle, ensuring database connections close properly:

```python
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
```

### Lab Exercise: Testing the Knowledge Graph Memory

Running the agent through a conversation sequence demonstrates how graph memory provides continuity across interactions:

```python
# Initialize agent
agent = KnowledgeGraphMemoryAgent()

print("Knowledge Graph Memory Agent")
print("Type 'quit' to exit\n")

try:
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        response = agent.chat(user_input)
        print(f"Agent: {response}\n")

finally:
    agent.close()
```

A sample conversation reveals the memory capabilities:

```
You: Tell me about Tesla's electric vehicles.
Agent: Tesla manufactures fully electric vehicles focused on performance,
range, and autonomous driving capabilities. Their Model 3, Y, S, and X
cover various market segments from affordable to luxury.

You: What companies did Elon Musk found?
Agent: Elon Musk co-founded or founded several companies including Tesla,
SpaceX, Neuralink, The Boring Company, and was an early investor in PayPal.

[New conversation session, later]

You: We discussed Tesla before. What was that about?
Agent: In our previous conversation, you asked about Tesla's electric
vehicles, and we discussed their product line including Models 3, Y, S,
and X. You also asked about companies Elon Musk founded, where Tesla came
up along with SpaceX and others.
```

Notice the third response: the agent recalled both past discussions by traversing from the "Tesla" entity node to previous Interaction nodes where Tesla was mentioned. Without graph memory, the agent would have no recollection of those past exchanges unless they remained in the chat history buffer (which would overflow after enough messages). The graph provides unbounded memory with precise recall by topic.

### Lab Tasks: Expanding the Agent's Capabilities

Now that you have a working knowledge graph memory agent, extend it with additional capabilities that deepen your understanding of graph-based memory patterns:

**Task 1: Visualize the Knowledge Graph**

Use Neo4j's browser interface to explore the graph your conversations created. Navigate to http://localhost:7474, authenticate with your credentials, and run this Cypher query to visualize entities and their interaction connections:

```cypher
// View all entities and their mentions
MATCH (e:Entity)-[:MENTIONED_IN]->(i:Interaction)
RETURN e, i
LIMIT 50
```

You'll see entity nodes connected to interaction nodes through MENTIONED_IN edges. Frequently discussed entities have many connections, while one-time mentions have single edges. This visualization reveals the knowledge accumulation pattern—the graph grows denser around topics you discuss repeatedly.

**Task 2: Test Cross-Conversation Memory**

Have multiple conversations mentioning the same entities, then query the agent about those entities without directly repeating earlier questions. For example:

```
[Session 1]
You: What's interesting about SpaceX?
Agent: [Response about SpaceX]

[Session 2, hours or days later]
You: We talked about space companies yesterday.
Agent: [Should recall SpaceX discussion without you saying "SpaceX"]
```

This tests whether the agent can infer "space companies" includes SpaceX based on the knowledge graph, demonstrating entity linking and relationship reasoning.

**Task 3: Query Graph Statistics**

Run these Cypher queries directly in Neo4j browser to understand your conversation graph's structure:

```cypher
// Count entities by type
MATCH (e:Entity)
RETURN e.type AS entity_type, count(*) AS count
ORDER BY count DESC

// Most frequently mentioned entities
MATCH (e:Entity)-[:MENTIONED_IN]->(i:Interaction)
RETURN e.name, count(i) AS mentions
ORDER BY mentions DESC
LIMIT 10

// Conversation timeline
MATCH (i:Interaction)
RETURN i.timestamp, substring(i.user_input, 0, 50) AS question_preview
ORDER BY i.timestamp DESC
LIMIT 20
```

These queries reveal memory patterns: which topics dominate your conversations, how entity mentions distribute across sessions, and the temporal structure of your interaction history.

### Challenge Extensions: Production-Ready Features

The basic agent provides functional graph memory, but production systems require additional capabilities. Implement these extensions to approach production-grade memory management:

**Challenge 1: Relationship Extraction Between Entities**

Currently, the agent only links entities to interactions. Extract and store relationships between entities mentioned in the same conversation turn. For example, "Elon Musk founded Tesla" should create (Elon Musk)-[:FOUNDED]->(Tesla) in addition to linking both to the interaction. This enables queries like "What companies did people we discussed found?" that traverse entity-to-entity relationships.

Implementation hint: Use dependency parsing (covered in Chapter 1.7.4) to identify subject-verb-object patterns, creating relationship edges with the verb as the edge type.

**Challenge 2: Temporal Queries**

Add time-based memory retrieval: "What did we discuss about AI last week?" requires filtering interactions by timestamp and checking for entity or topic mentions. Implement a query method that accepts temporal constraints:

```python
def recall_by_timeframe(
    self,
    topic: str,
    days_ago: int
) -> str:
    """Retrieve discussions about topic within time window."""
```

Implementation hint: Convert days_ago to timestamp threshold, filter Interaction nodes by timestamp range, search user_input and agent_response for topic mentions.

**Challenge 3: Entity Disambiguation**

Handle cases where the same name refers to different entities: "I met with Jordan" (person) versus "I'm traveling to Jordan" (country). Currently, both create the same Entity node, merging distinct concepts. Implement disambiguation using entity type and context:

```python
def _disambiguate_entity(
    self,
    entity_text: str,
    entity_type: str,
    context: str
) -> str:
    """
    Return canonical entity identifier handling ambiguous names.

    Args:
        entity_text: Extracted entity name
        entity_type: NER type (PERSON, GPE, etc.)
        context: Surrounding sentence for disambiguation

    Returns:
        Canonical entity ID (e.g., "Jordan_PERSON" vs "Jordan_GPE")
    """
```

Implementation hint: Combine entity_text with entity_type to create unique identifiers, or use entity linking to external knowledge bases (Wikidata) for canonical IDs.

**Challenge 4: Graph Summarization**

When an entity has hundreds of MENTIONED_IN edges, retrieving all past discussions overwhelms the LLM context window. Implement conversation summarization: periodically aggregate past interactions about an entity into a summary node, replacing many detailed interactions with a concise synthesis. This mirrors how human memory consolidates details into gist.

Implementation hint: Run periodic background task that finds entities with >10 interactions, uses LLM to summarize those discussions, creates Summary node, and links entity to summary while pruning or archiving old detailed interactions.

### Lab Validation and Checkpoints

Before considering the lab complete, validate these checkpoints to ensure you've built functional graph memory:

**Checkpoint 1: Basic Memory Retrieval**

The agent recalls past discussions when entities are mentioned again. Test by discussing a topic, ending the session, starting a new session, and referencing that topic indirectly—the agent should remember the earlier conversation.

**Checkpoint 2: Entity Accumulation**

The knowledge graph grows with conversations. Run this query to confirm entity nodes are being created and linked:

```cypher
MATCH (e:Entity)
RETURN count(e) AS total_entities
```

After 10-20 conversational turns mentioning various topics, you should see multiple entity nodes. If count remains zero or very low, entity extraction or storage isn't working.

**Checkpoint 3: Cross-Session Continuity**

Restart the agent application (closing and reopening it) and immediately ask about a topic from earlier sessions. The agent should have memory despite the application restart, proving persistence in Neo4j rather than in-memory storage.

**Checkpoint 4: Graph Structure**

Your graph should exhibit the expected structure: Entity nodes connected to Interaction nodes through MENTIONED_IN edges, with frequently discussed entities having many connections. Visualize this in Neo4j browser to confirm proper relationship creation.