import json
from openai import OpenAI

class SupportAgent:
    def __init__(self, perception: PerceptionModule, memory: HybridMemorySystem,
                 llm_client: OpenAI, embedding_client: OpenAI):
        self.perception = perception
        self.memory = memory
        self.llm = llm_client
        self.embedder = embedding_client

    def handle_ticket(self, raw_message: str, user_id: str) -> str:
        """Process support ticket using integrated memory and perception"""

        # Step 1: Perception transforms raw input into structured information
        perceived = self.perception.process_input(raw_message, user_id)

        # Step 2: Check episodic memory for user history
        cursor = self.memory.db_conn.execute(
            """SELECT timestamp, issue_category, entities, resolution
               FROM episodic_memory
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT 5""",
            (user_id,)
        )

        user_history = cursor.fetchall()

        # Step 3: If perception detected implicit reference, resolve it
        relevant_past_issue = None
        if perceived.implicit_references:
            # "same issue" refers to most recent issue in same category
            for hist in user_history:
                if hist[1] == perceived.issue_category:
                    relevant_past_issue = {
                        'timestamp': hist[0],
                        'category': hist[1],
                        'entities': json.loads(hist[2]),
                        'resolution': hist[3]
                    }
                    break

        # Step 4: Query semantic memory for relevant knowledge
        query_embedding = self.embedder.embeddings.create(
            input=f"{perceived.issue_category}: {' '.join(perceived.entities)}",
            model="text-embedding-3-small"
        ).data[0].embedding

        semantic_results = self.memory.vector_store.search(
            collection_name=self.memory.collection_name,
            query_vector=query_embedding,
            limit=3
        )

        # Step 5: Query procedural memory for learned resolutions
        cursor = self.memory.db_conn.execute(
            """SELECT successful_resolution,
                      CAST(success_count AS FLOAT) / total_attempts as success_rate
               FROM procedural_memory
               WHERE issue_pattern = ?
               ORDER BY success_rate DESC
               LIMIT 1""",
            (f"{perceived.issue_category}:{','.join(sorted(perceived.entities))}",)
        )

        procedural_result = cursor.fetchone()

        # Step 6: Construct context from memory systems
        context_parts = []

        if relevant_past_issue:
            context_parts.append(
                f"Previous issue (referenced by user): {relevant_past_issue['timestamp']} - "
                f"{relevant_past_issue['category']} involving {relevant_past_issue['entities']}. "
                f"Resolved via: {relevant_past_issue['resolution']}"
            )

        if procedural_result:
            context_parts.append(
                f"Learned resolution pattern ({procedural_result[1]:.0%} success rate): "
                f"{procedural_result[0]}"
            )

        if semantic_results:
            kb_context = "\n".join([
                f"- {result.payload.get('content', '')}"
                for result in semantic_results
            ])
            context_parts.append(f"Knowledge base:\n{kb_context}")

        # Step 7: Generate response using integrated context
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a technical support agent.
                Use provided context from user history, knowledge base, and learned patterns.
                Reference previous issues when the user implies them."""},
                {"role": "user", "content": f"""
                Current issue: {raw_message}

                Structured understanding:
                - Category: {perceived.issue_category}
                - Entities: {perceived.entities}
                - Sentiment: {perceived.sentiment}
                - Urgency: {perceived.urgency}/5

                Context from memory:
                {chr(10).join(context_parts) if context_parts else 'No relevant history'}

                Provide helpful resolution.
                """}
            ]
        )

        resolution = response.choices[0].message.content

        # Step 8: Persist to episodic memory
        self.memory.db_conn.execute(
            """INSERT INTO episodic_memory
               (user_id, issue_category, entities, resolution)
               VALUES (?, ?, ?, ?)""",
            (user_id, perceived.issue_category, json.dumps(perceived.entities),
             resolution)
        )
        self.memory.db_conn.commit()

        return resolution
