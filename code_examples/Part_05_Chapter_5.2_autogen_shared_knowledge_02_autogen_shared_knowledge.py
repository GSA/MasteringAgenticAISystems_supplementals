from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from typing import Dict, List, Set
import json

class SharedKnowledgeGraph:
    """Shared graph state accessible to all agents"""
    def __init__(self):
        self.thoughts: Dict[str, dict] = {}  # thought_id -> {content, agent, dependencies}
        self.edges: Set[tuple] = set()        # (source_id, target_id)

    def add_thought(self, thought_id: str, content: str, agent: str, depends_on: List[str] = None):
        """Agent contributes thought to shared graph"""
        self.thoughts[thought_id] = {
            "content": content,
            "agent": agent,
            "dependencies": depends_on or []
        }
        if depends_on:
            for dep_id in depends_on:
                self.edges.add((dep_id, thought_id))
        return thought_id

    def query_dependencies(self, topic: str) -> List[str]:
        """Find existing thoughts on topic to avoid duplication"""
        relevant = []
        for tid, thought in self.thoughts.items():
            if topic.lower() in thought["content"].lower():
                relevant.append(f"{tid}: {thought['content'][:100]}...")
        return relevant

    def get_dependent_thoughts(self, thought_id: str) -> List[str]:
        """Find thoughts that depend on this one (for invalidation)"""
        dependents = []
        for source, target in self.edges:
            if source == thought_id:
                dependents.append(target)
        # Recursively find transitive dependents
        all_dependents = set(dependents)
        for dep in dependents:
            all_dependents.update(self.get_dependent_thoughts(dep))
        return list(all_dependents)

# Initialize shared graph
shared_graph = SharedKnowledgeGraph()

# Configure specialized agents with graph access
financial_agent = AssistantAgent(
    name="FinancialAnalyst",
    system_message=f"""You are a financial analyst for datacenter investment decisions.
    You have access to a shared knowledge graph via shared_graph object.
    Before analyzing, check shared_graph.query_dependencies(topic) for existing work.
    Contribute findings via shared_graph.add_thought(id, content, agent, depends_on).
    Focus on: CapEx, OpEx, financing, tax implications, ROI projections.""",
    llm_config={"model": "gpt-4", "temperature": 0.7}
)

technical_agent = AssistantAgent(
    name="TechnicalArchitect",
    system_message=f"""You are a technical architect for datacenter infrastructure.
    Use shared knowledge graph to coordinate with other agents.
    Focus on: power/cooling, network connectivity, equipment, timelines, integration.""",
    llm_config={"model": "gpt-4", "temperature": 0.7}
)

risk_agent = AssistantAgent(
    name="RiskAssessor",
    system_message=f"""You are a risk analyst for infrastructure investments.
    Monitor shared graph for findings that introduce risk dependencies.
    Focus on: regulatory compliance, supply chain, geopolitical, demand uncertainty.""",
    llm_config={"model": "gpt-4", "temperature": 0.7}
)

# GroupChat coordinator manages turn-taking and aggregation
def custom_speaker_selection(last_speaker, groupchat):
    """Select next agent based on graph state and dependencies"""
    # If technical findings invalidate financial work, prioritize financial reanalysis
    # If partial analyses ready for synthesis, select coordinator for aggregation
    # Implement domain-specific scheduling logic here
    pass

groupchat = GroupChat(
    agents=[financial_agent, technical_agent, risk_agent],
    messages=[],
    max_round=20,
    speaker_selection_method=custom_speaker_selection
)

manager = GroupChatManager(groupchat=groupchat)

# Execution: agents contribute to shared graph through conversation
user_proxy = UserProxyAgent(name="user", human_input_mode="NEVER")
user_proxy.initiate_chat(
    manager,
    message="Evaluate building a 50MW datacenter. Use shared graph coordination."
)

# Extract final recommendation from aggregated thoughts
final_thoughts = [t for tid, t in shared_graph.thoughts.items()
                  if "recommendation" in t["content"].lower()]
