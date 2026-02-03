"""
Code Example 5.3.1: Hierarchical Planning Agent

Purpose: Demonstrate multi-level task decomposition for complex goals

Concepts Demonstrated:
- Hierarchical task decomposition: Breaking goals into phases → tasks → actions
- Reflection loops: Quality validation and adaptive replanning
- State management: Tracking progress through complex workflows

Prerequisites:
- LangGraph basics
- Understanding of state machines
- Familiarity with async/await patterns

Author: NVIDIA Certified Generative AI LLM Course
Chapter: 5, Section: 5.3
Exam Skill: 5.3 - Engineer Planning Strategies for Multi-Step Decision-Making
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

# Standard library imports
from typing import List, Dict, Any, TypedDict
import logging
from dataclasses import dataclass
import json

# Third-party imports
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.0,  # Deterministic for planning
    "max_tokens": 2048
}

# Planning configuration
PLANNING_CONFIG = {
    "max_phases": 5,
    "quality_threshold": 0.7,
    "max_replanning_attempts": 2
}

# ============================================================================
# STATE DEFINITION
# ============================================================================

class ResearchState(TypedDict):
    """
    State for hierarchical research planning agent.

    This state structure demonstrates proper planning state management
    with clear separation of concerns.

    Attributes:
        topic (str): Research topic to investigate
        research_plan (List[Dict]): Hierarchical plan structure
        current_phase (int): Current execution phase index
        findings (List[Dict]): Accumulated research findings
        report_sections (Dict[str, str]): Generated report sections
        quality_score (float): Overall research quality (0-1)
        replanning_needed (bool): Flag for adaptive replanning
        replanning_count (int): Number of replanning iterations
    """
    topic: str
    research_plan: List[Dict[str, Any]]
    current_phase: int
    findings: List[Dict[str, str]]
    report_sections: Dict[str, str]
    quality_score: float
    replanning_needed: bool
    replanning_count: int


# ============================================================================
# HIERARCHICAL PLANNER IMPLEMENTATION
# ============================================================================

@dataclass
class PlanningPhase:
    """
    Represents a single phase in hierarchical plan.

    Demonstrates structured planning with clear success criteria.
    """
    name: str
    tasks: List[str]
    tools: List[str]
    success_criteria: str
    dependencies: List[str] = None


class HierarchicalPlanner:
    """
    Hierarchical task decomposition for research planning.

    This class demonstrates the core pattern of breaking complex goals
    into multi-level task structures with clear dependencies.

    Key Concepts:
    - Three-level decomposition: Phases → Tasks → Actions
    - Dependency tracking between phases
    - Tool assignment per task
    - Measurable success criteria

    Example:
        >>> planner = HierarchicalPlanner(llm)
        >>> plan = planner.create_research_plan("NVIDIA NIM architecture")
        >>> print(f"Created plan with {len(plan)} phases")
    """

    def __init__(self, llm: ChatOpenAI):
        """
        Initialize hierarchical planner.

        Args:
            llm (ChatOpenAI): Language model for plan generation
        """
        self.llm = llm

    def create_research_plan(self, topic: str) -> List[Dict[str, Any]]:
        """
        Create hierarchical research plan with 3-level decomposition.

        Decomposition Strategy:
        Level 1: Research Phases (Background, Deep Dive, Analysis, Synthesis)
        Level 2: Specific Tasks per phase
        Level 3: Tool invocations and actions

        Args:
            topic (str): Research topic to plan for

        Returns:
            List[Dict]: Structured hierarchical plan

        Example:
            >>> plan = planner.create_research_plan("Transformer architectures")
            >>> assert len(plan) >= 3  # At least 3 phases
            >>> assert all('tasks' in phase for phase in plan)
        """

        logger.info(f"Creating hierarchical plan for: {topic}")

        # ----------------------------------------------------------------
        # PHASE 1: High-Level Decomposition
        # ----------------------------------------------------------------
        # Why: Break research into logical phases before detailed planning
        # Concept: Top-down hierarchical decomposition

        planning_prompt = f"""
        Create a hierarchical research plan for the topic: "{topic}"

        Use this 4-phase structure:
        1. Background Research - Understand fundamentals and context
        2. Deep Dive - Investigate specific technical aspects
        3. Comparative Analysis - Evaluate different approaches
        4. Synthesis - Integrate findings into coherent narrative

        For EACH phase, specify:
        - 2-3 specific tasks to complete
        - Required information sources/tools
        - Clear success criteria
        - Dependencies on previous phases

        Return as JSON array of phase objects.

        Example format:
        [
          {{
            "phase": "background",
            "tasks": ["search_fundamentals", "read_overview_papers"],
            "tools": ["web_search", "arxiv_api"],
            "success_criteria": "Core concepts and terminology identified",
            "dependencies": []
          }},
          ...
        ]
        """

        response = self.llm.invoke(planning_prompt)

        # ----------------------------------------------------------------
        # PHASE 2: Parse and Validate Plan
        # ----------------------------------------------------------------
        # Why: Ensure LLM output is well-formed and usable
        # Concept: Robust parsing with fallback

        try:
            plan = self._parse_plan_from_llm(response.content)
            logger.info(f"✓ Successfully parsed plan with {len(plan)} phases")
        except json.JSONDecodeError:
            logger.warning("LLM output not valid JSON, using structured fallback")
            plan = self._create_fallback_plan(topic)

        # ----------------------------------------------------------------
        # PHASE 3: Validate Plan Quality
        # ----------------------------------------------------------------
        # Why: Ensure plan meets quality standards
        # Concept: Pre-execution validation

        validation_result = self._validate_plan(plan)

        if not validation_result["valid"]:
            logger.warning(f"Plan validation issues: {validation_result['issues']}")
            plan = self._refine_plan(plan, validation_result["issues"])

        logger.info(f"✓ Created validated plan with {len(plan)} phases")

        return plan

    def _parse_plan_from_llm(self, llm_output: str) -> List[Dict[str, Any]]:
        """
        Parse LLM-generated plan into structured format.

        This demonstrates robust parsing of LLM outputs with
        proper error handling.

        Args:
            llm_output (str): Raw LLM response

        Returns:
            List[Dict]: Parsed plan structure

        Raises:
            json.JSONDecodeError: If parsing fails
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in llm_output:
            json_start = llm_output.find("```json") + 7
            json_end = llm_output.find("```", json_start)
            json_str = llm_output[json_start:json_end].strip()
        elif "```" in llm_output:
            json_start = llm_output.find("```") + 3
            json_end = llm_output.find("```", json_start)
            json_str = llm_output[json_start:json_end].strip()
        else:
            json_str = llm_output.strip()

        return json.loads(json_str)

    def _create_fallback_plan(self, topic: str) -> List[Dict[str, Any]]:
        """
        Create structured fallback plan if LLM parsing fails.

        Demonstrates: Graceful degradation with sensible defaults

        Args:
            topic (str): Research topic

        Returns:
            List[Dict]: Default hierarchical plan
        """
        return [
            {
                "phase": "background",
                "tasks": [
                    "search_general_information",
                    "identify_key_concepts",
                    "find_authoritative_sources"
                ],
                "tools": ["web_search", "wikipedia_api"],
                "success_criteria": "Core concepts and terminology documented",
                "dependencies": []
            },
            {
                "phase": "deep_dive",
                "tasks": [
                    "analyze_technical_details",
                    "examine_implementations",
                    "study_case_examples"
                ],
                "tools": ["arxiv_search", "github_search", "technical_docs"],
                "success_criteria": "Technical understanding with specific examples",
                "dependencies": ["background"]
            },
            {
                "phase": "comparative_analysis",
                "tasks": [
                    "compare_approaches",
                    "evaluate_tradeoffs",
                    "benchmark_performance"
                ],
                "tools": ["benchmark_db", "comparison_framework"],
                "success_criteria": "Clear comparison of 3+ approaches",
                "dependencies": ["deep_dive"]
            },
            {
                "phase": "synthesis",
                "tasks": [
                    "integrate_findings",
                    "create_narrative",
                    "generate_recommendations"
                ],
                "tools": ["llm_summarization", "report_generator"],
                "success_criteria": "Coherent report with actionable insights",
                "dependencies": ["comparative_analysis"]
            }
        ]

    def _validate_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate plan structure and completeness.

        Validation checks:
        - Required fields present
        - Reasonable number of phases (2-6)
        - Tasks specified for each phase
        - Success criteria defined
        - Dependencies are valid

        Args:
            plan (List[Dict]): Plan to validate

        Returns:
            Dict: Validation result with 'valid' bool and 'issues' list
        """
        issues = []

        # Check plan size
        if len(plan) < 2:
            issues.append("Plan has too few phases (minimum 2)")
        elif len(plan) > 6:
            issues.append("Plan has too many phases (maximum 6)")

        # Check each phase structure
        required_fields = ["phase", "tasks", "tools", "success_criteria"]

        for i, phase in enumerate(plan):
            for field in required_fields:
                if field not in phase:
                    issues.append(f"Phase {i} missing required field: {field}")

            # Validate tasks
            if "tasks" in phase and len(phase["tasks"]) == 0:
                issues.append(f"Phase {i} has no tasks defined")

            # Validate dependencies
            if "dependencies" in phase:
                for dep in phase["dependencies"]:
                    if not any(p["phase"] == dep for p in plan[:i]):
                        issues.append(f"Phase {i} has invalid dependency: {dep}")

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    def _refine_plan(
        self,
        plan: List[Dict[str, Any]],
        issues: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Refine plan to address validation issues.

        Demonstrates: Self-correction based on validation feedback

        Args:
            plan (List[Dict]): Original plan
            issues (List[str]): Identified issues

        Returns:
            List[Dict]: Refined plan
        """
        # For this example: Apply basic fixes
        # Production: Would use LLM to intelligently refine

        refined_plan = []

        for phase in plan:
            # Ensure all required fields
            if "phase" not in phase:
                phase["phase"] = f"phase_{len(refined_plan)}"
            if "tasks" not in phase or len(phase["tasks"]) == 0:
                phase["tasks"] = ["execute_phase_tasks"]
            if "tools" not in phase:
                phase["tools"] = ["generic_tool"]
            if "success_criteria" not in phase:
                phase["success_criteria"] = "Phase objectives met"

            refined_plan.append(phase)

        return refined_plan


# ============================================================================
# LANGGRAPH WORKFLOW CONSTRUCTION
# ============================================================================

def create_hierarchical_planning_workflow() -> StateGraph:
    """
    Create LangGraph workflow for hierarchical planning with reflection.

    Workflow Structure:
    1. plan_research: Create initial hierarchical plan
    2. execute_phase: Execute current phase tasks
    3. reflect_on_quality: Evaluate progress and quality
    4. replan_if_needed: Adaptive replanning on quality issues
    5. generate_report: Synthesize findings into report

    Returns:
        StateGraph: Compiled workflow ready for execution
    """

    graph = StateGraph(ResearchState)

    # Initialize LLM
    llm = ChatOpenAI(**MODEL_CONFIG)

    # ========================================================================
    # NODE 1: Create Initial Plan
    # ========================================================================

    def plan_research(state: ResearchState) -> ResearchState:
        """
        Create hierarchical research plan.

        This node demonstrates the initial planning phase where
        complex goals are decomposed into structured tasks.
        """
        logger.info(f"Planning research for: {state['topic']}")

        planner = HierarchicalPlanner(llm)
        plan = planner.create_research_plan(state["topic"])

        state["research_plan"] = plan
        state["current_phase"] = 0
        state["findings"] = []
        state["report_sections"] = {}
        state["quality_score"] = 0.0
        state["replanning_needed"] = False
        state["replanning_count"] = 0

        logger.info(f"✓ Initial plan created with {len(plan)} phases")

        return state

    # ========================================================================
    # NODE 2: Execute Current Phase
    # ========================================================================

    def execute_phase(state: ResearchState) -> ResearchState:
        """
        Execute tasks for current phase.

        Demonstrates:
        - Task execution with tool integration
        - Progress tracking
        - Result accumulation
        """
        current_plan = state["research_plan"][state["current_phase"]]
        phase_name = current_plan["phase"]

        logger.info(f"Executing phase: {phase_name}")

        # Simulate phase execution
        # In production: Would actually invoke tools and execute tasks

        execution_prompt = f"""
        Execute research phase: {phase_name}

        Tasks to complete:
        {current_plan['tasks']}

        Available tools:
        {current_plan['tools']}

        Success criteria:
        {current_plan['success_criteria']}

        Provide research findings for this phase.
        """

        response = llm.invoke(execution_prompt)

        # Store findings
        finding = {
            "phase": phase_name,
            "content": response.content,
            "tasks_completed": current_plan["tasks"],
            "sources": ["simulated_source_1", "simulated_source_2"]
        }

        state["findings"].append(finding)

        logger.info(f"✓ Completed phase: {phase_name}")
        logger.info(f"   Tasks executed: {len(current_plan['tasks'])}")

        return state

    # ========================================================================
    # NODE 3: Reflect on Quality
    # ========================================================================

    def reflect_on_quality(state: ResearchState) -> ResearchState:
        """
        Evaluate research quality and completeness.

        Demonstrates:
        - Self-evaluation and reflection
        - Quality scoring
        - Gap identification
        """
        logger.info("Reflecting on research quality...")

        reflection_prompt = f"""
        Evaluate the quality and completeness of research on: {state['topic']}

        Research conducted so far:
        - Phases completed: {state['current_phase'] + 1} / {len(state['research_plan'])}
        - Findings: {len(state['findings'])} sections

        Latest findings:
        {state['findings'][-1] if state['findings'] else 'None yet'}

        Rate the current research quality on scale 0-1:
        - Completeness: Are all key aspects covered?
        - Depth: Is the analysis sufficiently detailed?
        - Accuracy: Are claims well-supported?
        - Coherence: Do findings form a logical narrative?

        Identify any gaps that require additional research.

        Return: {{
          "quality_score": <float 0-1>,
          "gaps": [list of missing aspects],
          "recommendation": "continue" or "replan"
        }}
        """

        response = llm.invoke(reflection_prompt)

        # Parse reflection result
        # Simplified: In production would parse JSON
        quality_score = 0.75  # Simulated score
        gaps = []  # Simulated gap analysis

        state["quality_score"] = quality_score

        # Determine if replanning needed
        if quality_score < PLANNING_CONFIG["quality_threshold"]:
            if state["replanning_count"] < PLANNING_CONFIG["max_replanning_attempts"]:
                state["replanning_needed"] = True
                logger.warning(
                    f"⚠ Quality score {quality_score} below threshold "
                    f"{PLANNING_CONFIG['quality_threshold']}"
                )
            else:
                logger.info("Max replanning attempts reached, proceeding anyway")
        else:
            logger.info(f"✓ Quality score: {quality_score}")

        return state

    # ========================================================================
    # NODE 4: Adaptive Replanning
    # ========================================================================

    def replan_if_needed(state: ResearchState) -> ResearchState:
        """
        Replan if quality issues detected.

        Demonstrates:
        - Adaptive planning based on execution feedback
        - Gap filling through additional tasks
        - Learning from quality evaluation
        """
        if state["replanning_needed"]:
            logger.info("Replanning to address quality gaps...")

            planner = HierarchicalPlanner(llm)

            # Identify what's missing
            replan_prompt = f"""
            Current research plan had quality issues.
            Quality score: {state['quality_score']}
            Threshold: {PLANNING_CONFIG['quality_threshold']}

            Completed phases: {[f['phase'] for f in state['findings']]}

            Create supplementary research tasks to fill gaps.
            Focus on missing aspects and insufficient depth.
            """

            response = llm.invoke(replan_prompt)

            # Add gap-filling phase
            gap_phase = {
                "phase": "gap_filling",
                "tasks": ["address_missing_aspects", "deepen_analysis"],
                "tools": ["specialized_search", "expert_consultation"],
                "success_criteria": "Quality gaps addressed",
                "dependencies": []
            }

            state["research_plan"].append(gap_phase)
            state["replanning_count"] += 1
            state["replanning_needed"] = False

            logger.info("✓ Added gap-filling phase to plan")

        return state

    # ========================================================================
    # NODE 5: Generate Final Report
    # ========================================================================

    def generate_report(state: ResearchState) -> ResearchState:
        """
        Synthesize findings into structured report.

        Demonstrates:
        - Multi-source synthesis
        - Structured output generation
        - Quality-driven content creation
        """
        logger.info("Generating final research report...")

        synthesis_prompt = f"""
        Create a comprehensive research report on: {state['topic']}

        Research findings from {len(state['findings'])} phases:

        {json.dumps(state['findings'], indent=2)}

        Generate a well-structured report with these sections:
        1. Executive Summary (key findings and recommendations)
        2. Background (context and fundamentals)
        3. Technical Analysis (detailed investigation)
        4. Comparative Evaluation (if applicable)
        5. Conclusions and Recommendations

        Ensure all claims are grounded in the research findings.
        """

        response = llm.invoke(synthesis_prompt)

        # Parse into sections (simplified)
        state["report_sections"] = {
            "executive_summary": "Key findings summary...",
            "background": "Background research...",
            "analysis": "Technical analysis...",
            "conclusions": "Conclusions and recommendations..."
        }

        logger.info("✓ Report generated with all sections")
        logger.info(f"   Total findings synthesized: {len(state['findings'])}")
        logger.info(f"   Final quality score: {state['quality_score']}")

        return state

    # ========================================================================
    # GRAPH CONSTRUCTION
    # ========================================================================

    # Add nodes to graph
    graph.add_node("plan", plan_research)
    graph.add_node("execute", execute_phase)
    graph.add_node("reflect", reflect_on_quality)
    graph.add_node("replan", replan_if_needed)
    graph.add_node("report", generate_report)

    # Define edges and routing

    # Entry point
    graph.set_entry_point("plan")

    # Plan → Execute
    graph.add_edge("plan", "execute")

    # Execute → Continue or Reflect
    def should_continue_execution(state: ResearchState) -> str:
        """Route: More phases to execute or move to reflection?"""
        if state["current_phase"] < len(state["research_plan"]) - 1:
            state["current_phase"] += 1
            return "execute"
        return "reflect"

    graph.add_conditional_edges(
        "execute",
        should_continue_execution,
        {
            "execute": "execute",
            "reflect": "reflect"
        }
    )

    # Reflect → Replan
    graph.add_edge("reflect", "replan")

    # Replan → Execute more or Generate report
    def should_execute_more(state: ResearchState) -> str:
        """Route: Replanning added tasks or ready for report?"""
        # If replanning added phases, execute them
        if state["replanning_needed"] or \
           state["current_phase"] < len(state["research_plan"]) - 1:
            if state["current_phase"] < len(state["research_plan"]) - 1:
                state["current_phase"] += 1
            return "execute"
        return "report"

    graph.add_conditional_edges(
        "replan",
        should_execute_more,
        {
            "execute": "execute",
            "report": "report"
        }
    )

    # Report → End
    graph.add_edge("report", END)

    return graph.compile()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_hierarchical_planning():
    """Demonstrate basic hierarchical planning workflow."""
    print("\n" + "="*70)
    print("Example 1: Basic Hierarchical Planning")
    print("="*70)

    # Create workflow
    workflow = create_hierarchical_planning_workflow()

    # Initialize state
    initial_state = ResearchState(
        topic="NVIDIA NIM microservices architecture and deployment patterns",
        research_plan=[],
        current_phase=0,
        findings=[],
        report_sections={},
        quality_score=0.0,
        replanning_needed=False,
        replanning_count=0
    )

    # Execute workflow
    print(f"\nResearching: {initial_state['topic']}")
    print("Executing hierarchical planning workflow...\n")

    final_state = workflow.invoke(initial_state)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"✓ Research completed for: {final_state['topic']}")
    print(f"✓ Phases executed: {len(final_state['findings'])}")
    print(f"✓ Quality score: {final_state['quality_score']:.2f}")
    print(f"✓ Replanning iterations: {final_state['replanning_count']}")
    print(f"✓ Report sections: {list(final_state['report_sections'].keys())}")

    print("\nPhase execution summary:")
    for i, finding in enumerate(final_state['findings'], 1):
        print(f"  {i}. {finding['phase']}: {len(finding['tasks_completed'])} tasks")

    return final_state


def example_with_replanning():
    """Demonstrate hierarchical planning with adaptive replanning."""
    print("\n" + "="*70)
    print("Example 2: Hierarchical Planning with Replanning")
    print("="*70)

    # Note: In this example, we would set up conditions to trigger replanning
    # For demonstration, the reflection node may identify quality gaps

    print("\nThis example demonstrates adaptive replanning when quality")
    print("thresholds are not met. The agent will:")
    print("  1. Execute initial plan")
    print("  2. Reflect on quality")
    print("  3. Identify gaps")
    print("  4. Add supplementary research phases")
    print("  5. Re-execute to fill gaps")

    # Execute (similar to above)
    workflow = create_hierarchical_planning_workflow()

    initial_state = ResearchState(
        topic="Comparative analysis of inference optimization techniques",
        research_plan=[],
        current_phase=0,
        findings=[],
        report_sections={},
        quality_score=0.0,
        replanning_needed=False,
        replanning_count=0
    )

    final_state = workflow.invoke(initial_state)

    print(f"\n✓ Final quality score: {final_state['quality_score']:.2f}")
    print(f"✓ Replanning iterations: {final_state['replanning_count']}")

    if final_state['replanning_count'] > 0:
        print(f"⚠ Agent performed adaptive replanning to improve quality")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all hierarchical planning examples."""
    print("\n" + "="*70)
    print("Hierarchical Planning Agent Examples")
    print("="*70)

    # Run examples
    example_basic_hierarchical_planning()
    example_with_replanning()

    print("\n" + "="*70)
    print("All examples completed successfully! ✅")
    print("="*70)


if __name__ == "__main__":
    main()
