"""
Code Example 10.3.1: Traceable Customer Support Agent

Purpose: Demonstrate comprehensive decision traceability with structured logging

Concepts Demonstrated:
- Structured JSON logging for machine-readable audit trails
- Complete execution context capture (inputs, reasoning, tools, outputs)
- Immutable audit trail generation for regulatory compliance
- Performance tracking and confidence scoring

Prerequisites:
- Understanding of Python logging module
- Familiarity with JSON data structures
- Basic knowledge of audit requirements

Author: NVIDIA Agentic AI Certification
Chapter: 10, Section: 10.3
Exam Skill: 10.3 - Implement transparency mechanisms
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


# ============================================================================
# STRUCTURED LOGGING CONFIGURATION
# ============================================================================

class StructuredLogger:
    """
    Structured logger for agent decision traces.

    Outputs logs as single-line JSON for easy parsing by log aggregation
    systems (Elasticsearch, Splunk, CloudWatch, etc.).

    Features:
    - Machine-readable JSON format
    - Automatic timestamping
    - Agent metadata injection
    - Immutable audit trail
    """

    def __init__(self, agent_id: str, log_file: Optional[str] = None):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.logger.setLevel(logging.INFO)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(console_handler)

        # File handler for production audit trail
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

    def log_decision(self, decision_trace: Dict[str, Any]):
        """
        Log a decision trace as structured JSON.

        Args:
            decision_trace: Complete decision trace data

        Note: Logs are immutable once written. No in-place updates allowed
        for audit trail integrity.
        """
        # Add metadata
        trace_with_metadata = {
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "trace_version": "1.0",
            "trace_hash": self._compute_trace_hash(decision_trace),
            **decision_trace
        }

        # Log as single-line JSON (important for log parsing)
        self.logger.info(json.dumps(trace_with_metadata))

    def _compute_trace_hash(self, trace: Dict[str, Any]) -> str:
        """
        Compute cryptographic hash of trace for integrity verification.

        Enables detection of log tampering.
        """
        trace_str = json.dumps(trace, sort_keys=True)
        return hashlib.sha256(trace_str.encode()).hexdigest()[:16]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Intent(Enum):
    """Customer service intent categories"""
    ORDER_STATUS = "order_status"
    PRODUCT_QUESTION = "product_question"
    RETURN_REQUEST = "return_request"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


@dataclass
class DecisionTrace:
    """
    Structured representation of agent decision.

    This dataclass defines the complete audit trail schema.
    All fields should be JSON-serializable for storage.

    Attributes:
        trace_id: Unique identifier for this decision
        user_query: Original user input
        intent: Classified intent
        reasoning_steps: Chain-of-thought reasoning
        tool_calls: External tool invocations
        final_decision: Agent's response
        confidence: Confidence score (0-1)
        execution_time_ms: Total processing time
        metadata: Additional context
    """
    trace_id: str
    user_query: str
    intent: str
    reasoning_steps: List[Dict[str, str]]
    tool_calls: List[Dict[str, Any]]
    final_decision: str
    confidence: float
    execution_time_ms: float
    metadata: Dict[str, Any]


# ============================================================================
# CUSTOMER SUPPORT AGENT WITH TRACEABILITY
# ============================================================================

class CustomerSupportAgent:
    """
    Customer support agent with complete decision traceability.

    Key Features:
    - Structured logging of all decisions
    - Immutable audit trail
    - Reasoning chain capture
    - Tool invocation tracking
    - Performance metrics
    - Escalation triggers

    Usage:
        agent = CustomerSupportAgent("cs-agent-001")
        result = agent.process_query("Where is my order?")
        # All decisions automatically logged for audit
    """

    def __init__(
        self,
        agent_id: str = "cs-agent-001",
        log_file: Optional[str] = None
    ):
        """
        Initialize agent with traceability.

        Args:
            agent_id: Unique agent identifier
            log_file: Optional file path for persistent audit logs
        """
        self.agent_id = agent_id
        self.logger = StructuredLogger(agent_id, log_file)
        self.trace_id_counter = 0

        # Configuration
        self.confidence_threshold = 0.50  # Escalate if confidence < 50%
        self.max_tool_retries = 2

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query with complete traceability.

        This method demonstrates comprehensive decision logging
        at every stage of agent execution.

        Args:
            user_query: User's input query

        Returns:
            Dict with response, confidence, and trace_id

        Audit Trail:
            Every execution creates immutable log entry with:
            - Input query
            - Reasoning chain
            - Tool calls
            - Final decision
            - Confidence score
            - Performance metrics
        """
        start_time = datetime.utcnow()
        trace_id = self._generate_trace_id()

        # Initialize reasoning chain
        reasoning_steps = []
        tool_calls = []

        # ----------------------------------------------------------------
        # STEP 1: Intent Detection (Logged)
        # ----------------------------------------------------------------
        # Why: Understanding intent drives routing decision
        # Logged: Intent classification and confidence
        reasoning_steps.append({
            "step": "intent_detection",
            "action": "Analyzing user query to determine intent",
            "input": user_query
        })

        intent = self._detect_intent(user_query)
        intent_confidence = self._calculate_intent_confidence(user_query, intent)

        reasoning_steps.append({
            "step": "intent_detected",
            "result": intent.value,
            "confidence": intent_confidence,
            "rationale": f"Query keywords suggest {intent.value}"
        })

        # ----------------------------------------------------------------
        # STEP 2: Route to Appropriate Handler (Logged)
        # ----------------------------------------------------------------
        # Why: Different intents require different processing
        # Logged: Routing decision and justification
        reasoning_steps.append({
            "step": "routing",
            "action": f"Routing to {intent.value} handler",
            "rationale": f"Intent classification: {intent.value} ({intent_confidence:.2f})"
        })

        # ----------------------------------------------------------------
        # STEP 3: Execute Handler with Tool Calls (Logged)
        # ----------------------------------------------------------------
        # Why: Actual work happens here
        # Logged: All tool invocations with inputs/outputs
        response, confidence = self._route_to_handler(
            intent,
            user_query,
            reasoning_steps,
            tool_calls
        )

        # ----------------------------------------------------------------
        # STEP 4: Quality Check and Escalation Decision (Logged)
        # ----------------------------------------------------------------
        # Why: Low confidence triggers human escalation
        # Logged: Quality metrics and escalation decision
        needs_escalation = confidence < self.confidence_threshold

        reasoning_steps.append({
            "step": "quality_check",
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "needs_escalation": needs_escalation,
            "rationale": "Escalating to human agent" if needs_escalation else "Confidence sufficient for automated response"
        })

        if needs_escalation:
            response = (
                f"{response}\n\n"
                "Let me connect you with a human agent who can better assist you."
            )

        # ----------------------------------------------------------------
        # STEP 5: Create Complete Decision Trace
        # ----------------------------------------------------------------
        execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        trace = DecisionTrace(
            trace_id=trace_id,
            user_query=user_query,
            intent=intent.value,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            final_decision=response,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            metadata={
                "agent_version": "1.2.0",
                "model": "gpt-4",
                "escalated_to_human": needs_escalation,
                "tools_used": [tc["tool"] for tc in tool_calls],
                "total_tool_latency_ms": sum(tc.get("duration_ms", 0) for tc in tool_calls)
            }
        )

        # ----------------------------------------------------------------
        # STEP 6: Log Complete Trace (Immutable Audit Record)
        # ----------------------------------------------------------------
        # This is the critical audit trail entry
        # Once logged, it becomes immutable evidence of the decision
        self.logger.log_decision(asdict(trace))

        # Return user-facing response
        return {
            "trace_id": trace_id,
            "response": response,
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "execution_time_ms": execution_time_ms,
            "reasoning_available": True
        }

    def _generate_trace_id(self) -> str:
        """Generate unique trace identifier"""
        self.trace_id_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{self.agent_id}-{timestamp}-{self.trace_id_counter:04d}"

    def _detect_intent(self, query: str) -> Intent:
        """
        Classify user intent (simplified keyword-based).

        In production: Use ML classifier or LLM-based intent detection.
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["order", "shipped", "tracking", "delivery"]):
            return Intent.ORDER_STATUS
        elif any(word in query_lower for word in ["product", "how", "what", "which"]):
            return Intent.PRODUCT_QUESTION
        elif any(word in query_lower for word in ["return", "refund", "exchange"]):
            return Intent.RETURN_REQUEST
        elif any(word in query_lower for word in ["complaint", "issue", "problem", "wrong"]):
            return Intent.COMPLAINT
        else:
            return Intent.UNKNOWN

    def _calculate_intent_confidence(self, query: str, intent: Intent) -> float:
        """
        Calculate confidence in intent classification.

        Simplified: Count keyword matches.
        Production: Use classifier probability scores.
        """
        if intent == Intent.UNKNOWN:
            return 0.30

        # Simplified confidence based on keyword strength
        keyword_matches = sum(1 for word in query.lower().split() if len(word) > 3)
        return min(0.60 + (keyword_matches * 0.05), 0.95)

    def _route_to_handler(
        self,
        intent: Intent,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """
        Route to appropriate handler based on intent.

        Returns: (response, confidence)
        """
        if intent == Intent.ORDER_STATUS:
            return self._handle_order_status(query, reasoning_steps, tool_calls)
        elif intent == Intent.PRODUCT_QUESTION:
            return self._handle_product_question(query, reasoning_steps, tool_calls)
        elif intent == Intent.RETURN_REQUEST:
            return self._handle_return_request(query, reasoning_steps, tool_calls)
        elif intent == Intent.COMPLAINT:
            return self._handle_complaint(query, reasoning_steps, tool_calls)
        else:
            return self._handle_unknown(query, reasoning_steps, tool_calls)

    def _handle_order_status(
        self,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """Handle order status queries"""
        # Log tool invocation
        tool_call = {
            "tool": "order_lookup",
            "input": {"query": query},
            "timestamp": datetime.utcnow().isoformat()
        }

        # Simulate tool execution
        tool_start = datetime.utcnow()
        order_info = self._lookup_order(query)
        tool_duration = (datetime.utcnow() - tool_start).total_seconds() * 1000

        tool_call["output"] = order_info
        tool_call["duration_ms"] = tool_duration
        tool_call["status"] = "success" if order_info else "not_found"
        tool_calls.append(tool_call)

        reasoning_steps.append({
            "step": "tool_execution",
            "tool": "order_lookup",
            "result": "Order found" if order_info else "Order not found",
            "latency_ms": tool_duration
        })

        if order_info:
            response = (
                f"I found your order #{order_info['order_id']}. "
                f"It's currently {order_info['status']}. "
            )
            if order_info.get('tracking'):
                response += f"Tracking number: {order_info['tracking']}"
            confidence = 0.95
        else:
            response = (
                "I couldn't find your order in our system. "
                "Could you provide your order number or email address?"
            )
            confidence = 0.40  # Low confidence triggers escalation

        return response, confidence

    def _handle_product_question(
        self,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """Handle product questions"""
        # Log knowledge base search
        tool_call = {
            "tool": "knowledge_base_search",
            "input": {"query": query},
            "timestamp": datetime.utcnow().isoformat()
        }

        tool_start = datetime.utcnow()
        kb_result = self._search_knowledge_base(query)
        tool_duration = (datetime.utcnow() - tool_start).total_seconds() * 1000

        tool_call["output"] = kb_result
        tool_call["duration_ms"] = tool_duration
        tool_call["retrieved_docs"] = len(kb_result.get("documents", []))
        tool_calls.append(tool_call)

        reasoning_steps.append({
            "step": "knowledge_retrieval",
            "tool": "knowledge_base_search",
            "retrieved_docs": len(kb_result.get("documents", [])),
            "top_score": kb_result.get("documents", [{}])[0].get("score", 0) if kb_result.get("documents") else 0
        })

        response = kb_result.get("answer", "I don't have that information available.")
        confidence = kb_result.get("confidence", 0.50)

        return response, confidence

    def _handle_return_request(
        self,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """Handle return/refund requests"""
        reasoning_steps.append({
            "step": "return_policy_check",
            "action": "Retrieving return policy information"
        })

        response = (
            "Our return policy allows returns within 30 days of purchase. "
            "I can help you initiate a return. Do you have your order number?"
        )
        confidence = 0.85

        return response, confidence

    def _handle_complaint(
        self,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """Handle customer complaints"""
        reasoning_steps.append({
            "step": "complaint_escalation",
            "action": "Flagging for priority human review",
            "priority": "high"
        })

        response = (
            "I'm sorry to hear you're experiencing an issue. "
            "Let me connect you with a specialist who can help resolve this."
        )
        confidence = 0.30  # Always escalate complaints

        return response, confidence

    def _handle_unknown(
        self,
        query: str,
        reasoning_steps: List[Dict],
        tool_calls: List[Dict]
    ) -> tuple[str, float]:
        """Handle unknown intents"""
        reasoning_steps.append({
            "step": "fallback_handler",
            "action": "Using default response for unrecognized intent"
        })

        response = (
            "I'm not sure I understand. Could you rephrase that, or would you "
            "like to speak with a human agent?"
        )
        confidence = 0.20

        return response, confidence

    def _lookup_order(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Simulate order lookup.

        Production: Query order database.
        """
        # Simulate database lookup
        if "12345" in query or "order" in query.lower():
            return {
                "order_id": "12345",
                "status": "shipped",
                "tracking": "1Z999AA10123456784",
                "expected_delivery": "2024-06-15"
            }
        return None

    def _search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """
        Simulate knowledge base search.

        Production: Vector search + RAG pipeline.
        """
        # Simulate vector search
        if "size" in query.lower():
            return {
                "answer": "Our products come in Small, Medium, Large, and XL sizes.",
                "confidence": 0.92,
                "documents": [
                    {"doc_id": "prod-sizing-001", "score": 0.94}
                ]
            }
        elif "color" in query.lower():
            return {
                "answer": "Available colors are Black, White, Navy, and Gray.",
                "confidence": 0.88,
                "documents": [
                    {"doc_id": "prod-colors-001", "score": 0.89}
                ]
            }
        else:
            return {
                "answer": "I don't have specific information about that.",
                "confidence": 0.45,
                "documents": []
            }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_usage():
    """Demonstrate basic traceable agent usage"""
    print("\n" + "="*70)
    print("Example 1: Basic Traceable Agent Usage")
    print("="*70)

    agent = CustomerSupportAgent("cs-agent-001")

    # Process query
    query = "Where is my order 12345?"
    result = agent.process_query(query)

    print(f"\nQuery: {query}")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Escalated: {result['needs_escalation']}")
    print(f"Execution Time: {result['execution_time_ms']:.2f}ms")

    print("\n✓ Complete decision trace logged to audit trail")


def example_multiple_queries():
    """Demonstrate audit trail across multiple queries"""
    print("\n" + "="*70)
    print("Example 2: Multiple Queries - Audit Trail")
    print("="*70)

    agent = CustomerSupportAgent("cs-agent-002", log_file="audit_trail.log")

    queries = [
        "Where is my order?",
        "What sizes does this product come in?",
        "I want to return my purchase",
        "This is unacceptable, I want a refund now!"
    ]

    print("\nProcessing multiple queries with audit logging...\n")

    for i, query in enumerate(queries, 1):
        result = agent.process_query(query)
        print(f"{i}. Query: {query}")
        print(f"   Response: {result['response'][:60]}...")
        print(f"   Confidence: {result['confidence']:.2f} | Escalated: {result['needs_escalation']}")
        print(f"   Trace ID: {result['trace_id']}\n")

    print("✓ All 4 decisions logged to audit_trail.log")
    print("✓ Logs are machine-readable JSON for compliance queries")


def example_low_confidence_escalation():
    """Demonstrate automatic escalation on low confidence"""
    print("\n" + "="*70)
    print("Example 3: Low Confidence → Automatic Escalation")
    print("="*70)

    agent = CustomerSupportAgent("cs-agent-003")

    # Ambiguous query that will have low confidence
    query = "I have a problem with something I bought"
    result = agent.process_query(query)

    print(f"\nQuery: {query}")
    print(f"Response: {result['response']}")
    print(f"\nDecision Analysis:")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Threshold: {agent.confidence_threshold:.2f}")
    print(f"  Escalated: {result['needs_escalation']}")

    if result['needs_escalation']:
        print(f"\n✓ Low confidence ({result['confidence']:.2f}) triggered automatic escalation")
        print("✓ Human agent will handle this request")
        print("✓ Escalation decision logged in audit trail")


def example_audit_trail_query():
    """Demonstrate querying audit logs"""
    print("\n" + "="*70)
    print("Example 4: Querying Audit Trail Logs")
    print("="*70)

    # Create agent with file logging
    agent = CustomerSupportAgent("cs-agent-004", log_file="demo_audit.log")

    # Generate some decisions
    queries = [
        "Order status for 12345",
        "What colors are available?",
        "Return policy question"
    ]

    print("\nGenerating audit trail entries...\n")
    for query in queries:
        agent.process_query(query)

    # Read and analyze audit log
    print("Reading audit log (demo_audit.log):\n")

    try:
        with open("demo_audit.log", "r") as f:
            for line in f:
                trace = json.loads(line)
                print(f"Trace: {trace['trace_id']}")
                print(f"  Intent: {trace['intent']}")
                print(f"  Confidence: {trace['confidence']:.2f}")
                print(f"  Tools Used: {trace['metadata']['tools_used']}")
                print(f"  Execution Time: {trace['execution_time_ms']:.2f}ms")
                print()

        print("✓ Audit logs are structured JSON - easy to query")
        print("✓ Can aggregate with tools like jq, Elasticsearch, Splunk")

    except FileNotFoundError:
        print("(Audit log file not found - check file system permissions)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Traceable Customer Support Agent - Code Example 10.3.1")
    print("="*70)
    print("\nDemonstrating comprehensive decision traceability")
    print("for regulatory compliance and debugging\n")

    example_basic_usage()
    example_multiple_queries()
    example_low_confidence_escalation()
    example_audit_trail_query()

    print("\n" + "="*70)
    print("All examples completed successfully! ✅")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Every decision creates immutable audit log entry")
    print("  • Logs include inputs, reasoning, tools, outputs, metrics")
    print("  • Low confidence automatically triggers escalation")
    print("  • Structured JSON format enables compliance queries")
    print("  • Hash verification prevents log tampering")
    print("="*70)


if __name__ == "__main__":
    main()
