from enum import Enum
from typing import Optional

class ParameterSource(Enum):
    """Tracks where parameter values originated"""
    USER_INPUT = "direct_user_input"           # Explicitly stated by user
    RETRIEVED_CONTEXT = "retrieved_context"    # From RAG/database lookup
    TOOL_OUTPUT = "prior_tool_output"          # From earlier action
    INFERRED = "agent_inference"               # Agent inferred from context
    UNKNOWN = "unknown"                        # Cannot determine source

class GroundedParameter:
    """Parameter with provenance tracking for semantic validation"""
    def __init__(self, value, source: ParameterSource,
                 confidence: float, evidence: Optional[str] = None):
        self.value = value
        self.source = source
        self.confidence = confidence  # 0.0-1.0 confidence score
        self.evidence = evidence      # Text supporting this value

def validate_semantic_grounding(params: dict,
                                conversation_history: list,
                                tool_outputs: dict) -> tuple[bool, list[str]]:
    """Validate parameters are grounded in legitimate sources"""
    errors = []

    for param_name, grounded_param in params.items():
        # High-confidence user input or retrieved context: accept
        if grounded_param.source in [ParameterSource.USER_INPUT,
                                      ParameterSource.RETRIEVED_CONTEXT]:
            if grounded_param.confidence >= 0.8:
                continue
            else:
                errors.append(
                    f"{param_name}: Low confidence ({grounded_param.confidence}) "
                    f"for {grounded_param.source.value}"
                )

        # Prior tool output: verify output still valid
        elif grounded_param.source == ParameterSource.TOOL_OUTPUT:
            if not verify_tool_output_still_valid(grounded_param.evidence, tool_outputs):
                errors.append(
                    f"{param_name}: References stale tool output {grounded_param.evidence}"
                )

        # Inferred or unknown: flag for manual review if critical parameter
        else:
            if param_name in CRITICAL_PARAMETERS:
                errors.append(
                    f"{param_name}: Critical parameter with insufficient grounding "
                    f"(source: {grounded_param.source.value})"
                )

    return len(errors) == 0, errors