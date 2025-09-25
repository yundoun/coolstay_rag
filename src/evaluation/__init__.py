"""
CoolStay RAG evaluation module

Provides ReAct evaluation system and Human-in-the-Loop feedback functionality.

Core Components:
- ReActEvaluationAgent: 6-dimensional ReAct evaluation system
- HITLInterface: Human-in-the-Loop feedback collection and processing
"""

from .react_evaluator import (
    ReActEvaluationAgent,
    EvaluationDimension,
    EvaluationScore,
    DimensionEvaluation,
    ReActEvaluationResult,
    create_react_evaluator,
    evaluate_simple,
    convert_to_dict
)

from .hitl_handler import (
    HITLInterface,
    FeedbackType,
    FeedbackSource,
    InteractionContext,
    HumanFeedback,
    HITLSession,
    FeedbackAnalysis,
    create_hitl_interface,
    collect_simple_rating,
    create_feedback_dashboard
)

__all__ = [
    # ReAct Evaluator
    "ReActEvaluationAgent",
    "EvaluationDimension",
    "EvaluationScore",
    "DimensionEvaluation",
    "ReActEvaluationResult",
    "create_react_evaluator",
    "evaluate_simple",
    "convert_to_dict",

    # HITL Handler
    "HITLInterface",
    "FeedbackType",
    "FeedbackSource",
    "InteractionContext",
    "HumanFeedback",
    "HITLSession",
    "FeedbackAnalysis",
    "create_hitl_interface",
    "collect_simple_rating",
    "create_feedback_dashboard"
]