"""
CoolStay RAG routing module

Provides question analysis, domain routing, and response integration functionality.

Core Components:
- QuestionAnalyzer: Question analysis and classification
- DomainRouter: Domain-specific routing and agent selection
- ResponseIntegrator: Multi-agent response integration
"""

from .question_analyzer import (
    QuestionAnalyzer,
    QuestionType,
    QuestionAnalysis,
    create_question_analyzer,
    analyze_question_simple
)

from .domain_router import (
    DomainRouter,
    RoutingStrategy,
    RoutingDecision,
    RoutingResult,
    create_domain_router,
    route_question_simple,
    analyze_routing_decision
)

from .response_integrator import (
    ResponseIntegrator,
    IntegrationStrategy,
    IntegratedResponse,
    ResponseWeights,
    create_response_integrator,
    integrate_simple,
    evaluate_integration_quality
)

__all__ = [
    # Question Analyzer
    "QuestionAnalyzer",
    "QuestionType",
    "QuestionAnalysis",
    "create_question_analyzer",
    "analyze_question_simple",

    # Domain Router
    "DomainRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingResult",
    "create_domain_router",
    "route_question_simple",
    "analyze_routing_decision",

    # Response Integrator
    "ResponseIntegrator",
    "IntegrationStrategy",
    "IntegratedResponse",
    "ResponseWeights",
    "create_response_integrator",
    "integrate_simple",
    "evaluate_integration_quality"
]