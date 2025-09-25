"""
CoolStay RAG pipeline module

Provides integrated RAG pipeline and workflow management functionality.

Core Components:
- IntegratedRAGPipeline: Complete end-to-end RAG pipeline
- WorkflowManager: LangGraph-based complex workflow management
"""

from .rag_pipeline import (
    IntegratedRAGPipeline,
    PipelineMode,
    PipelineStage,
    PipelineResult,
    PipelineConfig,
    create_pipeline,
    process_question_simple,
    analyze_pipeline_performance
)

from .workflow_manager import (
    WorkflowManager,
    WorkflowState,
    WorkflowStage as WFStage,
    WorkflowDecision,
    WorkflowConfig,
    create_workflow_manager,
    process_with_workflow
)

__all__ = [
    # RAG Pipeline
    "IntegratedRAGPipeline",
    "PipelineMode",
    "PipelineStage",
    "PipelineResult",
    "PipelineConfig",
    "create_pipeline",
    "process_question_simple",
    "analyze_pipeline_performance",

    # Workflow Manager
    "WorkflowManager",
    "WorkflowState",
    "WFStage",  # Alias to avoid naming conflict
    "WorkflowDecision",
    "WorkflowConfig",
    "create_workflow_manager",
    "process_with_workflow"
]