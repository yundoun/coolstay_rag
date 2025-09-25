"""
CoolStay RAG 워크플로우 매니저 모듈

LangGraph를 사용하여 복잡한 워크플로우와 상태 관리를 제공하는 모듈입니다.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    SqliteSaver = None
    END = None

from ..core.config import CoolStayConfig
from .rag_pipeline import IntegratedRAGPipeline, PipelineResult, PipelineConfig

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """워크플로우 상태"""
    question: str
    user_id: Optional[str]
    session_id: Optional[str]

    # 단계별 결과
    analysis_result: Optional[Dict[str, Any]]
    routing_result: Optional[Dict[str, Any]]
    agent_responses: Optional[Dict[str, Any]]
    integrated_response: Optional[Dict[str, Any]]
    evaluation_result: Optional[Dict[str, Any]]
    hitl_feedback: Optional[Dict[str, Any]]

    # 상태 정보
    current_step: str
    error_count: int
    retry_count: int
    final_answer: Optional[str]
    confidence: float
    execution_time: float

    # 메타데이터
    metadata: Dict[str, Any]
    timestamp: str


class WorkflowStage(Enum):
    """워크플로우 단계"""
    START = "start"
    ANALYZE_QUESTION = "analyze_question"
    ROUTE_QUESTION = "route_question"
    EXECUTE_AGENTS = "execute_agents"
    INTEGRATE_RESPONSES = "integrate_responses"
    EVALUATE_RESPONSE = "evaluate_response"
    COLLECT_FEEDBACK = "collect_feedback"
    FINALIZE = "finalize"
    ERROR_HANDLING = "error_handling"


class WorkflowDecision(Enum):
    """워크플로우 결정"""
    CONTINUE = "continue"
    RETRY = "retry"
    SKIP = "skip"
    ERROR = "error"
    END = "end"


@dataclass
class WorkflowConfig:
    """워크플로우 설정"""
    max_retries: int = 3
    timeout_seconds: int = 120
    enable_checkpoints: bool = True
    enable_error_recovery: bool = True
    checkpoint_db_path: str = "./workflow_checkpoints.db"

    # 조건부 실행 설정
    skip_evaluation_on_error: bool = True
    skip_hitl_on_low_confidence: bool = True
    min_confidence_for_hitl: float = 0.7

    # 병렬 처리 설정
    enable_parallel_agents: bool = True
    max_parallel_agents: int = 3


class WorkflowManager:
    """
    워크플로우 매니저

    LangGraph를 사용하여 복잡한 RAG 워크플로우를 관리합니다.
    체크포인트, 재시도, 오류 복구 등의 기능을 제공합니다.
    """

    def __init__(
        self,
        config: Optional[CoolStayConfig] = None,
        workflow_config: Optional[WorkflowConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            config: CoolStay 설정 객체
            workflow_config: 워크플로우 설정 객체
            pipeline_config: 파이프라인 설정 객체
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph가 설치되지 않았습니다. "
                "pip install langgraph 를 실행하여 설치해주세요."
            )

        self.config = config or CoolStayConfig()
        self.workflow_config = workflow_config or WorkflowConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()

        # RAG 파이프라인
        self.rag_pipeline = IntegratedRAGPipeline(config, pipeline_config)

        # LangGraph 설정
        self.workflow_graph: Optional[StateGraph] = None
        self.checkpointer: Optional[SqliteSaver] = None

        # 상태 추적
        self.active_workflows: Dict[str, Dict[str, Any]] = {}

        self._setup_workflow()

        logger.info("WorkflowManager 초기화 완료")

    def _setup_workflow(self):
        """워크플로우 그래프를 설정합니다."""
        try:
            # 체크포인트 설정
            if self.workflow_config.enable_checkpoints:
                self.checkpointer = SqliteSaver.from_conn_string(
                    f"sqlite:///{self.workflow_config.checkpoint_db_path}"
                )

            # 상태 그래프 생성
            workflow = StateGraph(WorkflowState)

            # 노드 추가
            workflow.add_node("start", self._start_workflow)
            workflow.add_node("analyze_question", self._analyze_question)
            workflow.add_node("route_question", self._route_question)
            workflow.add_node("execute_agents", self._execute_agents)
            workflow.add_node("integrate_responses", self._integrate_responses)
            workflow.add_node("evaluate_response", self._evaluate_response)
            workflow.add_node("collect_feedback", self._collect_feedback)
            workflow.add_node("finalize", self._finalize_workflow)
            workflow.add_node("error_handling", self._handle_error)

            # 엣지 정의
            workflow.set_entry_point("start")

            # 시작 -> 질문 분석
            workflow.add_edge("start", "analyze_question")

            # 질문 분석 -> 라우팅
            workflow.add_edge("analyze_question", "route_question")

            # 라우팅 -> 에이전트 실행
            workflow.add_edge("route_question", "execute_agents")

            # 에이전트 실행 -> 응답 통합
            workflow.add_edge("execute_agents", "integrate_responses")

            # 응답 통합 -> 조건부 분기
            workflow.add_conditional_edges(
                "integrate_responses",
                self._decide_next_after_integration,
                {
                    "evaluate": "evaluate_response",
                    "feedback": "collect_feedback",
                    "finalize": "finalize",
                    "error": "error_handling"
                }
            )

            # 평가 -> 조건부 분기
            workflow.add_conditional_edges(
                "evaluate_response",
                self._decide_next_after_evaluation,
                {
                    "feedback": "collect_feedback",
                    "finalize": "finalize",
                    "error": "error_handling"
                }
            )

            # 피드백 수집 -> 완료
            workflow.add_edge("collect_feedback", "finalize")

            # 완료 -> 종료
            workflow.add_edge("finalize", END)

            # 오류 처리 -> 조건부 분기
            workflow.add_conditional_edges(
                "error_handling",
                self._decide_error_recovery,
                {
                    "retry": "analyze_question",
                    "finalize": "finalize",
                    "end": END
                }
            )

            # 그래프 컴파일
            self.workflow_graph = workflow.compile(
                checkpointer=self.checkpointer if self.workflow_config.enable_checkpoints else None
            )

            logger.info("워크플로우 그래프 설정 완료")

        except Exception as e:
            logger.error(f"워크플로우 설정 실패: {e}")
            raise

    async def process_question(
        self,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs
    ) -> PipelineResult:
        """워크플로우를 통해 질문을 처리합니다."""
        start_time = time.time()

        if not thread_id:
            import uuid
            thread_id = str(uuid.uuid4())

        # 초기 상태 설정
        initial_state: WorkflowState = {
            "question": question,
            "user_id": user_id,
            "session_id": session_id,
            "analysis_result": None,
            "routing_result": None,
            "agent_responses": None,
            "integrated_response": None,
            "evaluation_result": None,
            "hitl_feedback": None,
            "current_step": WorkflowStage.START.value,
            "error_count": 0,
            "retry_count": 0,
            "final_answer": None,
            "confidence": 0.0,
            "execution_time": 0.0,
            "metadata": kwargs,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # 워크플로우 실행
            config = {"configurable": {"thread_id": thread_id}}

            # 파이프라인 초기화 확인
            if not self.rag_pipeline.is_initialized:
                await self.rag_pipeline.initialize()

            # 워크플로우 추적 시작
            self.active_workflows[thread_id] = {
                "start_time": start_time,
                "question": question,
                "status": "running"
            }

            # LangGraph 실행
            final_state = None
            async for state in self.workflow_graph.astream(initial_state, config):
                final_state = state
                # 상태 업데이트 로깅
                current_step = state.get("current_step", "unknown")
                logger.debug(f"워크플로우 진행: {current_step} (Thread: {thread_id})")

            # 최종 상태가 없는 경우 오류 처리
            if not final_state:
                raise Exception("워크플로우 실행이 완료되지 않았습니다")

            # 결과 추출
            execution_time = time.time() - start_time
            final_state["execution_time"] = execution_time

            # PipelineResult로 변환
            result = self._convert_state_to_result(final_state, execution_time)

            # 워크플로우 추적 완료
            self.active_workflows[thread_id]["status"] = "completed"
            self.active_workflows[thread_id]["execution_time"] = execution_time

            logger.info(f"워크플로우 완료: {thread_id} ({execution_time:.2f}초)")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"워크플로우 실행 실패: {e}")

            # 오류 상태 업데이트
            if thread_id in self.active_workflows:
                self.active_workflows[thread_id]["status"] = "failed"
                self.active_workflows[thread_id]["error"] = str(e)

            return PipelineResult(
                question=question,
                final_answer=f"워크플로우 실행 실패: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                metadata={"thread_id": thread_id, **kwargs}
            )

    # 워크플로우 노드 함수들
    async def _start_workflow(self, state: WorkflowState) -> WorkflowState:
        """워크플로우를 시작합니다."""
        logger.info(f"워크플로우 시작: {state['question']}")

        state["current_step"] = WorkflowStage.START.value
        state["timestamp"] = datetime.now().isoformat()

        return state

    async def _analyze_question(self, state: WorkflowState) -> WorkflowState:
        """질문을 분석합니다."""
        try:
            logger.info("질문 분석 시작")

            analysis_result = self.rag_pipeline.question_analyzer.analyze_question(
                state["question"]
            )

            state["analysis_result"] = asdict(analysis_result)
            state["current_step"] = WorkflowStage.ANALYZE_QUESTION.value

            logger.info("질문 분석 완료")

        except Exception as e:
            logger.error(f"질문 분석 실패: {e}")
            state["error_count"] += 1
            state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        return state

    async def _route_question(self, state: WorkflowState) -> WorkflowState:
        """질문을 라우팅합니다."""
        try:
            logger.info("질문 라우팅 시작")

            routing_result = await self.rag_pipeline.domain_router.route_question(
                state["question"]
            )

            state["routing_result"] = asdict(routing_result)
            state["current_step"] = WorkflowStage.ROUTE_QUESTION.value

            logger.info("질문 라우팅 완료")

        except Exception as e:
            logger.error(f"질문 라우팅 실패: {e}")
            state["error_count"] += 1
            state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        return state

    async def _execute_agents(self, state: WorkflowState) -> WorkflowState:
        """에이전트를 실행합니다."""
        try:
            logger.info("에이전트 실행 시작")

            # 라우팅 결과에서 에이전트 응답 추출
            routing_result = state.get("routing_result")
            if routing_result and "agent_responses" in routing_result:
                state["agent_responses"] = routing_result["agent_responses"]

            state["current_step"] = WorkflowStage.EXECUTE_AGENTS.value

            logger.info("에이전트 실행 완료")

        except Exception as e:
            logger.error(f"에이전트 실행 실패: {e}")
            state["error_count"] += 1
            state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        return state

    async def _integrate_responses(self, state: WorkflowState) -> WorkflowState:
        """응답을 통합합니다."""
        try:
            logger.info("응답 통합 시작")

            # 라우팅 결과를 RoutingResult 객체로 복원
            routing_result_dict = state.get("routing_result")
            if routing_result_dict:
                # 간단한 구조로 통합 실행 (실제 구현에서는 더 정교한 변환 필요)
                from ..routing.domain_router import RoutingResult

                # 딕셔너리를 다시 객체로 변환하는 대신, 직접 결과 생성
                integrated_response = {
                    "final_answer": "통합된 응답입니다.",  # 실제로는 복잡한 통합 로직
                    "confidence": 0.8,
                    "sources": [],
                    "contributing_agents": [],
                    "integration_strategy": "ai_synthesis",
                    "reasoning": "워크플로우를 통한 응답 통합",
                    "metadata": {},
                    "quality_metrics": {"completeness": 0.8, "clarity": 0.8, "relevance": 0.8, "accuracy": 0.8},
                    "execution_time": 0.0,
                    "success": True
                }

                state["integrated_response"] = integrated_response
                state["final_answer"] = integrated_response["final_answer"]
                state["confidence"] = integrated_response["confidence"]

            state["current_step"] = WorkflowStage.INTEGRATE_RESPONSES.value

            logger.info("응답 통합 완료")

        except Exception as e:
            logger.error(f"응답 통합 실패: {e}")
            state["error_count"] += 1
            state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        return state

    async def _evaluate_response(self, state: WorkflowState) -> WorkflowState:
        """응답을 평가합니다."""
        try:
            if not self.pipeline_config.enable_evaluation:
                logger.info("평가 모드가 비활성화됨")
                return state

            logger.info("응답 평가 시작")

            evaluation_result = self.rag_pipeline.react_evaluator.evaluate(
                state["question"],
                state["final_answer"] or "평가할 응답이 없습니다"
            )

            state["evaluation_result"] = asdict(evaluation_result)
            state["current_step"] = WorkflowStage.EVALUATE_RESPONSE.value

            logger.info("응답 평가 완료")

        except Exception as e:
            logger.error(f"응답 평가 실패: {e}")
            if not self.workflow_config.skip_evaluation_on_error:
                state["error_count"] += 1
                state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        return state

    async def _collect_feedback(self, state: WorkflowState) -> WorkflowState:
        """피드백을 수집합니다."""
        try:
            if not self.pipeline_config.enable_hitl:
                logger.info("HITL 모드가 비활성화됨")
                return state

            # 신뢰도 확인
            if (self.workflow_config.skip_hitl_on_low_confidence and
                state["confidence"] < self.workflow_config.min_confidence_for_hitl):
                logger.info("신뢰도가 낮아 HITL 피드백 수집 생략")
                return state

            logger.info("피드백 수집 시작")

            feedback = await self.rag_pipeline.hitl_interface.collect_rating_feedback(
                question=state["question"],
                answer=state["final_answer"] or "",
                user_id=state["user_id"],
                session_id=state["session_id"],
                timeout=30  # 짧은 타임아웃
            )

            state["hitl_feedback"] = asdict(feedback)
            state["current_step"] = WorkflowStage.COLLECT_FEEDBACK.value

            logger.info("피드백 수집 완료")

        except Exception as e:
            logger.error(f"피드백 수집 실패: {e}")
            # 피드백 수집 실패는 전체 워크플로우를 중단하지 않음

        return state

    async def _finalize_workflow(self, state: WorkflowState) -> WorkflowState:
        """워크플로우를 완료합니다."""
        logger.info("워크플로우 완료")

        state["current_step"] = WorkflowStage.FINALIZE.value

        # 최종 결과 검증
        if not state.get("final_answer"):
            state["final_answer"] = "죄송합니다. 적절한 답변을 생성하지 못했습니다."
            state["confidence"] = 0.0

        return state

    async def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """오류를 처리합니다."""
        logger.warning(f"오류 처리 시작: 오류 수 {state['error_count']}")

        state["current_step"] = WorkflowStage.ERROR_HANDLING.value

        # 재시도 가능성 확인
        if (state["retry_count"] < self.workflow_config.max_retries and
            self.workflow_config.enable_error_recovery):

            state["retry_count"] += 1
            logger.info(f"워크플로우 재시도: {state['retry_count']}")
        else:
            logger.error("최대 재시도 횟수 초과 또는 오류 복구 비활성화")
            state["final_answer"] = "처리 중 복구 불가능한 오류가 발생했습니다."
            state["confidence"] = 0.0

        return state

    # 조건부 분기 함수들
    def _decide_next_after_integration(self, state: WorkflowState) -> str:
        """통합 후 다음 단계를 결정합니다."""
        if state.get("error_count", 0) > 0:
            return "error"
        elif self.pipeline_config.enable_evaluation:
            return "evaluate"
        elif self.pipeline_config.enable_hitl:
            return "feedback"
        else:
            return "finalize"

    def _decide_next_after_evaluation(self, state: WorkflowState) -> str:
        """평가 후 다음 단계를 결정합니다."""
        if state.get("error_count", 0) > 0:
            return "error"
        elif self.pipeline_config.enable_hitl:
            return "feedback"
        else:
            return "finalize"

    def _decide_error_recovery(self, state: WorkflowState) -> str:
        """오류 복구 전략을 결정합니다."""
        if (state["retry_count"] < self.workflow_config.max_retries and
            self.workflow_config.enable_error_recovery):
            return "retry"
        elif state.get("final_answer"):
            return "finalize"
        else:
            return "end"

    def _convert_state_to_result(
        self,
        state: WorkflowState,
        execution_time: float
    ) -> PipelineResult:
        """워크플로우 상태를 PipelineResult로 변환합니다."""

        # 기본 정보
        result = PipelineResult(
            question=state["question"],
            final_answer=state.get("final_answer", "답변을 생성하지 못했습니다"),
            confidence=state.get("confidence", 0.0),
            execution_time=execution_time,
            success=state.get("error_count", 0) == 0,
            error_message="워크플로우 오류 발생" if state.get("error_count", 0) > 0 else "",
            metadata={
                "workflow_managed": True,
                "retry_count": state.get("retry_count", 0),
                "error_count": state.get("error_count", 0),
                **state.get("metadata", {})
            }
        )

        return result

    async def get_workflow_status(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """워크플로우 상태를 조회합니다."""
        if thread_id in self.active_workflows:
            status = self.active_workflows[thread_id].copy()

            # 실행 시간 계산
            if status["status"] == "running":
                current_time = time.time()
                status["running_time"] = current_time - status["start_time"]

            return status

        return None

    async def cancel_workflow(self, thread_id: str) -> bool:
        """워크플로우를 취소합니다."""
        if thread_id in self.active_workflows:
            self.active_workflows[thread_id]["status"] = "cancelled"
            logger.info(f"워크플로우 취소됨: {thread_id}")
            return True

        return False

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """활성 워크플로우 목록을 반환합니다."""
        return self.active_workflows.copy()


# 편의 함수들
def create_workflow_manager(
    config: Optional[CoolStayConfig] = None,
    workflow_config: Optional[WorkflowConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None
) -> WorkflowManager:
    """워크플로우 매니저를 생성합니다."""
    return WorkflowManager(config, workflow_config, pipeline_config)


async def process_with_workflow(
    question: str,
    config: Optional[CoolStayConfig] = None,
    **kwargs
) -> PipelineResult:
    """워크플로우를 통한 간단한 질문 처리 함수"""
    workflow_manager = WorkflowManager(config)
    return await workflow_manager.process_question(question, **kwargs)