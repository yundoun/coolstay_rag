"""
CoolStay RAG 통합 파이프라인 모듈

모든 RAG 컴포넌트를 통합하여 완전한 엔드투엔드 파이프라인을 제공하는 모듈입니다.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime

from ..core.config import CoolStayConfig
from ..data import ChromaManager
from ..agents import (
    BaseRAGAgent, CorrectiveRAGAgent, WebSearchAgent, AgentResponse,
    create_all_domain_agents, create_all_corrective_agents, create_web_agent
)
from ..routing import (
    QuestionAnalyzer, DomainRouter, ResponseIntegrator,
    RoutingResult, IntegratedResponse
)
from ..evaluation import (
    ReActEvaluationAgent, HITLInterface,
    ReActEvaluationResult, HumanFeedback, FeedbackType
)

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """파이프라인 모드"""
    BASIC = "basic"                    # 기본 RAG
    CORRECTIVE = "corrective"          # 교정 RAG
    MULTI_AGENT = "multi_agent"        # 다중 에이전트
    FULL_PIPELINE = "full_pipeline"    # 전체 파이프라인
    EVALUATION_MODE = "evaluation"     # 평가 모드


class PipelineStage(Enum):
    """파이프라인 단계"""
    INITIALIZATION = "initialization"
    QUESTION_ANALYSIS = "question_analysis"
    ROUTING = "routing"
    AGENT_EXECUTION = "agent_execution"
    RESPONSE_INTEGRATION = "response_integration"
    EVALUATION = "evaluation"
    HITL_FEEDBACK = "hitl_feedback"
    COMPLETION = "completion"


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    question: str
    final_answer: str
    confidence: float
    execution_time: float

    # 상세 결과
    routing_result: Optional[RoutingResult] = None
    integrated_response: Optional[IntegratedResponse] = None
    evaluation_result: Optional[ReActEvaluationResult] = None
    hitl_feedback: Optional[HumanFeedback] = None

    # 메타데이터
    pipeline_mode: PipelineMode = PipelineMode.BASIC
    stages_completed: List[PipelineStage] = None
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.stages_completed is None:
            self.stages_completed = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    mode: PipelineMode = PipelineMode.FULL_PIPELINE
    enable_evaluation: bool = True
    enable_hitl: bool = False
    enable_web_search: bool = True
    enable_corrective_rag: bool = True

    # 성능 설정
    max_concurrent_agents: int = 5
    agent_timeout: int = 30
    evaluation_timeout: int = 15
    hitl_timeout: int = 60

    # 품질 임계값
    min_confidence_threshold: float = 0.6
    min_quality_threshold: float = 0.7
    enable_quality_checks: bool = True


class IntegratedRAGPipeline:
    """
    통합 RAG 파이프라인

    모든 RAG 컴포넌트를 연결하여 완전한 질문-답변 시스템을 제공합니다.
    """

    def __init__(
        self,
        config: Optional[CoolStayConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            config: CoolStay 설정 객체
            pipeline_config: 파이프라인 설정 객체
        """
        self.config = config or CoolStayConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()

        # 컴포넌트 초기화
        self.chroma_manager: Optional[ChromaManager] = None
        self.question_analyzer: Optional[QuestionAnalyzer] = None
        self.domain_router: Optional[DomainRouter] = None
        self.response_integrator: Optional[ResponseIntegrator] = None
        self.react_evaluator: Optional[ReActEvaluationAgent] = None
        self.hitl_interface: Optional[HITLInterface] = None
        self.corrective_agents: Optional[Dict[str, CorrectiveRAGAgent]] = None

        # 상태 추적
        self.is_initialized = False
        self.initialization_error = None

        logger.info("IntegratedRAGPipeline 생성 완료")

    async def initialize(self) -> bool:
        """파이프라인을 초기화합니다."""
        try:
            logger.info("파이프라인 초기화 시작")

            # 디버깅: config 객체 속성 확인
            logger.info(f"DEBUG: config 타입: {type(self.config)}")
            logger.info(f"DEBUG: config 속성들: {dir(self.config)}")

            # 1. ChromaDB 관리자 초기화
            logger.info("ChromaDB 관리자 초기화 시작...")
            try:
                self.chroma_manager = ChromaManager()
                logger.info("ChromaDB 관리자 초기화 완료")
            except Exception as e:
                logger.error(f"ChromaDB 관리자 초기화 실패: {e}")
                raise

            # 2. 질문 분석기 초기화
            logger.info("질문 분석기 초기화 시작...")
            try:
                self.question_analyzer = QuestionAnalyzer()
                logger.info("질문 분석기 초기화 완료")
            except Exception as e:
                logger.error(f"질문 분석기 초기화 실패: {e}")
                raise

            # 3. 도메인 라우터 초기화 및 에이전트 로드
            logger.info("도메인 라우터 초기화 시작...")
            try:
                self.domain_router = DomainRouter(self.config)
                await self.domain_router.initialize_agents()
                logger.info("도메인 라우터 및 에이전트 초기화 완료")

                # Corrective RAG 에이전트 초기화 (품질 검증용)
                if self.pipeline_config.enable_corrective_rag:
                    logger.info("Corrective RAG 에이전트 초기화...")
                    self.corrective_agents = create_all_corrective_agents(
                        llm=None,
                        chroma_manager=self.chroma_manager
                    )
                    logger.info(f"Corrective RAG 에이전트 {len(self.corrective_agents)}개 생성 완료")
            except Exception as e:
                logger.error(f"도메인 라우터 초기화 실패: {e}")
                import traceback
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
                raise

            # 4. 응답 통합기 초기화
            logger.info("응답 통합기 초기화 시작...")
            try:
                self.response_integrator = ResponseIntegrator(self.config)
                logger.info("응답 통합기 초기화 완료")
            except Exception as e:
                logger.error(f"응답 통합기 초기화 실패: {e}")
                raise

            # 5. 평가 시스템 초기화 (선택사항)
            if self.pipeline_config.enable_evaluation:
                logger.info("ReAct 평가 시스템 초기화 시작...")
                try:
                    self.react_evaluator = ReActEvaluationAgent(self.config)
                    logger.info("ReAct 평가 시스템 초기화 완료")
                except Exception as e:
                    logger.error(f"ReAct 평가 시스템 초기화 실패: {e}")
                    raise

            # 6. HITL 인터페이스 초기화 (선택사항)
            if self.pipeline_config.enable_hitl:
                logger.info("HITL 인터페이스 초기화 시작...")
                try:
                    self.hitl_interface = HITLInterface(self.config)
                    logger.info("HITL 인터페이스 초기화 완료")
                except Exception as e:
                    logger.error(f"HITL 인터페이스 초기화 실패: {e}")
                    raise

            self.is_initialized = True
            logger.info("파이프라인 초기화 완료")
            return True

        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"파이프라인 초기화 실패: {e}")
            import traceback
            logger.error(f"전체 스택 트레이스: {traceback.format_exc()}")
            return False

    async def process_question(
        self,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> PipelineResult:
        """질문을 처리하고 완전한 결과를 반환합니다."""
        logger.info(f"=== process_question 시작: {question} ===")
        start_time = time.time()
        stages_completed = []

        try:
            # 초기화 확인
            logger.info(f"초기화 상태: {self.is_initialized}")
            if not self.is_initialized:
                logger.info("파이프라인 초기화 필요")
                if not await self.initialize():
                    logger.error(f"초기화 실패: {self.initialization_error}")
                    return self._create_error_result(
                        question,
                        f"파이프라인 초기화 실패: {self.initialization_error}",
                        time.time() - start_time
                    )

            stages_completed.append(PipelineStage.INITIALIZATION)

            # 1. 질문 분석 단계
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [1단계] 질문 분석 (Question Analysis)                         ║
╚══════════════════════════════════════════════════════════════╝
📝 원본 질문: {question}
""")
            try:
                analysis_result = self.question_analyzer.analyze_question(question)
                print(f"""
✅ 분석 완료:
   - 질문 타입: {analysis_result.question_type.value if hasattr(analysis_result, 'question_type') else 'N/A'}
   - 주요 도메인: {analysis_result.primary_domains if hasattr(analysis_result, 'primary_domains') else 'N/A'}
   - 보조 도메인: {analysis_result.secondary_domains if hasattr(analysis_result, 'secondary_domains') else 'N/A'}
   - 복잡도: {analysis_result.complexity if hasattr(analysis_result, 'complexity') else 'N/A'}
   - 신뢰도: {analysis_result.confidence_score if hasattr(analysis_result, 'confidence_score') else 'N/A'}
   - 웹 검색 필요: {analysis_result.requires_web_search if hasattr(analysis_result, 'requires_web_search') else 'N/A'}
   - 의도: {analysis_result.intent if hasattr(analysis_result, 'intent') else 'N/A'}
   - 키워드: {analysis_result.keywords if hasattr(analysis_result, 'keywords') else 'N/A'}
""")
            except Exception as e:
                logger.error(f"질문 분석 실패: {e}")
                import traceback
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
                raise
            stages_completed.append(PipelineStage.QUESTION_ANALYSIS)

            # 2. 라우팅 및 에이전트 실행 단계
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [2단계] 라우팅 결정 (Routing Decision)                        ║
╚══════════════════════════════════════════════════════════════╝
🎯 의사결정 과정:
   - 분석된 도메인 수: {len(analysis_result.primary_domains) if hasattr(analysis_result, 'primary_domains') else 0}
   - 복잡도 기반 전략 선택 중...
""")
            routing_result = await self.domain_router.route_question(question)

            print(f"""
✅ 라우팅 완료:
   - 라우팅 전략: {routing_result.routing_decision.strategy.value if hasattr(routing_result, 'routing_decision') else 'N/A'}
   - 선택된 에이전트: {routing_result.routing_decision.primary_agents if hasattr(routing_result, 'routing_decision') else 'N/A'}
   - 활성화된 에이전트 수: {len(routing_result.agent_responses) if hasattr(routing_result, 'agent_responses') else 0}
   - 의사결정 이유: {routing_result.routing_decision.reasoning if hasattr(routing_result, 'routing_decision') else 'N/A'}
""")
            stages_completed.append(PipelineStage.ROUTING)
            stages_completed.append(PipelineStage.AGENT_EXECUTION)

            # 3. 응답 통합 단계
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [3단계] 응답 통합 (Response Integration)                      ║
╚══════════════════════════════════════════════════════════════╝
🔄 통합 프로세스:
   - 수집된 응답 수: {len(routing_result.agent_responses) if hasattr(routing_result, 'agent_responses') else 0}
   - 응답 에이전트: {list(routing_result.agent_responses.keys()) if hasattr(routing_result, 'agent_responses') else []}
""")
            integrated_response = self.response_integrator.integrate_responses(routing_result)

            print(f"""
✅ 통합 완료:
   - 통합 전략: {integrated_response.integration_strategy.value if hasattr(integrated_response, 'integration_strategy') else 'N/A'}
   - 최종 신뢰도: {integrated_response.confidence if hasattr(integrated_response, 'confidence') else 'N/A'}
   - 품질 메트릭스:
     • 완전성: {integrated_response.quality_metrics.get('completeness', 'N/A') if hasattr(integrated_response, 'quality_metrics') else 'N/A'}
     • 명확성: {integrated_response.quality_metrics.get('clarity', 'N/A') if hasattr(integrated_response, 'quality_metrics') else 'N/A'}
     • 관련성: {integrated_response.quality_metrics.get('relevance', 'N/A') if hasattr(integrated_response, 'quality_metrics') else 'N/A'}
     • 정확성: {integrated_response.quality_metrics.get('accuracy', 'N/A') if hasattr(integrated_response, 'quality_metrics') else 'N/A'}
""")
            stages_completed.append(PipelineStage.RESPONSE_INTEGRATION)

            # 4. 품질 검증
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [4단계] 품질 검증 (Quality Validation)                       ║
╚══════════════════════════════════════════════════════════════╝
🔍 품질 검증 설정:
   - 품질 검증 활성화: {self.pipeline_config.enable_quality_checks}
   - Corrective RAG 활성화: {self.pipeline_config.enable_corrective_rag}
   - 최소 신뢰도 임계값: {getattr(self.pipeline_config, 'min_confidence_threshold', '설정되지 않음')}
   - 최소 품질 임계값: {getattr(self.pipeline_config, 'min_quality_threshold', '설정되지 않음')}
""")

            if self.pipeline_config.enable_quality_checks:
                quality_result = self._validate_response_quality(integrated_response)
                print(f"""
🎯 품질 검증 결과:
   - 현재 신뢰도: {integrated_response.confidence}
   - 품질 메트릭스: {integrated_response.quality_metrics}
   - 답변 길이: {len(integrated_response.final_answer.strip())}자
   - 품질 검증 통과: {'✅ 통과' if quality_result else '❌ 미달'}
""")
                if not quality_result:
                    logger.warning("응답 품질이 임계값 미달")
                    print("🔄 품질 개선이 필요합니다. Corrective RAG 메커니즘 실행 중...")

                    # Corrective RAG를 통한 품질 개선 시도
                    if self.pipeline_config.enable_corrective_rag and self.corrective_agents:
                        print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [품질 개선] Corrective RAG 재처리 시작                        ║
╚══════════════════════════════════════════════════════════════╝
🎯 품질 미달 원인:
   - 현재 신뢰도: {integrated_response.confidence} < {self.pipeline_config.min_confidence_threshold}
   - 품질 점수: {sum(integrated_response.quality_metrics.values())/len(integrated_response.quality_metrics):.2f} < {self.pipeline_config.min_quality_threshold}
""")

                        # 가장 낮은 품질의 도메인을 선택하여 Corrective RAG 적용
                        # 또는 무작위로 하나 선택 (모두 동일한 경우)
                        lowest_quality_domain = None
                        lowest_score = 1.0

                        print(f"🔍 도메인별 응답 상태:")
                        for agent_name, response in routing_result.agent_responses.items():
                            score = getattr(response, 'confidence_score', 0.8)  # 기본값 0.8
                            print(f"   - {agent_name}: 신뢰도 {score}")
                            if score < lowest_score:
                                lowest_score = score
                                lowest_quality_domain = agent_name

                        # 모든 도메인이 동일한 점수면 첫 번째 도메인 선택
                        if lowest_quality_domain is None and routing_result.agent_responses:
                            lowest_quality_domain = list(routing_result.agent_responses.keys())[0]
                            print(f"ℹ️  모든 도메인이 동일한 품질 - {lowest_quality_domain} 선택")

                        # 모든 개별 도메인의 품질이 높은 경우 - 통합 전략 변경
                        all_high_quality = all(
                            getattr(response, 'confidence_score', 0.8) >= 0.95
                            for response in routing_result.agent_responses.values()
                        )

                        if all_high_quality:
                            print("📊 모든 도메인이 고품질 - 통합 전략 변경 시도")

                            # 다른 통합 전략 시도
                            original_strategy = integrated_response.integration_strategy
                            print(f"   - 기존 전략: {original_strategy.value}")

                            # 전략 순서: AI_SYNTHESIS → CONSENSUS_BUILDING → WEIGHTED_MERGE
                            from ..routing.response_integrator import IntegrationStrategy
                            alternate_strategies = [
                                IntegrationStrategy.CONSENSUS_BUILDING,
                                IntegrationStrategy.WEIGHTED_MERGE,
                                IntegrationStrategy.CONFIDENCE_RANKING
                            ]

                            for strategy in alternate_strategies:
                                if strategy != original_strategy:
                                    print(f"   - 대안 전략 시도: {strategy.value}")

                                    # 통합 전략 강제 변경
                                    routing_result.routing_decision.strategy = strategy
                                    integrated_response = self.response_integrator.integrate_responses(routing_result)

                                    if self._validate_response_quality(integrated_response):
                                        print(f"✅ {strategy.value} 전략으로 품질 개선 성공!")
                                        break
                                    else:
                                        print(f"   ↳ {strategy.value} 전략도 품질 미달")
                            else:
                                print("⚠️  모든 통합 전략 시도 후에도 품질 미달 - 현재 최선의 결과 사용")

                        elif lowest_quality_domain and lowest_quality_domain in self.corrective_agents:
                            # 기존 로직: 품질이 낮은 도메인 개선
                            print(f"🔧 {lowest_quality_domain} 도메인 Corrective RAG 적용")
                            corrective_agent = self.corrective_agents[lowest_quality_domain]

                            # Corrective RAG 실행
                            corrected_response = corrective_agent.corrective_query(question)

                            # 개선된 응답으로 교체
                            if corrected_response.quality_assessment.confidence_score > lowest_score:
                                print(f"✅ 품질 개선 성공! ({lowest_score:.2f} → {corrected_response.quality_assessment.confidence_score:.2f})")

                                # 응답 업데이트
                                routing_result.agent_responses[lowest_quality_domain] = AgentResponse(
                                    answer=corrected_response.final_answer,
                                    source_documents=corrected_response.source_documents,
                                    domain=corrected_response.domain,
                                    agent_name=corrected_response.agent_name,
                                    confidence_score=corrected_response.quality_assessment.confidence_score
                                )

                                # 통합 재실행
                                print("🔄 개선된 응답으로 통합 재실행...")
                                integrated_response = self.response_integrator.integrate_responses(routing_result)

                                # 재검증
                                if self._validate_response_quality(integrated_response):
                                    print("✅ 품질 개선 후 검증 통과!")
                                else:
                                    print("⚠️  품질 개선 후에도 미달 - 현재 결과로 진행")
                            else:
                                print("⚠️  Corrective RAG 후에도 품질 개선 미미")
                        else:
                            print("⚠️  Corrective RAG 적용 가능한 도메인 없음")
                    else:
                        print("⚠️  Corrective RAG가 비활성화되어 있거나 사용 불가")
                else:
                    print("✅ 품질 검증을 성공적으로 통과했습니다.")
            else:
                print("⚠️  품질 검증이 비활성화되어 있습니다.")

            # 5. 평가 단계 (선택사항)
            evaluation_result = None
            if self.pipeline_config.enable_evaluation and self.react_evaluator:
                logger.info("ReAct 평가 시작")
                try:
                    evaluation_result = self.react_evaluator.evaluate(
                        question, integrated_response.final_answer
                    )
                    stages_completed.append(PipelineStage.EVALUATION)
                except Exception as e:
                    logger.error(f"평가 실패: {e}")

            # 6. HITL 피드백 단계 (선택사항)
            hitl_feedback = None
            if self.pipeline_config.enable_hitl and self.hitl_interface:
                logger.info("HITL 피드백 수집 시작")
                try:
                    hitl_feedback = await self._collect_hitl_feedback(
                        question, integrated_response.final_answer,
                        user_id, session_id
                    )
                    stages_completed.append(PipelineStage.HITL_FEEDBACK)
                except Exception as e:
                    logger.error(f"HITL 피드백 수집 실패: {e}")

            stages_completed.append(PipelineStage.COMPLETION)
            execution_time = time.time() - start_time

            # 최종 결과 생성
            result = PipelineResult(
                question=question,
                final_answer=integrated_response.final_answer,
                confidence=integrated_response.confidence,
                execution_time=execution_time,
                routing_result=routing_result,
                integrated_response=integrated_response,
                evaluation_result=evaluation_result,
                hitl_feedback=hitl_feedback,
                pipeline_mode=self.pipeline_config.mode,
                stages_completed=stages_completed,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'analysis_result': asdict(analysis_result),
                    **kwargs
                }
            )

            print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [최종] 파이프라인 처리 완료                                   ║
╚══════════════════════════════════════════════════════════════╝
⭐ 최종 결과:
   - 총 처리 시간: {execution_time:.2f}초
   - 완료된 단계: {[stage.value for stage in stages_completed]}
   - 최종 신뢰도: {integrated_response.confidence if integrated_response else 'N/A'}
   - 답변 길이: {len(integrated_response.final_answer) if integrated_response else 0}자
   - 성공 여부: ✅ 성공
═══════════════════════════════════════════════════════════════
""")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"질문 처리 실패: {e}")

            return PipelineResult(
                question=question,
                final_answer=f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                execution_time=execution_time,
                pipeline_mode=self.pipeline_config.mode,
                stages_completed=stages_completed,
                success=False,
                error_message=str(e),
                metadata={'user_id': user_id, 'session_id': session_id, **kwargs}
            )

    async def process_batch(
        self,
        questions: List[str],
        **kwargs
    ) -> List[PipelineResult]:
        """여러 질문을 일괄 처리합니다."""
        logger.info(f"일괄 처리 시작: {len(questions)}개 질문")

        # 병렬 처리를 위한 태스크 생성
        max_concurrent = min(
            self.pipeline_config.max_concurrent_agents,
            len(questions)
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single_question(question: str, index: int) -> PipelineResult:
            async with semaphore:
                return await self.process_question(
                    question,
                    batch_index=index,
                    **kwargs
                )

        # 병렬 실행
        tasks = [
            process_single_question(question, i)
            for i, question in enumerate(questions)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = self._create_error_result(
                    questions[i],
                    f"일괄 처리 중 오류: {str(result)}",
                    0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        logger.info(f"일괄 처리 완료: {len(processed_results)}개 결과")
        return processed_results

    def _validate_response_quality(self, response: IntegratedResponse) -> bool:
        """응답 품질을 검증합니다."""
        # 신뢰도 검증
        if response.confidence < self.pipeline_config.min_confidence_threshold:
            return False

        # 품질 메트릭 검증
        if response.quality_metrics:
            avg_quality = sum(response.quality_metrics.values()) / len(response.quality_metrics)
            if avg_quality < self.pipeline_config.min_quality_threshold:
                return False

        # 답변 길이 검증
        if len(response.final_answer.strip()) < 10:
            return False

        return True

    async def _collect_hitl_feedback(
        self,
        question: str,
        answer: str,
        user_id: Optional[str],
        session_id: Optional[str]
    ) -> Optional[HumanFeedback]:
        """HITL 피드백을 수집합니다."""
        try:
            feedback = await self.hitl_interface.collect_rating_feedback(
                question=question,
                answer=answer,
                user_id=user_id,
                session_id=session_id,
                timeout=self.pipeline_config.hitl_timeout
            )
            return feedback
        except Exception as e:
            logger.error(f"HITL 피드백 수집 실패: {e}")
            return None

    def _create_error_result(
        self,
        question: str,
        error_message: str,
        execution_time: float
    ) -> PipelineResult:
        """오류 결과를 생성합니다."""
        return PipelineResult(
            question=question,
            final_answer=f"처리 중 오류가 발생했습니다: {error_message}",
            confidence=0.0,
            execution_time=execution_time,
            pipeline_mode=self.pipeline_config.mode,
            stages_completed=[PipelineStage.INITIALIZATION],
            success=False,
            error_message=error_message
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태를 반환합니다."""
        status = {
            'initialized': self.is_initialized,
            'pipeline_mode': self.pipeline_config.mode.value,
            'components': {},
            'configuration': {
                'enable_evaluation': self.pipeline_config.enable_evaluation,
                'enable_hitl': self.pipeline_config.enable_hitl,
                'enable_web_search': self.pipeline_config.enable_web_search,
                'enable_corrective_rag': self.pipeline_config.enable_corrective_rag
            }
        }

        if self.is_initialized:
            # 컴포넌트 상태
            status['components']['chroma_manager'] = self.chroma_manager is not None
            status['components']['question_analyzer'] = self.question_analyzer is not None
            status['components']['domain_router'] = self.domain_router is not None
            status['components']['response_integrator'] = self.response_integrator is not None
            status['components']['react_evaluator'] = self.react_evaluator is not None
            status['components']['hitl_interface'] = self.hitl_interface is not None

            # 에이전트 상태
            if self.domain_router:
                status['agents'] = self.domain_router.get_agent_status()

        else:
            status['initialization_error'] = self.initialization_error

        return status

    async def run_health_check(self) -> Dict[str, Any]:
        """시스템 헬스체크를 실행합니다."""
        health_status = {
            'overall_health': 'unknown',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            # 1. 초기화 상태 확인
            health_status['checks']['initialization'] = {
                'status': 'pass' if self.is_initialized else 'fail',
                'message': self.initialization_error if not self.is_initialized else 'OK'
            }

            if not self.is_initialized:
                health_status['overall_health'] = 'fail'
                return health_status

            # 2. 간단한 질문 처리 테스트
            test_question = "꿀스테이 RAG 시스템 테스트"
            test_start = time.time()

            try:
                # 타임아웃을 짧게 설정하여 빠른 체크
                test_result = await asyncio.wait_for(
                    self.process_question(test_question),
                    timeout=10.0
                )

                test_duration = time.time() - test_start

                health_status['checks']['processing'] = {
                    'status': 'pass' if test_result.success else 'fail',
                    'message': test_result.error_message if not test_result.success else 'OK',
                    'duration': test_duration
                }

            except asyncio.TimeoutError:
                health_status['checks']['processing'] = {
                    'status': 'fail',
                    'message': '처리 시간 초과',
                    'duration': time.time() - test_start
                }

            # 3. 전체 상태 결정
            failed_checks = [
                check for check in health_status['checks'].values()
                if check['status'] == 'fail'
            ]

            if not failed_checks:
                health_status['overall_health'] = 'pass'
            elif len(failed_checks) == len(health_status['checks']):
                health_status['overall_health'] = 'fail'
            else:
                health_status['overall_health'] = 'warn'

        except Exception as e:
            health_status['overall_health'] = 'fail'
            health_status['checks']['system'] = {
                'status': 'fail',
                'message': f'헬스체크 실행 오류: {str(e)}'
            }

        return health_status

    async def shutdown(self):
        """파이프라인을 정리하고 종료합니다."""
        logger.info("파이프라인 종료 시작")

        try:
            # HITL 세션 정리
            if self.hitl_interface:
                for session_id in list(self.hitl_interface.active_sessions.keys()):
                    await self.hitl_interface.end_hitl_session(session_id)

            # ChromaDB 연결 정리
            if self.chroma_manager:
                # ChromaDB는 자동으로 연결이 정리됨
                pass

            self.is_initialized = False
            logger.info("파이프라인 종료 완료")

        except Exception as e:
            logger.error(f"파이프라인 종료 중 오류: {e}")


# 편의 함수들
def create_pipeline(
    config: Optional[CoolStayConfig] = None,
    pipeline_config: Optional[PipelineConfig] = None
) -> IntegratedRAGPipeline:
    """파이프라인을 생성합니다."""
    return IntegratedRAGPipeline(config, pipeline_config)


async def process_question_simple(
    question: str,
    config: Optional[CoolStayConfig] = None
) -> PipelineResult:
    """간단한 질문 처리 함수"""
    pipeline = IntegratedRAGPipeline(config)
    await pipeline.initialize()
    return await pipeline.process_question(question)


def analyze_pipeline_performance(results: List[PipelineResult]) -> Dict[str, Any]:
    """파이프라인 성능을 분석합니다."""
    if not results:
        return {'message': '분석할 결과가 없습니다.'}

    # 성공률
    successful_results = [r for r in results if r.success]
    success_rate = len(successful_results) / len(results)

    # 실행 시간 통계
    execution_times = [r.execution_time for r in results]
    avg_execution_time = sum(execution_times) / len(execution_times)

    # 신뢰도 통계
    confidences = [r.confidence for r in successful_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # 단계 완료 통계
    stage_completion = {}
    for result in results:
        for stage in result.stages_completed:
            stage_name = stage.value
            stage_completion[stage_name] = stage_completion.get(stage_name, 0) + 1

    return {
        'total_results': len(results),
        'success_rate': success_rate,
        'average_execution_time': avg_execution_time,
        'execution_time_range': (min(execution_times), max(execution_times)),
        'average_confidence': avg_confidence,
        'stage_completion_rates': {
            stage: count / len(results)
            for stage, count in stage_completion.items()
        },
        'error_analysis': {
            'total_errors': len(results) - len(successful_results),
            'common_errors': [r.error_message for r in results if not r.success][:5]
        }
    }