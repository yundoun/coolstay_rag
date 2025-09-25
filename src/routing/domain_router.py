"""
CoolStay RAG 도메인 라우터 모듈

질문 분석 결과를 바탕으로 적절한 에이전트를 선택하고 라우팅하는 모듈입니다.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.config import CoolStayConfig
from ..agents import (
    BaseRAGAgent, CorrectiveRAGAgent, WebSearchAgent,
    AgentResponse, AgentStatus,
    create_agent, create_corrective_agent, create_web_agent,
    create_all_domain_agents, create_all_corrective_agents
)
from .question_analyzer import QuestionAnalyzer, QuestionAnalysis, QuestionType

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """라우팅 전략"""
    SINGLE = "single"          # 단일 에이전트
    MULTI_PARALLEL = "parallel"    # 병렬 다중 에이전트
    MULTI_SEQUENTIAL = "sequential"  # 순차 다중 에이전트
    HYBRID = "hybrid"          # 하이브리드 (내부 + 웹)


@dataclass
class RoutingDecision:
    """라우팅 결정 결과"""
    strategy: RoutingStrategy
    primary_agents: List[str]  # 주요 에이전트들
    secondary_agents: List[str] = None  # 보조 에이전트들
    web_search_required: bool = False
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class RoutingResult:
    """라우팅 실행 결과"""
    question: str
    routing_decision: RoutingDecision
    agent_responses: Dict[str, AgentResponse]
    web_response: Optional[AgentResponse] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""


class DomainRouter:
    """
    도메인 라우터

    질문 분석 결과를 바탕으로 적절한 에이전트를 선택하고,
    효율적인 라우팅 전략을 결정하여 질문을 처리합니다.
    """

    def __init__(self, config: Optional[CoolStayConfig] = None):
        """
        Args:
            config: CoolStay 설정 객체
        """
        self.config = config or CoolStayConfig()
        self.question_analyzer = QuestionAnalyzer(config)

        # 에이전트 초기화
        self.domain_agents: Dict[str, BaseRAGAgent] = {}
        self.corrective_agents: Dict[str, CorrectiveRAGAgent] = {}
        self.web_agent: Optional[WebSearchAgent] = None

        # 라우팅 규칙 설정
        self._setup_routing_rules()

        logger.info("DomainRouter 초기화 완료")

    def _setup_routing_rules(self):
        """라우팅 규칙 설정"""
        # 도메인별 키워드 매핑
        self.domain_keywords = {
            'hr_policy': [
                '인사', '채용', '직원', '급여', '연차', '휴가', '복지',
                '평가', '승진', '교육', '근무', '출퇴근', 'HR'
            ],
            'tech_policy': [
                '기술', '개발', '프로그래밍', '시스템', '서버', '데이터베이스',
                '보안', '네트워크', '클라우드', 'IT', '소프트웨어'
            ],
            'architecture': [
                '아키텍처', '설계', '구조', '시스템', '컴포넌트', '모듈',
                '패턴', '프레임워크', '인프라', '플랫폼'
            ],
            'component': [
                '컴포넌트', '모듈', '라이브러리', '패키지', '인터페이스',
                'API', '서비스', '기능', '구현'
            ],
            'deployment': [
                '배포', '릴리즈', '운영', '배치', '환경', 'CI/CD',
                'DevOps', '모니터링', '로그', '성능'
            ],
            'development': [
                '개발', '코딩', '프로그래밍', '테스트', '디버깅',
                '버전', 'Git', '협업', '리뷰', '문서화'
            ],
            'business_policy': [
                '비즈니스', '정책', '규정', '절차', '승인', '계약',
                '고객', '서비스', '마케팅', '영업', '재무'
            ]
        }

        # 질문 유형별 라우팅 전략
        self.type_strategy_mapping = {
            QuestionType.FACTUAL: RoutingStrategy.SINGLE,
            QuestionType.PROCEDURAL: RoutingStrategy.SINGLE,
            QuestionType.COMPARATIVE: RoutingStrategy.MULTI_PARALLEL,
            QuestionType.ANALYTICAL: RoutingStrategy.MULTI_PARALLEL,
            QuestionType.CREATIVE: RoutingStrategy.HYBRID,
            QuestionType.TROUBLESHOOTING: RoutingStrategy.MULTI_SEQUENTIAL
        }

    async def initialize_agents(self):
        """에이전트들을 초기화합니다."""
        try:
            # 도메인 에이전트 생성
            self.domain_agents = create_all_domain_agents(self.config)
            logger.info(f"도메인 에이전트 {len(self.domain_agents)}개 초기화 완료")

            # Corrective RAG 에이전트 생성
            self.corrective_agents = create_all_corrective_agents(self.config)
            logger.info(f"Corrective RAG 에이전트 {len(self.corrective_agents)}개 초기화 완료")

            # 웹 에이전트 생성
            self.web_agent = create_web_agent(self.config)
            logger.info("웹 에이전트 초기화 완료")

            return True

        except Exception as e:
            logger.error(f"에이전트 초기화 실패: {e}")
            return False

    def analyze_and_route(self, question: str) -> RoutingDecision:
        """질문을 분석하고 라우팅 결정을 만듭니다."""
        try:
            # 질문 분석
            analysis = self.question_analyzer.analyze_question(question)

            # 라우팅 전략 결정
            strategy = self._determine_strategy(analysis)

            # 에이전트 선택
            primary_agents = self._select_primary_agents(analysis)
            secondary_agents = self._select_secondary_agents(analysis, primary_agents)

            # 웹 검색 필요성 판단
            web_search_required = self._should_use_web_search(analysis)

            # 신뢰도 계산
            confidence = self._calculate_confidence(analysis, primary_agents)

            # 추론 과정 설명
            reasoning = self._generate_reasoning(analysis, strategy, primary_agents)

            return RoutingDecision(
                strategy=strategy,
                primary_agents=primary_agents,
                secondary_agents=secondary_agents,
                web_search_required=web_search_required,
                confidence=confidence,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"라우팅 분석 실패: {e}")
            # 기본 전략으로 폴백
            return RoutingDecision(
                strategy=RoutingStrategy.SINGLE,
                primary_agents=['business_policy'],
                confidence=0.3,
                reasoning=f"분석 실패로 기본 전략 사용: {str(e)}"
            )

    def _determine_strategy(self, analysis: QuestionAnalysis) -> RoutingStrategy:
        """질문 분석 결과를 바탕으로 라우팅 전략을 결정합니다."""
        # 질문 유형 기반 전략
        base_strategy = self.type_strategy_mapping.get(
            analysis.question_type,
            RoutingStrategy.SINGLE
        )

        # 도메인 개수에 따른 조정
        if len(analysis.relevant_domains) > 2:
            if base_strategy == RoutingStrategy.SINGLE:
                return RoutingStrategy.MULTI_PARALLEL

        # 복잡도에 따른 조정
        if analysis.complexity >= 0.8:
            if base_strategy == RoutingStrategy.SINGLE:
                return RoutingStrategy.MULTI_SEQUENTIAL

        # 최신성 요구사항
        if analysis.requires_latest_info:
            if base_strategy != RoutingStrategy.HYBRID:
                return RoutingStrategy.HYBRID

        return base_strategy

    def _select_primary_agents(self, analysis: QuestionAnalysis) -> List[str]:
        """주요 에이전트를 선택합니다."""
        if not analysis.relevant_domains:
            # 폴백: 비즈니스 정책 에이전트 사용
            return ['business_policy']

        # 신뢰도가 높은 상위 도메인들 선택
        sorted_domains = sorted(
            analysis.relevant_domains.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 최대 3개 에이전트 선택
        primary_agents = []
        for domain, confidence in sorted_domains[:3]:
            if confidence >= 0.3:  # 최소 신뢰도 임계값
                primary_agents.append(domain)

        return primary_agents if primary_agents else ['business_policy']

    def _select_secondary_agents(
        self,
        analysis: QuestionAnalysis,
        primary_agents: List[str]
    ) -> List[str]:
        """보조 에이전트를 선택합니다."""
        secondary_agents = []

        # 관련 도메인 중 주요 에이전트에 포함되지 않은 것들
        for domain, confidence in analysis.relevant_domains.items():
            if domain not in primary_agents and confidence >= 0.2:
                secondary_agents.append(domain)

        # 최대 2개 보조 에이전트
        return secondary_agents[:2]

    def _should_use_web_search(self, analysis: QuestionAnalysis) -> bool:
        """웹 검색이 필요한지 판단합니다."""
        # 최신 정보 요구
        if analysis.requires_latest_info:
            return True

        # 모든 도메인의 신뢰도가 낮음
        if analysis.relevant_domains:
            max_confidence = max(analysis.relevant_domains.values())
            if max_confidence < 0.4:
                return True

        # 특정 키워드가 포함된 경우
        web_keywords = ['최신', '현재', '오늘', '요즘', '트렌드', '뉴스']
        question_lower = analysis.original_question.lower()
        if any(keyword in question_lower for keyword in web_keywords):
            return True

        return False

    def _calculate_confidence(
        self,
        analysis: QuestionAnalysis,
        primary_agents: List[str]
    ) -> float:
        """라우팅 결정의 신뢰도를 계산합니다."""
        if not analysis.relevant_domains:
            return 0.3

        # 주요 에이전트들의 평균 신뢰도
        agent_confidences = []
        for agent in primary_agents:
            if agent in analysis.relevant_domains:
                agent_confidences.append(analysis.relevant_domains[agent])

        if not agent_confidences:
            return 0.3

        base_confidence = sum(agent_confidences) / len(agent_confidences)

        # 질문 명확성에 따른 조정
        clarity_boost = min(analysis.clarity * 0.2, 0.2)

        return min(base_confidence + clarity_boost, 1.0)

    def _generate_reasoning(
        self,
        analysis: QuestionAnalysis,
        strategy: RoutingStrategy,
        primary_agents: List[str]
    ) -> str:
        """라우팅 결정의 추론 과정을 설명합니다."""
        reasoning_parts = []

        # 질문 유형
        reasoning_parts.append(f"질문 유형: {analysis.question_type.value}")

        # 주요 도메인
        if analysis.relevant_domains:
            top_domains = sorted(
                analysis.relevant_domains.items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            domain_str = ", ".join([f"{d}({c:.2f})" for d, c in top_domains])
            reasoning_parts.append(f"관련 도메인: {domain_str}")

        # 전략 선택 이유
        strategy_reason = {
            RoutingStrategy.SINGLE: "단순한 사실 질문",
            RoutingStrategy.MULTI_PARALLEL: "다중 도메인 비교 분석",
            RoutingStrategy.MULTI_SEQUENTIAL: "복잡한 문제 해결",
            RoutingStrategy.HYBRID: "최신 정보 필요"
        }.get(strategy, "기본 전략")

        reasoning_parts.append(f"전략: {strategy.value} ({strategy_reason})")

        # 선택된 에이전트
        reasoning_parts.append(f"에이전트: {', '.join(primary_agents)}")

        return " | ".join(reasoning_parts)

    async def route_question(self, question: str) -> RoutingResult:
        """질문을 라우팅하고 실행합니다."""
        import time
        start_time = time.time()

        try:
            # 라우팅 결정
            routing_decision = self.analyze_and_route(question)

            # 에이전트 응답 수집
            agent_responses = {}
            web_response = None

            # 전략에 따른 실행
            if routing_decision.strategy == RoutingStrategy.SINGLE:
                agent_responses = await self._execute_single(
                    question, routing_decision.primary_agents[0]
                )
            elif routing_decision.strategy == RoutingStrategy.MULTI_PARALLEL:
                agent_responses = await self._execute_parallel(
                    question, routing_decision.primary_agents
                )
            elif routing_decision.strategy == RoutingStrategy.MULTI_SEQUENTIAL:
                agent_responses = await self._execute_sequential(
                    question, routing_decision.primary_agents
                )
            elif routing_decision.strategy == RoutingStrategy.HYBRID:
                agent_responses = await self._execute_parallel(
                    question, routing_decision.primary_agents
                )
                if routing_decision.web_search_required and self.web_agent:
                    web_response = await self._execute_web_search(question)

            execution_time = time.time() - start_time

            return RoutingResult(
                question=question,
                routing_decision=routing_decision,
                agent_responses=agent_responses,
                web_response=web_response,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"질문 라우팅 실패: {e}")

            return RoutingResult(
                question=question,
                routing_decision=RoutingDecision(
                    strategy=RoutingStrategy.SINGLE,
                    primary_agents=['business_policy']
                ),
                agent_responses={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    async def _execute_single(self, question: str, agent_name: str) -> Dict[str, AgentResponse]:
        """단일 에이전트 실행"""
        responses = {}

        # Corrective RAG 에이전트 우선 사용
        if agent_name in self.corrective_agents:
            agent = self.corrective_agents[agent_name]
            response = agent.corrective_query(question)
            responses[agent_name] = AgentResponse(
                content=response.final_answer,
                metadata={
                    'domain': agent_name,
                    'quality_score': response.final_quality.overall_score,
                    'iterations': response.iterations,
                    'sources': response.sources
                },
                status=AgentStatus.SUCCESS if response.final_answer else AgentStatus.FAILED
            )
        elif agent_name in self.domain_agents:
            agent = self.domain_agents[agent_name]
            response = agent.query(question)
            responses[agent_name] = response

        return responses

    async def _execute_parallel(self, question: str, agent_names: List[str]) -> Dict[str, AgentResponse]:
        """병렬 다중 에이전트 실행"""
        tasks = []

        for agent_name in agent_names:
            if agent_name in self.corrective_agents:
                task = asyncio.create_task(
                    self._async_corrective_query(self.corrective_agents[agent_name], question)
                )
            elif agent_name in self.domain_agents:
                task = asyncio.create_task(
                    self._async_agent_query(self.domain_agents[agent_name], question)
                )
            else:
                continue

            tasks.append((agent_name, task))

        responses = {}
        for agent_name, task in tasks:
            try:
                response = await task
                responses[agent_name] = response
            except Exception as e:
                logger.error(f"에이전트 {agent_name} 실행 실패: {e}")
                responses[agent_name] = AgentResponse(
                    content=f"에이전트 실행 실패: {str(e)}",
                    status=AgentStatus.FAILED
                )

        return responses

    async def _execute_sequential(self, question: str, agent_names: List[str]) -> Dict[str, AgentResponse]:
        """순차 다중 에이전트 실행"""
        responses = {}
        context = question

        for agent_name in agent_names:
            try:
                if agent_name in self.corrective_agents:
                    response = await self._async_corrective_query(
                        self.corrective_agents[agent_name], context
                    )
                elif agent_name in self.domain_agents:
                    response = await self._async_agent_query(
                        self.domain_agents[agent_name], context
                    )
                else:
                    continue

                responses[agent_name] = response

                # 다음 에이전트를 위한 컨텍스트 업데이트
                if response.status == AgentStatus.SUCCESS:
                    context = f"{question}\n\n이전 분석: {response.content}"

            except Exception as e:
                logger.error(f"에이전트 {agent_name} 순차 실행 실패: {e}")
                responses[agent_name] = AgentResponse(
                    content=f"에이전트 실행 실패: {str(e)}",
                    status=AgentStatus.FAILED
                )

        return responses

    async def _execute_web_search(self, question: str) -> Optional[AgentResponse]:
        """웹 검색 실행"""
        if not self.web_agent:
            return None

        try:
            return await self._async_agent_query(self.web_agent, question)
        except Exception as e:
            logger.error(f"웹 검색 실패: {e}")
            return AgentResponse(
                content=f"웹 검색 실패: {str(e)}",
                status=AgentStatus.FAILED
            )

    async def _async_agent_query(self, agent: BaseRAGAgent, question: str) -> AgentResponse:
        """비동기 에이전트 쿼리"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.query, question)

    async def _async_corrective_query(self, agent: CorrectiveRAGAgent, question: str) -> AgentResponse:
        """비동기 Corrective RAG 쿼리"""
        loop = asyncio.get_event_loop()
        corrective_response = await loop.run_in_executor(None, agent.corrective_query, question)

        return AgentResponse(
            content=corrective_response.final_answer,
            metadata={
                'domain': agent.domain,
                'quality_score': corrective_response.final_quality.overall_score,
                'iterations': corrective_response.iterations,
                'sources': corrective_response.sources
            },
            status=AgentStatus.SUCCESS if corrective_response.final_answer else AgentStatus.FAILED
        )

    def get_agent_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보를 반환합니다."""
        return {
            'domain_agents': len(self.domain_agents),
            'corrective_agents': len(self.corrective_agents),
            'web_agent_available': self.web_agent is not None,
            'total_agents': len(self.domain_agents) + len(self.corrective_agents) + (1 if self.web_agent else 0)
        }


# 편의 함수들
def create_domain_router(config: Optional[CoolStayConfig] = None) -> DomainRouter:
    """도메인 라우터를 생성합니다."""
    return DomainRouter(config)


async def route_question_simple(
    question: str,
    config: Optional[CoolStayConfig] = None
) -> RoutingResult:
    """간단한 질문 라우팅 함수"""
    router = DomainRouter(config)
    await router.initialize_agents()
    return await router.route_question(question)


def analyze_routing_decision(decision: RoutingDecision) -> Dict[str, Any]:
    """라우팅 결정을 분석합니다."""
    return {
        'strategy': decision.strategy.value,
        'primary_agents': decision.primary_agents,
        'secondary_agents': decision.secondary_agents,
        'web_search_required': decision.web_search_required,
        'confidence': decision.confidence,
        'reasoning': decision.reasoning
    }