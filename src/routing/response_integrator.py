"""
CoolStay RAG 응답 통합 모듈

여러 에이전트의 응답을 통합하여 최종 답변을 생성하는 모듈입니다.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json
from datetime import datetime

from ..core.config import CoolStayConfig
from ..core.llm import CoolStayLLM, get_default_llm
from ..agents import AgentResponse, AgentStatus
from .domain_router import RoutingResult, RoutingStrategy

logger = logging.getLogger(__name__)


class IntegrationStrategy(Enum):
    """통합 전략"""
    SIMPLE_MERGE = "simple_merge"      # 단순 병합
    WEIGHTED_MERGE = "weighted_merge"  # 가중치 기반 병합
    AI_SYNTHESIS = "ai_synthesis"      # AI 기반 종합
    CONFIDENCE_RANKING = "confidence_ranking"  # 신뢰도 기반 순위
    CONSENSUS_BUILDING = "consensus_building"  # 합의 구축


@dataclass
class IntegratedResponse:
    """통합 응답"""
    final_answer: str
    confidence: float
    sources: List[str]
    contributing_agents: List[str]
    integration_strategy: IntegrationStrategy
    reasoning: str
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    execution_time: float
    success: bool = True
    error_message: str = ""


@dataclass
class ResponseWeights:
    """응답 가중치"""
    agent_name: str
    quality_weight: float = 0.0
    confidence_weight: float = 0.0
    relevance_weight: float = 0.0
    completeness_weight: float = 0.0
    total_weight: float = 0.0


class ResponseIntegrator:
    """
    응답 통합기

    여러 에이전트의 응답을 분석하고 통합하여
    최적의 최종 답변을 생성합니다.
    """

    def __init__(self, config: Optional[CoolStayConfig] = None):
        """
        Args:
            config: CoolStay 설정 객체
        """
        self.config = config or CoolStayConfig()
        self.llm = get_default_llm()

        # 통합 프롬프트 설정
        self._setup_integration_prompts()

        # 품질 평가 기준
        self._setup_quality_metrics()

        logger.info("ResponseIntegrator 초기화 완료")

    def _setup_integration_prompts(self):
        """통합용 프롬프트들을 설정합니다."""

        self.synthesis_prompt = """
당신은 여러 전문가의 의견을 종합하여 최고 품질의 답변을 만드는 전문가입니다.

다음 질문에 대해 여러 전문가들이 제공한 답변들을 분석하고 종합해주세요:

**질문**: {question}

**전문가 답변들**:
{agent_responses}

**종합 지침**:
1. 각 답변의 핵심 내용을 파악하세요
2. 일관된 정보는 통합하고, 상충하는 정보는 신뢰도를 비교하세요
3. 누락된 중요한 관점이 있다면 지적하세요
4. 명확하고 완전한 최종 답변을 제공하세요

**출력 형식**:
```json
{{
    "final_answer": "종합된 최종 답변",
    "confidence": 0.85,
    "key_insights": ["핵심 인사이트 1", "핵심 인사이트 2"],
    "consensus_points": ["합의점 1", "합의점 2"],
    "conflicting_points": ["상충점 1과 해결방안"],
    "quality_assessment": {{
        "completeness": 0.9,
        "accuracy": 0.8,
        "relevance": 0.9,
        "clarity": 0.85
    }}
}}
```
"""

        self.conflict_resolution_prompt = """
다음 상충하는 정보들을 분석하여 최적의 해결방안을 제시해주세요:

**질문**: {question}

**상충하는 답변들**:
{conflicting_responses}

각 답변의 신뢰도, 근거의 강도, 정보의 최신성을 고려하여
가장 적절한 답변을 선택하거나 통합된 답변을 제시해주세요.

출력: JSON 형식으로 final_answer, reasoning, confidence를 포함해주세요.
"""

        self.quality_evaluation_prompt = """
다음 답변의 품질을 평가해주세요:

**질문**: {question}
**답변**: {answer}

다음 기준으로 0-1 점수를 매겨주세요:
- completeness: 답변의 완전성
- accuracy: 정확성
- relevance: 관련성
- clarity: 명확성
- usefulness: 실용성

JSON 형식으로 각 점수와 총평을 제공해주세요.
"""

    def _setup_quality_metrics(self):
        """품질 평가 기준을 설정합니다."""
        self.quality_weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'relevance': 0.25,
            'clarity': 0.20
        }

        self.minimum_quality_threshold = 0.6
        self.confidence_threshold = 0.7

    def integrate_responses(self, routing_result: RoutingResult) -> IntegratedResponse:
        """라우팅 결과의 응답들을 통합합니다."""
        import time
        start_time = time.time()

        try:
            # 통합 전략 결정
            strategy = self._determine_integration_strategy(routing_result)

            # 응답 전처리
            valid_responses = self._preprocess_responses(routing_result.agent_responses)

            if not valid_responses:
                return self._create_error_response(
                    "유효한 응답이 없습니다",
                    time.time() - start_time
                )

            # 전략에 따른 통합 실행
            if strategy == IntegrationStrategy.SIMPLE_MERGE:
                result = self._simple_merge(routing_result, valid_responses)
            elif strategy == IntegrationStrategy.WEIGHTED_MERGE:
                result = self._weighted_merge(routing_result, valid_responses)
            elif strategy == IntegrationStrategy.AI_SYNTHESIS:
                result = self._ai_synthesis(routing_result, valid_responses)
            elif strategy == IntegrationStrategy.CONFIDENCE_RANKING:
                result = self._confidence_ranking(routing_result, valid_responses)
            elif strategy == IntegrationStrategy.CONSENSUS_BUILDING:
                result = self._consensus_building(routing_result, valid_responses)
            else:
                result = self._simple_merge(routing_result, valid_responses)

            # 웹 검색 결과 통합
            if routing_result.web_response and routing_result.web_response.status == AgentStatus.READY:
                result = self._integrate_web_response(result, routing_result.web_response)

            # 실행 시간 설정
            result.execution_time = time.time() - start_time

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"응답 통합 실패: {e}")
            return self._create_error_response(str(e), execution_time)

    def _determine_integration_strategy(self, routing_result: RoutingResult) -> IntegrationStrategy:
        """통합 전략을 결정합니다."""
        num_responses = len([r for r in routing_result.agent_responses.values()
                           if r.status == AgentStatus.READY])

        # 단일 응답인 경우
        if num_responses == 1:
            return IntegrationStrategy.SIMPLE_MERGE

        # 라우팅 전략에 따른 매핑
        strategy_mapping = {
            RoutingStrategy.SINGLE: IntegrationStrategy.SIMPLE_MERGE,
            RoutingStrategy.MULTI_PARALLEL: IntegrationStrategy.AI_SYNTHESIS,
            RoutingStrategy.MULTI_SEQUENTIAL: IntegrationStrategy.WEIGHTED_MERGE,
            RoutingStrategy.HYBRID: IntegrationStrategy.CONSENSUS_BUILDING
        }

        return strategy_mapping.get(
            routing_result.routing_decision.strategy,
            IntegrationStrategy.AI_SYNTHESIS
        )

    def _preprocess_responses(
        self,
        agent_responses: Dict[str, AgentResponse]
    ) -> Dict[str, AgentResponse]:
        """응답을 전처리합니다."""
        valid_responses = {}

        for agent_name, response in agent_responses.items():
            # 성공한 응답만 포함
            if response.status == AgentStatus.READY and response.answer.strip():
                # 최소 길이 체크
                if len(response.answer.strip()) >= 10:
                    valid_responses[agent_name] = response

        return valid_responses

    def _simple_merge(
        self,
        routing_result: RoutingResult,
        valid_responses: Dict[str, AgentResponse]
    ) -> IntegratedResponse:
        """단순 병합 전략"""
        if len(valid_responses) == 1:
            # 단일 응답인 경우
            agent_name, response = next(iter(valid_responses.items()))
            return IntegratedResponse(
                final_answer=response.answer,
                confidence=0.8,
                sources=response.metadata.get('sources', []),
                contributing_agents=[agent_name],
                integration_strategy=IntegrationStrategy.SIMPLE_MERGE,
                reasoning=f"단일 에이전트 {agent_name} 응답 사용",
                metadata=response.metadata or {},
                quality_metrics=self._calculate_basic_quality_metrics(response.answer),
                execution_time=0.0
            )

        # 여러 응답을 단순 연결
        merged_content = []
        all_sources = []
        contributing_agents = []

        for agent_name, response in valid_responses.items():
            merged_content.append(f"**{agent_name} 관점**:\n{response.answer}")
            if response.metadata and 'sources' in response.metadata:
                all_sources.extend(response.metadata['sources'])
            contributing_agents.append(agent_name)

        final_answer = "\n\n".join(merged_content)

        return IntegratedResponse(
            final_answer=final_answer,
            confidence=0.7,
            sources=list(set(all_sources)),
            contributing_agents=contributing_agents,
            integration_strategy=IntegrationStrategy.SIMPLE_MERGE,
            reasoning="다중 에이전트 응답 단순 병합",
            metadata={'merge_count': len(valid_responses)},
            quality_metrics=self._calculate_basic_quality_metrics(final_answer),
            execution_time=0.0
        )

    def _weighted_merge(
        self,
        routing_result: RoutingResult,
        valid_responses: Dict[str, AgentResponse]
    ) -> IntegratedResponse:
        """가중치 기반 병합"""
        # 응답별 가중치 계산
        weights = self._calculate_response_weights(valid_responses, routing_result)

        # 가중치가 가장 높은 응답을 기본으로 사용
        primary_response = max(weights, key=lambda w: w.total_weight)
        primary_content = valid_responses[primary_response.agent_name].answer

        # 보조 정보 추가
        supplementary_info = []
        for weight in weights:
            if weight.agent_name != primary_response.agent_name and weight.total_weight > 0.3:
                agent_response = valid_responses[weight.agent_name]
                supplementary_info.append(
                    f"\n**추가 정보 ({weight.agent_name})**:\n{agent_response.answer[:200]}..."
                )

        final_answer = primary_content
        if supplementary_info:
            final_answer += "\n\n" + "\n".join(supplementary_info)

        # 모든 소스 수집
        all_sources = []
        contributing_agents = []
        for weight in weights:
            contributing_agents.append(weight.agent_name)
            response = valid_responses[weight.agent_name]
            if response.metadata and 'sources' in response.metadata:
                all_sources.extend(response.metadata['sources'])

        return IntegratedResponse(
            final_answer=final_answer,
            confidence=min(primary_response.total_weight + 0.1, 0.95),
            sources=list(set(all_sources)),
            contributing_agents=contributing_agents,
            integration_strategy=IntegrationStrategy.WEIGHTED_MERGE,
            reasoning=f"주요 응답: {primary_response.agent_name} (가중치: {primary_response.total_weight:.2f})",
            metadata={
                'primary_agent': primary_response.agent_name,
                'weights': {w.agent_name: w.total_weight for w in weights}
            },
            quality_metrics=self._calculate_basic_quality_metrics(final_answer),
            execution_time=0.0
        )

    def _ai_synthesis(
        self,
        routing_result: RoutingResult,
        valid_responses: Dict[str, AgentResponse]
    ) -> IntegratedResponse:
        """AI 기반 종합"""
        try:
            # 응답들을 형식화
            agent_responses_text = self._format_responses_for_synthesis(valid_responses)

            # LLM을 통한 종합
            synthesis_result = self.llm.invoke(
                self.synthesis_prompt.format(
                    question=routing_result.question,
                    agent_responses=agent_responses_text
                )
            )

            # JSON 파싱 시도
            try:
                # JSON 코드 블록 제거
                content = synthesis_result.content.strip()
                if content.startswith('```json'):
                    content = content[7:]  # '```json' 제거
                if content.endswith('```'):
                    content = content[:-3]  # '```' 제거
                content = content.strip()

                parsed_result = json.loads(content)

                final_answer = parsed_result.get('final_answer', synthesis_result.content)
                confidence = parsed_result.get('confidence', 0.8)
                quality_metrics = parsed_result.get('quality_assessment', {})

            except json.JSONDecodeError:
                # JSON 파싱 실패 시 원본 사용
                final_answer = synthesis_result.content
                confidence = 0.75
                quality_metrics = self._calculate_basic_quality_metrics(final_answer)

            # 모든 소스 수집
            all_sources = []
            contributing_agents = list(valid_responses.keys())
            for response in valid_responses.values():
                if response.metadata and 'sources' in response.metadata:
                    all_sources.extend(response.metadata['sources'])

            return IntegratedResponse(
                final_answer=final_answer,
                confidence=confidence,
                sources=list(set(all_sources)),
                contributing_agents=contributing_agents,
                integration_strategy=IntegrationStrategy.AI_SYNTHESIS,
                reasoning="AI 기반 다중 응답 종합 분석",
                metadata={'synthesis_model': self.llm.model_name},
                quality_metrics=quality_metrics or self._calculate_basic_quality_metrics(final_answer),
                execution_time=0.0
            )

        except Exception as e:
            logger.error(f"AI 종합 실패: {e}")
            # 폴백: 가중치 기반 병합
            return self._weighted_merge(routing_result, valid_responses)

    def _confidence_ranking(
        self,
        routing_result: RoutingResult,
        valid_responses: Dict[str, AgentResponse]
    ) -> IntegratedResponse:
        """신뢰도 기반 순위"""
        # 응답별 신뢰도 점수 계산
        scored_responses = []

        for agent_name, response in valid_responses.items():
            # 메타데이터에서 품질 점수 추출
            quality_score = 0.7  # 기본값
            if response.metadata:
                if 'quality_score' in response.metadata:
                    quality_score = response.metadata['quality_score']
                elif 'confidence' in response.metadata:
                    quality_score = response.metadata['confidence']

            # 응답 길이 고려 (너무 짧거나 긴 것은 감점)
            length_score = min(len(response.answer) / 500, 1.0)
            if len(response.answer) < 50:
                length_score *= 0.5

            total_score = (quality_score * 0.7) + (length_score * 0.3)
            scored_responses.append((agent_name, response, total_score))

        # 점수 순으로 정렬
        scored_responses.sort(key=lambda x: x[2], reverse=True)

        # 최고 점수 응답을 기본으로 사용
        best_agent, best_response, best_score = scored_responses[0]
        final_answer = best_response.answer

        # 상위 2개 응답이 비슷한 점수라면 통합
        if len(scored_responses) > 1:
            second_score = scored_responses[1][2]
            if abs(best_score - second_score) < 0.1:
                second_agent, second_response, _ = scored_responses[1]
                final_answer += f"\n\n**추가 관점 ({second_agent})**:\n{second_response.answer[:300]}..."

        # 소스 수집
        all_sources = []
        contributing_agents = [item[0] for item in scored_responses]
        for _, response, _ in scored_responses:
            if response.metadata and 'sources' in response.metadata:
                all_sources.extend(response.metadata['sources'])

        return IntegratedResponse(
            final_answer=final_answer,
            confidence=min(best_score, 0.95),
            sources=list(set(all_sources)),
            contributing_agents=contributing_agents,
            integration_strategy=IntegrationStrategy.CONFIDENCE_RANKING,
            reasoning=f"최고 신뢰도 응답: {best_agent} (점수: {best_score:.2f})",
            metadata={
                'rankings': {agent: score for agent, _, score in scored_responses}
            },
            quality_metrics=self._calculate_basic_quality_metrics(final_answer),
            execution_time=0.0
        )

    def _consensus_building(
        self,
        routing_result: RoutingResult,
        valid_responses: Dict[str, AgentResponse]
    ) -> IntegratedResponse:
        """합의 구축 전략"""
        try:
            # 공통점과 차이점 분석
            common_points = self._find_common_points(valid_responses)
            differences = self._find_differences(valid_responses)

            # 합의된 내용을 기반으로 답변 구성
            final_parts = []

            if common_points:
                final_parts.append("**합의된 핵심 내용**:")
                for point in common_points:
                    final_parts.append(f"• {point}")

            # 차이점이 있다면 균형있게 제시
            if differences:
                final_parts.append("\n**추가 고려사항**:")
                for diff in differences[:3]:  # 최대 3개만
                    final_parts.append(f"• {diff}")

            # 종합 결론
            if len(valid_responses) > 1:
                conclusion = self._generate_consensus_conclusion(routing_result.question, valid_responses)
                if conclusion:
                    final_parts.append(f"\n**종합 결론**:\n{conclusion}")

            final_answer = "\n".join(final_parts)

            # 소스 수집
            all_sources = []
            contributing_agents = list(valid_responses.keys())
            for response in valid_responses.values():
                if response.metadata and 'sources' in response.metadata:
                    all_sources.extend(response.metadata['sources'])

            return IntegratedResponse(
                final_answer=final_answer,
                confidence=0.85,
                sources=list(set(all_sources)),
                contributing_agents=contributing_agents,
                integration_strategy=IntegrationStrategy.CONSENSUS_BUILDING,
                reasoning=f"{len(common_points)}개 합의점, {len(differences)}개 차이점 통합",
                metadata={
                    'common_points': len(common_points),
                    'differences': len(differences)
                },
                quality_metrics=self._calculate_basic_quality_metrics(final_answer),
                execution_time=0.0
            )

        except Exception as e:
            logger.error(f"합의 구축 실패: {e}")
            # 폴백: AI 종합
            return self._ai_synthesis(routing_result, valid_responses)

    def _calculate_response_weights(
        self,
        valid_responses: Dict[str, AgentResponse],
        routing_result: RoutingResult
    ) -> List[ResponseWeights]:
        """응답별 가중치를 계산합니다."""
        weights = []

        for agent_name, response in valid_responses.items():
            weight = ResponseWeights(agent_name=agent_name)

            # 품질 가중치 (메타데이터에서 추출)
            if response.metadata and 'quality_score' in response.metadata:
                weight.quality_weight = response.metadata['quality_score']
            else:
                weight.quality_weight = 0.7  # 기본값

            # 신뢰도 가중치
            if agent_name in routing_result.routing_decision.primary_agents:
                weight.confidence_weight = 0.8
            elif routing_result.routing_decision.secondary_agents and agent_name in routing_result.routing_decision.secondary_agents:
                weight.confidence_weight = 0.6
            else:
                weight.confidence_weight = 0.4

            # 관련성 가중치 (도메인 매칭 확인)
            question_lower = routing_result.question.lower()
            domain_keywords = self._get_domain_keywords(agent_name)
            keyword_matches = sum(1 for keyword in domain_keywords if keyword in question_lower)
            weight.relevance_weight = min(keyword_matches / max(len(domain_keywords), 1) + 0.3, 1.0)

            # 완전성 가중치 (응답 길이 고려)
            content_length = len(response.answer)
            if content_length > 200:
                weight.completeness_weight = min(content_length / 1000, 1.0)
            else:
                weight.completeness_weight = content_length / 200

            # 총 가중치 계산
            weight.total_weight = (
                weight.quality_weight * 0.3 +
                weight.confidence_weight * 0.3 +
                weight.relevance_weight * 0.2 +
                weight.completeness_weight * 0.2
            )

            weights.append(weight)

        return weights

    def _get_domain_keywords(self, agent_name: str) -> List[str]:
        """도메인별 키워드를 반환합니다."""
        domain_keywords = {
            'hr_policy': ['인사', '채용', '직원', '급여', '연차'],
            'tech_policy': ['기술', '개발', '시스템', '보안'],
            'architecture': ['아키텍처', '설계', '구조', '패턴'],
            'component': ['컴포넌트', '모듈', 'API', '서비스'],
            'deployment': ['배포', '운영', '환경', 'DevOps'],
            'development': ['개발', '코딩', '테스트', 'Git'],
            'business_policy': ['비즈니스', '정책', '절차', '고객']
        }
        return domain_keywords.get(agent_name, [])

    def _format_responses_for_synthesis(self, valid_responses: Dict[str, AgentResponse]) -> str:
        """응답들을 종합용으로 형식화합니다."""
        formatted_responses = []

        for agent_name, response in valid_responses.items():
            formatted_response = f"""
**{agent_name.upper()} 전문가**:
{response.answer}

품질 정보: {response.metadata.get('quality_score', 'N/A') if response.metadata else 'N/A'}
소스: {', '.join(response.metadata.get('sources', [])) if response.metadata and response.metadata.get('sources') else '내부 문서'}
"""
            formatted_responses.append(formatted_response.strip())

        return "\n\n" + "="*50 + "\n\n".join(formatted_responses)

    def _find_common_points(self, valid_responses: Dict[str, AgentResponse]) -> List[str]:
        """응답들 간의 공통점을 찾습니다."""
        # 간단한 키워드 기반 공통점 찾기
        common_points = []

        if len(valid_responses) < 2:
            return common_points

        # 모든 응답에서 공통으로 언급되는 주요 개념들
        all_contents = [response.answer.lower() for response in valid_responses.values()]

        # 공통 키워드 추출 (간단한 구현)
        common_keywords = [
            '정책', '규정', '절차', '승인', '문서', '시스템',
            '개발', '기술', '보안', '관리', '서비스', '사용자'
        ]

        for keyword in common_keywords:
            if all(keyword in content for content in all_contents):
                common_points.append(f"{keyword} 관련 내용이 모든 응답에서 언급됨")

        return common_points[:5]  # 최대 5개

    def _find_differences(self, valid_responses: Dict[str, AgentResponse]) -> List[str]:
        """응답들 간의 차이점을 찾습니다."""
        differences = []

        if len(valid_responses) < 2:
            return differences

        # 각 에이전트별 고유한 관점 추출
        for agent_name, response in valid_responses.items():
            # 해당 도메인 특화 키워드 체크
            domain_keywords = self._get_domain_keywords(agent_name)
            unique_mentions = []

            for keyword in domain_keywords:
                if keyword in response.answer.lower():
                    # 다른 응답에서 언급되지 않은 경우
                    other_responses = [r.answer.lower() for name, r in valid_responses.items() if name != agent_name]
                    if not any(keyword in other_content for other_content in other_responses):
                        unique_mentions.append(keyword)

            if unique_mentions:
                differences.append(f"{agent_name}에서만 언급: {', '.join(unique_mentions)}")

        return differences[:3]  # 최대 3개

    def _generate_consensus_conclusion(
        self,
        question: str,
        valid_responses: Dict[str, AgentResponse]
    ) -> str:
        """합의 기반 결론을 생성합니다."""
        try:
            # 간단한 결론 생성
            agent_count = len(valid_responses)

            if agent_count == 1:
                return ""

            conclusion_parts = [
                f"{agent_count}개 전문 영역의 관점을 종합한 결과,",
            ]

            # 주요 에이전트별 핵심 포인트
            key_points = []
            for agent_name, response in valid_responses.items():
                # 첫 문장 추출 (간단한 구현)
                first_sentence = response.answer.split('.')[0][:100]
                if first_sentence:
                    key_points.append(f"{agent_name}: {first_sentence}")

            if key_points:
                conclusion_parts.append("다음과 같은 핵심 내용들이 확인되었습니다:")
                conclusion_parts.extend(f"• {point}" for point in key_points[:3])

            return "\n".join(conclusion_parts)

        except Exception as e:
            logger.error(f"결론 생성 실패: {e}")
            return ""

    def _integrate_web_response(
        self,
        base_response: IntegratedResponse,
        web_response: AgentResponse
    ) -> IntegratedResponse:
        """웹 검색 결과를 통합합니다."""
        # 웹 검색 정보를 추가 섹션으로 포함
        web_section = f"\n\n**최신 웹 정보**:\n{web_response.answer}"

        updated_response = IntegratedResponse(
            final_answer=base_response.final_answer + web_section,
            confidence=min(base_response.confidence + 0.1, 0.95),
            sources=base_response.sources + web_response.metadata.get('sources', []),
            contributing_agents=base_response.contributing_agents + ['web_search'],
            integration_strategy=base_response.integration_strategy,
            reasoning=base_response.reasoning + " + 웹 검색 정보 통합",
            metadata={
                **base_response.metadata,
                'web_search_included': True
            },
            quality_metrics=base_response.quality_metrics,
            execution_time=base_response.execution_time
        )

        return updated_response

    def _calculate_basic_quality_metrics(self, content: str) -> Dict[str, float]:
        """기본 품질 지표를 계산합니다."""
        # 간단한 휴리스틱 기반 품질 평가
        length = len(content)

        # 완전성 (길이 기반)
        completeness = min(length / 500, 1.0) if length > 50 else length / 50

        # 명확성 (구조화 정도)
        clarity = 0.7  # 기본값
        if '**' in content or '•' in content or '\n' in content:
            clarity += 0.2

        # 관련성 (기본값)
        relevance = 0.8

        # 정확성 (기본값)
        accuracy = 0.75

        return {
            'completeness': completeness,
            'clarity': min(clarity, 1.0),
            'relevance': relevance,
            'accuracy': accuracy
        }

    def _create_error_response(self, error_message: str, execution_time: float) -> IntegratedResponse:
        """오류 응답을 생성합니다."""
        return IntegratedResponse(
            final_answer=f"응답 통합 중 오류가 발생했습니다: {error_message}",
            confidence=0.0,
            sources=[],
            contributing_agents=[],
            integration_strategy=IntegrationStrategy.SIMPLE_MERGE,
            reasoning="오류 발생으로 인한 기본 응답",
            metadata={'error': True},
            quality_metrics={'completeness': 0.0, 'clarity': 0.0, 'relevance': 0.0, 'accuracy': 0.0},
            execution_time=execution_time,
            success=False,
            error_message=error_message
        )


# 편의 함수들
def create_response_integrator(config: Optional[CoolStayConfig] = None) -> ResponseIntegrator:
    """응답 통합기를 생성합니다."""
    return ResponseIntegrator(config)


def integrate_simple(
    question: str,
    agent_responses: Dict[str, AgentResponse],
    config: Optional[CoolStayConfig] = None
) -> IntegratedResponse:
    """간단한 응답 통합 함수"""
    integrator = ResponseIntegrator(config)

    # 더미 라우팅 결과 생성
    from .domain_router import RoutingResult, RoutingDecision, RoutingStrategy

    routing_result = RoutingResult(
        question=question,
        routing_decision=RoutingDecision(
            strategy=RoutingStrategy.SINGLE,
            primary_agents=list(agent_responses.keys())
        ),
        agent_responses=agent_responses,
        execution_time=0.0
    )

    return integrator.integrate_responses(routing_result)


def evaluate_integration_quality(integrated_response: IntegratedResponse) -> Dict[str, Any]:
    """통합 응답의 품질을 평가합니다."""
    return {
        'overall_score': sum(integrated_response.quality_metrics.values()) / len(integrated_response.quality_metrics),
        'confidence': integrated_response.confidence,
        'contributing_agents': len(integrated_response.contributing_agents),
        'has_sources': len(integrated_response.sources) > 0,
        'integration_strategy': integrated_response.integration_strategy.value,
        'success': integrated_response.success
    }