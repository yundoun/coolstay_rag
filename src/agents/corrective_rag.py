"""
CoolStay RAG 시스템 Corrective RAG 모듈

이 모듈은 품질 평가 기반 자가교정 RAG 메커니즘을 제공합니다.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..core.llm import CoolStayLLM, get_default_llm
from .base_agent import BaseRAGAgent, AgentResponse, AgentStatus

logger = logging.getLogger(__name__)


class AnswerQuality(Enum):
    """답변 품질 등급"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityAssessment:
    """품질 평가 결과"""
    overall_quality: AnswerQuality
    relevance_score: float      # 관련성 (0-1)
    accuracy_score: float       # 정확성 (0-1)
    completeness_score: float   # 완성도 (0-1)
    confidence_score: float     # 확신도 (0-1)
    reasoning: str              # 평가 이유
    needs_improvement: bool     # 개선 필요 여부
    improvement_suggestions: List[str] = None  # 개선 제안


@dataclass
class CorrectiveResponse:
    """교정 RAG 응답"""
    final_answer: str
    source_documents: List[Document]
    domain: str
    agent_name: str
    quality_assessment: QualityAssessment
    iterations: int
    total_processing_time: float
    iteration_history: List[Dict[str, Any]]
    status: AgentStatus = AgentStatus.READY
    metadata: Optional[Dict[str, Any]] = None


class QualityEvaluator:
    """답변 품질 평가기"""

    def __init__(self, llm: Optional[CoolStayLLM] = None):
        """
        품질 평가기 초기화

        Args:
            llm: 사용할 LLM 인스턴스
        """
        self.llm = llm or get_default_llm()
        self._setup_evaluation_prompt()

    def _setup_evaluation_prompt(self):
        """평가 프롬프트 설정"""
        self.evaluation_prompt = ChatPromptTemplate.from_template("""
당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.

**평가 기준 (각각 0.0-1.0 점수):**
1. **관련성 (Relevance)**: 질문과 답변이 얼마나 관련이 있는가?
2. **정확성 (Accuracy)**: 제공된 정보가 얼마나 정확한가?
3. **완성도 (Completeness)**: 질문에 대한 답변이 얼마나 완전한가?
4. **확신도 (Confidence)**: 답변의 신뢰도는 얼마나 높은가?

**질문:** {question}

**제공된 컨텍스트:**
{context}

**생성된 답변:**
{answer}

**평가 지침:**
- 각 기준에 대해 0.0~1.0 점수를 매기세요 (1.0이 최고)
- 전체적인 품질 등급을 결정하세요 (excellent/good/fair/poor)
- 개선이 필요한지 판단하고 그 이유를 설명하세요
- 개선이 필요하다면 구체적인 제안을 제공하세요

다음 JSON 형식으로 응답하세요:
{{
    "overall_quality": "excellent|good|fair|poor",
    "relevance_score": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "confidence_score": 0.0-1.0,
    "reasoning": "평가 이유를 상세히 설명",
    "needs_improvement": true|false,
    "improvement_suggestions": ["개선 제안 1", "개선 제안 2", ...]
}}
""")

        self.evaluation_chain = (
            self.evaluation_prompt
            | self.llm.llm
            | JsonOutputParser()
        )

    def evaluate(self, question: str, context: List[str], answer: str) -> QualityAssessment:
        """답변 품질 평가"""
        try:
            context_text = "\n\n".join(context) if context else "컨텍스트 없음"

            result = self.evaluation_chain.invoke({
                "question": question,
                "context": context_text,
                "answer": answer
            })

            return QualityAssessment(
                overall_quality=AnswerQuality(result["overall_quality"]),
                relevance_score=result["relevance_score"],
                accuracy_score=result["accuracy_score"],
                completeness_score=result["completeness_score"],
                confidence_score=result["confidence_score"],
                reasoning=result["reasoning"],
                needs_improvement=result["needs_improvement"],
                improvement_suggestions=result.get("improvement_suggestions", [])
            )

        except Exception as e:
            logger.error(f"품질 평가 실패: {e}")
            return QualityAssessment(
                overall_quality=AnswerQuality.FAIR,
                relevance_score=0.5,
                accuracy_score=0.5,
                completeness_score=0.5,
                confidence_score=0.5,
                reasoning="평가 과정에서 오류 발생",
                needs_improvement=True,
                improvement_suggestions=["시스템 오류 해결 필요"]
            )


class QueryRewriter:
    """쿼리 재작성기"""

    def __init__(self, llm: Optional[CoolStayLLM] = None):
        """
        쿼리 재작성기 초기화

        Args:
            llm: 사용할 LLM 인스턴스
        """
        self.llm = llm or get_default_llm()
        self._setup_rewrite_prompt()

    def _setup_rewrite_prompt(self):
        """재작성 프롬프트 설정"""
        self.rewrite_prompt = ChatPromptTemplate.from_template("""
다음 질문에 대한 검색 결과가 만족스럽지 않습니다.
더 나은 검색을 위해 질문을 개선해주세요.

**원래 질문:** {original_question}

**검색된 컨텍스트:**
{context}

**품질 평가 피드백:**
{quality_feedback}

**도메인 정보:** {domain_description}

**개선 요청:**
- 더 구체적이고 명확한 검색 쿼리로 개선
- {domain_description} 도메인에 특화된 용어 사용
- 여러 관점에서 접근할 수 있도록 확장
- 검색 성능을 높일 수 있는 키워드 추가

**개선된 질문 (한국어로만):**
""")

        # LangChain pipeline 표현식을 호환성을 위해 함수 호출로 변경
        from langchain_core.runnables import RunnableLambda

        self.rewrite_chain = (
            self.rewrite_prompt
            | self.llm.llm
            | RunnableLambda(lambda x: x.content.strip())
        )

    def rewrite(self, original_question: str, context: List[str],
                quality_feedback: str, domain_description: str) -> str:
        """쿼리 재작성"""
        try:
            context_text = "\n\n".join(context) if context else "관련 정보 없음"

            improved_query = self.rewrite_chain.invoke({
                "original_question": original_question,
                "context": context_text,
                "quality_feedback": quality_feedback,
                "domain_description": domain_description
            })

            return improved_query.strip()

        except Exception as e:
            logger.error(f"쿼리 재작성 실패: {e}")
            return original_question


class CorrectiveRAGAgent(BaseRAGAgent):
    """교정 RAG 에이전트"""

    def __init__(self, domain: str, llm: Optional[CoolStayLLM] = None,
                 chroma_manager=None, max_iterations: int = 3,
                 quality_threshold: float = 0.7):
        """
        교정 RAG 에이전트 초기화

        Args:
            domain: 담당 도메인
            llm: LLM 인스턴스
            chroma_manager: ChromaDB 관리자
            max_iterations: 최대 교정 반복 횟수
            quality_threshold: 품질 임계값 (이하면 재시도)
        """
        super().__init__(domain, llm, chroma_manager)

        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

        # 교정 컴포넌트 초기화
        self.quality_evaluator = QualityEvaluator(llm)
        self.query_rewriter = QueryRewriter(llm)

        logger.info(f"🔧 {domain} Corrective RAG 에이전트 초기화 완료")

    def corrective_query(self, question: str, **kwargs) -> CorrectiveResponse:
        """교정 RAG 질문 처리"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        iteration_history = []
        current_question = question
        best_response = None
        best_quality = None

        try:
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                logger.info(f"🔄 {self.domain} 교정 반복 {iteration + 1}/{self.max_iterations}")

                # 1. 기본 RAG 처리
                basic_response = super().process_query(current_question)

                # 2. 품질 평가
                context_texts = [doc.page_content for doc in basic_response.source_documents]
                quality = self.quality_evaluator.evaluate(
                    question, context_texts, basic_response.answer
                )

                iteration_time = time.time() - iteration_start

                # 반복 기록
                iteration_record = {
                    'iteration': iteration + 1,
                    'question': current_question,
                    'answer': basic_response.answer,
                    'quality': quality,
                    'documents_count': len(basic_response.source_documents),
                    'processing_time': iteration_time,
                    'is_original_question': (iteration == 0)
                }
                iteration_history.append(iteration_record)

                # 3. 품질 확인
                avg_quality = (
                    quality.relevance_score + quality.accuracy_score +
                    quality.completeness_score + quality.confidence_score
                ) / 4

                # 최고 품질 응답 추적
                if best_quality is None or avg_quality > best_quality:
                    best_response = basic_response
                    best_quality = avg_quality

                # 품질이 충족되면 종료
                if not quality.needs_improvement or avg_quality >= self.quality_threshold:
                    logger.info(f"✅ {self.domain} 품질 목표 달성 (점수: {avg_quality:.2f})")
                    break

                # 마지막 반복이면 종료
                if iteration >= self.max_iterations - 1:
                    logger.info(f"🔄 {self.domain} 최대 반복 횟수 도달")
                    break

                # 4. 쿼리 재작성
                logger.info(f"🔧 {self.domain} 쿼리 재작성 중...")
                current_question = self.query_rewriter.rewrite(
                    question, context_texts, quality.reasoning, self.description
                )

                logger.info(f"   원래 질문: {question}")
                logger.info(f"   개선된 질문: {current_question}")

            total_time = time.time() - start_time

            # 최종 응답 생성
            final_response = CorrectiveResponse(
                final_answer=best_response.answer,
                source_documents=best_response.source_documents,
                domain=self.domain,
                agent_name=self.agent_name,
                quality_assessment=quality,
                iterations=len(iteration_history),
                total_processing_time=total_time,
                iteration_history=iteration_history,
                status=AgentStatus.READY,
                metadata={
                    'original_question': question,
                    'final_question': current_question,
                    'max_iterations': self.max_iterations,
                    'quality_threshold': self.quality_threshold,
                    'achieved_quality': best_quality,
                    'improvement_used': len(iteration_history) > 1
                }
            )

            self.status = AgentStatus.READY
            logger.info(f"✅ {self.domain} 교정 RAG 완료: {len(iteration_history)}회 반복, 품질 {best_quality:.2f}")

            return final_response

        except Exception as e:
            self.status = AgentStatus.ERROR
            error_msg = str(e)
            logger.error(f"❌ {self.domain} 교정 RAG 실패: {error_msg}")

            # 기본 응답 반환 (오류 발생 시)
            fallback_response = super().process_query(question)

            return CorrectiveResponse(
                final_answer=f"교정 처리 중 오류 발생. 기본 답변: {fallback_response.answer}",
                source_documents=fallback_response.source_documents,
                domain=self.domain,
                agent_name=self.agent_name,
                quality_assessment=QualityAssessment(
                    overall_quality=AnswerQuality.FAIR,
                    relevance_score=0.5,
                    accuracy_score=0.5,
                    completeness_score=0.5,
                    confidence_score=0.5,
                    reasoning=f"교정 처리 오류: {error_msg}",
                    needs_improvement=True
                ),
                iterations=len(iteration_history),
                total_processing_time=time.time() - start_time,
                iteration_history=iteration_history,
                status=AgentStatus.ERROR,
                metadata={'error': error_msg}
            )

    def process_query(self, question: str, enable_corrective: bool = True, **kwargs) -> CorrectiveResponse:
        """질문 처리 (교정 기능 포함)"""
        if enable_corrective:
            return self.corrective_query(question, **kwargs)
        else:
            # 기본 RAG만 사용
            basic_response = super().process_query(question, **kwargs)
            return CorrectiveResponse(
                final_answer=basic_response.answer,
                source_documents=basic_response.source_documents,
                domain=self.domain,
                agent_name=self.agent_name,
                quality_assessment=QualityAssessment(
                    overall_quality=AnswerQuality.GOOD,
                    relevance_score=0.8,
                    accuracy_score=0.8,
                    completeness_score=0.8,
                    confidence_score=0.8,
                    reasoning="기본 RAG 사용 (교정 비활성화)",
                    needs_improvement=False
                ),
                iterations=1,
                total_processing_time=basic_response.processing_time or 0.0,
                iteration_history=[{
                    'iteration': 1,
                    'question': question,
                    'answer': basic_response.answer,
                    'quality': None,
                    'documents_count': len(basic_response.source_documents),
                    'processing_time': basic_response.processing_time,
                    'is_original_question': True
                }],
                status=basic_response.status,
                metadata={'corrective_disabled': True}
            )

    def get_corrective_stats(self) -> Dict[str, Any]:
        """교정 통계 반환"""
        base_status = self.get_status()

        corrective_stats = {
            **base_status,
            'corrective_features': {
                'max_iterations': self.max_iterations,
                'quality_threshold': self.quality_threshold,
                'quality_evaluator_ready': self.quality_evaluator is not None,
                'query_rewriter_ready': self.query_rewriter is not None
            }
        }

        return corrective_stats


# 편의 함수들
def create_corrective_agent(domain: str, llm: Optional[CoolStayLLM] = None,
                           chroma_manager=None, max_iterations: int = 3) -> CorrectiveRAGAgent:
    """교정 RAG 에이전트 생성 편의 함수"""
    return CorrectiveRAGAgent(domain, llm, chroma_manager, max_iterations)


def create_all_corrective_agents(llm: Optional[CoolStayLLM] = None,
                                chroma_manager=None) -> Dict[str, CorrectiveRAGAgent]:
    """모든 도메인 교정 에이전트 생성"""
    from ..core.config import config

    agents = {}

    for domain in config.domain_list:
        try:
            agent = CorrectiveRAGAgent(domain, llm, chroma_manager)
            if agent.status != AgentStatus.ERROR:
                agents[domain] = agent
                logger.info(f"✅ {domain} 교정 에이전트 생성 완료")
            else:
                logger.warning(f"⚠️ {domain} 교정 에이전트 생성 실패")
        except Exception as e:
            logger.error(f"❌ {domain} 교정 에이전트 생성 중 오류: {e}")

    logger.info(f"🎉 교정 에이전트 생성 완료: {len(agents)}/{len(config.domain_list)}개")
    return agents


if __name__ == "__main__":
    # 교정 RAG 에이전트 테스트
    print("🔧 CoolStay 교정 RAG 에이전트 테스트")
    print("=" * 50)

    # 단일 교정 에이전트 테스트
    test_domain = "hr_policy"
    print(f"🔍 {test_domain} 교정 에이전트 생성 중...")

    agent = create_corrective_agent(test_domain, max_iterations=2)
    stats = agent.get_corrective_stats()

    print(f"📊 교정 에이전트 상태:")
    print(f"  도메인: {stats['domain']}")
    print(f"  상태: {stats['status']}")
    print(f"  최대 반복: {stats['corrective_features']['max_iterations']}")
    print(f"  품질 임계값: {stats['corrective_features']['quality_threshold']}")

    # 헬스 체크
    health = agent.health_check()
    print(f"\n🏥 헬스 체크: {health['overall_status']}")

    # 교정 질문 처리 테스트
    if health['overall_status'] in ['healthy', 'degraded']:
        print(f"\n💬 교정 RAG 테스트:")
        test_question = "연차는 언제 사용할 수 있나요?"
        print(f"   질문: {test_question}")

        # 교정 기능 활성화
        response = agent.process_query(test_question, enable_corrective=True)

        print(f"   최종 답변: {response.final_answer[:200]}...")
        print(f"   총 처리 시간: {response.total_processing_time:.2f}초")
        print(f"   반복 횟수: {response.iterations}")
        print(f"   품질 점수: {(response.quality_assessment.relevance_score + response.quality_assessment.accuracy_score + response.quality_assessment.completeness_score + response.quality_assessment.confidence_score) / 4:.2f}")
        print(f"   개선 사용: {'예' if response.metadata.get('improvement_used') else '아니오'}")

        # 반복 히스토리
        print(f"\n📈 반복 히스토리:")
        for i, history in enumerate(response.iteration_history):
            print(f"   반복 {history['iteration']}: {history['processing_time']:.2f}초, 문서 {history['documents_count']}개")
    else:
        print(f"\n❌ 에이전트가 정상 상태가 아니어서 테스트를 건너뜁니다.")

    print(f"\n🎯 모든 교정 에이전트 생성 테스트:")
    all_agents = create_all_corrective_agents()
    print(f"   생성된 교정 에이전트: {len(all_agents)}개")