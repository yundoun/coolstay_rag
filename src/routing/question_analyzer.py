"""
CoolStay RAG 시스템 질문 분석 모듈

이 모듈은 사용자 질문을 분석하여 적절한 도메인과 에이전트를 선택하는 기능을 제공합니다.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from ..core.config import CoolStayConfig

from ..core.config import config, get_domain_config
from ..core.llm import CoolStayLLM, get_default_llm

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """질문 유형"""
    SPECIFIC_DOMAIN = "specific_domain"      # 특정 도메인 질문
    MULTI_DOMAIN = "multi_domain"           # 다중 도메인 질문
    GENERAL = "general"                     # 일반적인 질문
    WEB_SEARCH = "web_search"              # 웹 검색 필요
    UNCLEAR = "unclear"                     # 불분명한 질문


class UrgencyLevel(Enum):
    """긴급도 레벨"""
    HIGH = "high"       # 높음 (즉시 처리)
    MEDIUM = "medium"   # 보통 (일반 처리)
    LOW = "low"         # 낮음 (지연 처리 가능)


@dataclass
class QuestionAnalysis:
    """질문 분석 결과"""
    question: str
    question_type: QuestionType
    primary_domains: List[str]              # 주요 관련 도메인들
    secondary_domains: List[str]            # 부차적 관련 도메인들
    confidence_score: float                 # 분석 확신도 (0-1)
    urgency_level: UrgencyLevel
    keywords: List[str]                     # 추출된 키워드
    intent: str                            # 질문 의도
    complexity: str                        # 복잡도 (simple/medium/complex)
    requires_web_search: bool              # 웹 검색 필요 여부
    reasoning: str                         # 분석 근거
    metadata: Optional[Dict[str, Any]] = None


class QuestionAnalyzer:
    """질문 분석기"""

    def __init__(self, llm: Optional[CoolStayLLM] = None):
        """
        질문 분석기 초기화

        Args:
            llm: 사용할 LLM 인스턴스
        """
        self.llm = llm or get_default_llm()
        self.domain_info = self._prepare_domain_info()
        self._setup_analysis_prompt()

    def _prepare_domain_info(self) -> Dict[str, Dict[str, Any]]:
        """도메인 정보 준비"""
        domain_info = {}

        for domain in config.domain_list:
            try:
                domain_config = get_domain_config(domain)
                domain_info[domain] = {
                    'description': domain_config.description,
                    'keywords': domain_config.keywords
                }
            except ValueError:
                logger.warning(f"도메인 설정 로드 실패: {domain}")

        # 웹 검색 도메인 추가
        domain_info['web_search'] = config.web_search_config

        return domain_info

    def _setup_analysis_prompt(self):
        """질문 분석 프롬프트 설정"""
        # 도메인 정보를 텍스트로 변환
        domain_descriptions = []
        for domain, info in self.domain_info.items():
            keywords = ', '.join(info['keywords']) if info['keywords'] else '없음'
            domain_descriptions.append(f"- **{domain}**: {info['description']} (키워드: {keywords})")

        domain_info_text = '\n'.join(domain_descriptions)

        self.analysis_prompt = ChatPromptTemplate.from_template(f"""
당신은 꿀스테이 RAG 시스템의 질문 분석 전문가입니다.
사용자 질문을 분석하여 적절한 도메인과 처리 방법을 결정해야 합니다.

**사용 가능한 도메인:**
{domain_info_text}

**분석할 질문:** {{question}}

**분석 지침:**
1. **질문 유형 분류:**
   - specific_domain: 특정 도메인에만 관련된 질문
   - multi_domain: 여러 도메인에 걸친 복합 질문
   - general: 도메인 구분이 명확하지 않은 일반 질문
   - web_search: 최신 정보나 외부 정보가 필요한 질문
   - unclear: 질문 의도가 불분명한 질문

2. **도메인 선택:**
   - primary_domains: 가장 관련성이 높은 도메인들 (1-3개)
   - secondary_domains: 부차적으로 관련된 도메인들 (0-2개)
   - 키워드 매칭과 의미적 유사성을 모두 고려

3. **웹 검색 필요성 판단:**
   - "최신", "현재", "2024", "뉴스", "동향" 등의 키워드
   - 시간 민감적 정보 요구
   - 내부 문서에 없을 것 같은 외부 정보

4. **긴급도 평가:**
   - high: 업무 차단, 긴급 처리 필요
   - medium: 일반적인 업무 문의
   - low: 참고용, 교육용 질문

다음 JSON 형식으로 응답하세요:
{{
    "question_type": "specific_domain|multi_domain|general|web_search|unclear",
    "primary_domains": ["domain1", "domain2"],
    "secondary_domains": ["domain3"],
    "confidence_score": 0.0-1.0,
    "urgency_level": "high|medium|low",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "intent": "질문의 핵심 의도를 한 문장으로",
    "complexity": "simple|medium|complex",
    "requires_web_search": true|false,
    "reasoning": "분석 근거를 상세히 설명"
}}
""")

        self.analysis_chain = (
            self.analysis_prompt
            | self.llm.llm
            | JsonOutputParser()
        )

    def analyze_question(self, question: str) -> QuestionAnalysis:
        """질문 분석 수행"""
        try:
            # LLM을 통한 고급 분석
            result = self.analysis_chain.invoke({"question": question})

            # 기본값 처리 및 검증
            question_type = QuestionType(result.get("question_type", "general"))
            urgency_level = UrgencyLevel(result.get("urgency_level", "medium"))

            # 도메인 검증 및 정리
            primary_domains = self._validate_domains(result.get("primary_domains", []))
            secondary_domains = self._validate_domains(result.get("secondary_domains", []))

            # 규칙 기반 보완 분석
            rule_based_analysis = self._rule_based_analysis(question)

            # 결과 통합
            final_analysis = QuestionAnalysis(
                question=question,
                question_type=question_type,
                primary_domains=primary_domains or rule_based_analysis['domains'],
                secondary_domains=secondary_domains,
                confidence_score=max(0.0, min(1.0, result.get("confidence_score", 0.7))),
                urgency_level=urgency_level,
                keywords=result.get("keywords", []) + rule_based_analysis['keywords'],
                intent=result.get("intent", "질문 분석"),
                complexity=result.get("complexity", "medium"),
                requires_web_search=result.get("requires_web_search", False) or rule_based_analysis['web_search'],
                reasoning=result.get("reasoning", "LLM 기반 분석"),
                metadata={
                    'llm_analysis': result,
                    'rule_based_analysis': rule_based_analysis,
                    'domain_count': len(primary_domains) + len(secondary_domains)
                }
            )

            logger.info(f"✅ 질문 분석 완료: {question_type.value}, 도메인 {len(primary_domains)}개")
            return final_analysis

        except Exception as e:
            logger.error(f"질문 분석 실패: {e}")
            # 폴백: 규칙 기반 분석만 사용
            return self._fallback_analysis(question)

    def _validate_domains(self, domains: List[str]) -> List[str]:
        """도메인 목록 검증"""
        valid_domains = []
        all_domains = list(config.domain_list) + ['web_search']

        for domain in domains:
            if domain in all_domains:
                valid_domains.append(domain)
            else:
                logger.warning(f"잘못된 도메인 무시: {domain}")

        return valid_domains

    def _rule_based_analysis(self, question: str) -> Dict[str, Any]:
        """규칙 기반 보완 분석"""
        question_lower = question.lower()
        result = {
            'domains': [],
            'keywords': [],
            'web_search': False
        }

        # 도메인별 키워드 매칭
        for domain, info in self.domain_info.items():
            if domain == 'web_search':
                continue

            matches = sum(1 for keyword in info['keywords']
                         if keyword.lower() in question_lower)

            if matches > 0:
                result['domains'].append(domain)
                # 매칭된 키워드 추출
                matched_keywords = [kw for kw in info['keywords']
                                  if kw.lower() in question_lower]
                result['keywords'].extend(matched_keywords)

        # 웹 검색 필요성 판단
        web_indicators = [
            '최신', '현재', '오늘', '2024', '뉴스', '동향', '트렌드',
            '업데이트', '변화', '발표', '출시', 'latest', 'current', 'news'
        ]

        for indicator in web_indicators:
            if indicator in question_lower:
                result['web_search'] = True
                break

        # 기본 도메인 (아무것도 매칭되지 않으면)
        if not result['domains']:
            result['domains'] = ['business_policy']  # 기본적으로 비즈니스 정책으로

        return result

    def _fallback_analysis(self, question: str) -> QuestionAnalysis:
        """폴백 분석 (LLM 분석 실패 시)"""
        rule_based = self._rule_based_analysis(question)

        return QuestionAnalysis(
            question=question,
            question_type=QuestionType.GENERAL,
            primary_domains=rule_based['domains'],
            secondary_domains=[],
            confidence_score=0.5,
            urgency_level=UrgencyLevel.MEDIUM,
            keywords=rule_based['keywords'],
            intent="규칙 기반 분석",
            complexity="medium",
            requires_web_search=rule_based['web_search'],
            reasoning="LLM 분석 실패로 규칙 기반 분석 사용",
            metadata={'fallback': True}
        )

    def batch_analyze(self, questions: List[str]) -> List[QuestionAnalysis]:
        """여러 질문 일괄 분석"""
        results = []

        for i, question in enumerate(questions):
            try:
                logger.info(f"질문 분석 중 {i+1}/{len(questions)}: {question[:50]}...")
                analysis = self.analyze_question(question)
                results.append(analysis)
            except Exception as e:
                logger.error(f"질문 {i+1} 분석 실패: {e}")
                results.append(self._fallback_analysis(question))

        logger.info(f"✅ 일괄 질문 분석 완료: {len(results)}개")
        return results

    def get_domain_statistics(self, analyses: List[QuestionAnalysis]) -> Dict[str, Any]:
        """분석 결과 통계"""
        if not analyses:
            return {}

        stats = {
            'total_questions': len(analyses),
            'question_types': {},
            'domain_frequency': {},
            'urgency_distribution': {},
            'avg_confidence': 0.0,
            'web_search_required': 0,
            'complexity_distribution': {}
        }

        # 통계 수집
        total_confidence = 0.0

        for analysis in analyses:
            # 질문 유형 분포
            qtype = analysis.question_type.value
            stats['question_types'][qtype] = stats['question_types'].get(qtype, 0) + 1

            # 도메인 빈도
            for domain in analysis.primary_domains + analysis.secondary_domains:
                stats['domain_frequency'][domain] = stats['domain_frequency'].get(domain, 0) + 1

            # 긴급도 분포
            urgency = analysis.urgency_level.value
            stats['urgency_distribution'][urgency] = stats['urgency_distribution'].get(urgency, 0) + 1

            # 복잡도 분포
            complexity = analysis.complexity
            stats['complexity_distribution'][complexity] = stats['complexity_distribution'].get(complexity, 0) + 1

            # 확신도 합계
            total_confidence += analysis.confidence_score

            # 웹 검색 필요 수
            if analysis.requires_web_search:
                stats['web_search_required'] += 1

        # 평균 확신도
        stats['avg_confidence'] = total_confidence / len(analyses)

        return stats

    def suggest_improvements(self, analysis: QuestionAnalysis) -> List[str]:
        """질문 개선 제안"""
        suggestions = []

        if analysis.confidence_score < 0.6:
            suggestions.append("질문을 더 구체적으로 작성해보세요")

        if analysis.question_type == QuestionType.UNCLEAR:
            suggestions.append("질문의 목적이나 원하는 답변을 명확히 해주세요")

        if not analysis.primary_domains:
            suggestions.append("관련 분야나 카테고리를 명시해보세요")

        if len(analysis.keywords) < 2:
            suggestions.append("핵심 키워드를 더 포함해보세요")

        return suggestions


# 편의 함수들
def analyze_question(question: str, llm: Optional[CoolStayLLM] = None) -> QuestionAnalysis:
    """질문 분석 편의 함수"""
    analyzer = QuestionAnalyzer(llm)
    return analyzer.analyze_question(question)


def get_best_domains(question: str, max_domains: int = 2) -> List[str]:
    """질문에 가장 적합한 도메인 반환"""
    analysis = analyze_question(question)
    all_domains = analysis.primary_domains + analysis.secondary_domains
    return all_domains[:max_domains]


def needs_web_search(question: str) -> bool:
    """웹 검색 필요 여부 판단"""
    analysis = analyze_question(question)
    return analysis.requires_web_search


if __name__ == "__main__":
    # 질문 분석기 테스트
    print("🔍 CoolStay 질문 분석기 테스트")
    print("=" * 50)

    analyzer = QuestionAnalyzer()

    # 테스트 질문들
    test_questions = [
        "연차 휴가는 어떻게 신청하나요?",
        "React 컴포넌트 개발 가이드라인이 있나요?",
        "CI/CD 파이프라인 설정 방법을 알려주세요",
        "최신 AI 개발 동향이 궁금합니다",
        "회사 정책과 기술 표준을 모두 알고 싶어요"
    ]

    print(f"🧪 {len(test_questions)}개 질문 분석 테스트:\n")

    analyses = []
    for i, question in enumerate(test_questions, 1):
        print(f"📋 질문 {i}: {question}")

        try:
            analysis = analyzer.analyze_question(question)
            analyses.append(analysis)

            print(f"   유형: {analysis.question_type.value}")
            print(f"   주 도메인: {', '.join(analysis.primary_domains)}")
            print(f"   부 도메인: {', '.join(analysis.secondary_domains) if analysis.secondary_domains else '없음'}")
            print(f"   웹 검색: {'필요' if analysis.requires_web_search else '불필요'}")
            print(f"   확신도: {analysis.confidence_score:.2f}")
            print(f"   긴급도: {analysis.urgency_level.value}")
            print(f"   복잡도: {analysis.complexity}")
            print(f"   키워드: {', '.join(analysis.keywords[:3])}")

            # 개선 제안
            suggestions = analyzer.suggest_improvements(analysis)
            if suggestions:
                print(f"   💡 개선 제안: {suggestions[0]}")

        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")

        print()

    # 통계 분석
    if analyses:
        print("📊 분석 통계:")
        stats = analyzer.get_domain_statistics(analyses)

        print(f"   평균 확신도: {stats['avg_confidence']:.2f}")
        print(f"   웹 검색 필요: {stats['web_search_required']}/{stats['total_questions']}개")

        print(f"   도메인 빈도:")
        for domain, count in sorted(stats['domain_frequency'].items(), key=lambda x: x[1], reverse=True):
            print(f"     - {domain}: {count}회")

        print(f"   질문 유형 분포:")
        for qtype, count in stats['question_types'].items():
            print(f"     - {qtype}: {count}개")
    else:
        print("❌ 분석된 질문이 없습니다.")


# 편의 함수들
def create_question_analyzer(config: Optional[CoolStayConfig] = None) -> QuestionAnalyzer:
    """질문 분석기를 생성합니다."""
    return QuestionAnalyzer(config)


def analyze_question_simple(
    question: str,
    config: Optional[CoolStayConfig] = None
) -> QuestionAnalysis:
    """간단한 질문 분석 함수"""
    analyzer = QuestionAnalyzer(config)
    return analyzer.analyze_question(question)