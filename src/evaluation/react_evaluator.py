"""
CoolStay RAG ReAct 평가 모듈

ReAct (Reasoning + Acting) 패러다임을 사용하여
응답의 품질을 6차원으로 평가하는 모듈입니다.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from ..core.config import CoolStayConfig
from ..core.llm import CoolStayLLM, get_default_llm

logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """평가 차원"""
    RELEVANCE = "relevance"        # 관련성
    ACCURACY = "accuracy"          # 정확성
    COMPLETENESS = "completeness"  # 완전성
    CLARITY = "clarity"           # 명확성
    USEFULNESS = "usefulness"     # 실용성
    TIMELINESS = "timeliness"     # 시의적절성


class EvaluationScore(Enum):
    """평가 점수"""
    EXCELLENT = 10    # 우수
    GOOD = 8         # 좋음
    ACCEPTABLE = 6   # 수용 가능
    POOR = 4        # 부족
    VERY_POOR = 2   # 매우 부족
    UNACCEPTABLE = 0 # 수용 불가


@dataclass
class DimensionEvaluation:
    """차원별 평가 결과"""
    dimension: EvaluationDimension
    score: int  # 0-10 점수
    reasoning: str
    evidence: List[str]
    suggestions: List[str]


@dataclass
class ReActEvaluationResult:
    """ReAct 평가 결과"""
    question: str
    answer: str
    dimension_scores: Dict[EvaluationDimension, DimensionEvaluation]
    overall_score: int  # 60점 만점
    grade: str  # A+, A, B+, B, C+, C, D, F
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    evaluation_reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime


class ReActEvaluationAgent:
    """
    ReAct 평가 에이전트

    Reasoning과 Acting을 결합하여 체계적이고 깊이 있는
    응답 평가를 수행합니다.
    """

    def __init__(self, config: Optional[CoolStayConfig] = None):
        """
        Args:
            config: CoolStay 설정 객체
        """
        self.config = config or CoolStayConfig()
        self.llm = get_default_llm()

        # 평가 기준 설정
        self._setup_evaluation_criteria()

        # 평가 프롬프트 설정
        self._setup_evaluation_prompts()

        # 등급 체계 설정
        self._setup_grading_system()

        logger.info("ReActEvaluationAgent 초기화 완료")

    def _setup_evaluation_criteria(self):
        """평가 기준을 설정합니다."""
        self.evaluation_criteria = {
            EvaluationDimension.RELEVANCE: {
                "description": "질문과의 관련성 및 핵심 포인트 다루기",
                "excellent": "질문의 모든 요소에 정확하게 답변하고 핵심을 완벽히 파악",
                "good": "질문의 대부분 요소에 적절히 답변하고 핵심을 잘 파악",
                "acceptable": "질문의 기본 요소에는 답변하나 일부 누락",
                "poor": "질문과 부분적으로만 관련된 답변",
                "very_poor": "질문과의 관련성이 낮음",
                "unacceptable": "질문과 무관한 답변"
            },
            EvaluationDimension.ACCURACY: {
                "description": "정보의 정확성 및 신뢰성",
                "excellent": "모든 정보가 정확하고 근거가 명확함",
                "good": "대부분의 정보가 정확하고 신뢰할 만함",
                "acceptable": "기본 정보는 정확하나 세부사항에서 일부 부정확",
                "poor": "일부 정보가 부정확하거나 근거 부족",
                "very_poor": "다수의 부정확한 정보 포함",
                "unacceptable": "대부분의 정보가 부정확하거나 오해의 소지"
            },
            EvaluationDimension.COMPLETENESS: {
                "description": "답변의 완전성 및 포괄성",
                "excellent": "모든 중요한 측면을 포괄하고 충분히 설명",
                "good": "대부분의 중요한 측면을 다루고 적절히 설명",
                "acceptable": "기본적인 내용은 포함하나 일부 측면 누락",
                "poor": "중요한 정보나 측면이 상당히 누락",
                "very_poor": "매우 제한적이고 불완전한 답변",
                "unacceptable": "답변이 너무 불완전하여 실용성 없음"
            },
            EvaluationDimension.CLARITY: {
                "description": "답변의 명확성 및 이해하기 쉬운 정도",
                "excellent": "매우 명확하고 구조화되어 이해하기 쉬움",
                "good": "명확하고 논리적으로 구성되어 이해 용이",
                "acceptable": "기본적으로 이해 가능하나 일부 모호함",
                "poor": "구조가 불분명하고 이해하기 어려운 부분 있음",
                "very_poor": "매우 혼란스럽고 이해하기 어려움",
                "unacceptable": "전혀 이해할 수 없는 답변"
            },
            EvaluationDimension.USEFULNESS: {
                "description": "실무에서의 활용 가능성 및 실용성",
                "excellent": "즉시 실무에 적용 가능하고 매우 유용함",
                "good": "실무에 활용하기 좋고 유용한 정보 제공",
                "acceptable": "기본적인 실용성은 있으나 제한적",
                "poor": "실용성이 떨어지고 활용하기 어려움",
                "very_poor": "실무 적용이 어렵고 유용성 낮음",
                "unacceptable": "전혀 실용적이지 않음"
            },
            EvaluationDimension.TIMELINESS: {
                "description": "정보의 시의적절성 및 최신성",
                "excellent": "최신 정보가 반영되고 시의적절함",
                "good": "대체로 최신 정보이고 적절한 시점의 정보",
                "acceptable": "기본적인 최신성은 유지하나 일부 구식 정보",
                "poor": "일부 정보가 구식이고 시의성 부족",
                "very_poor": "대부분 구식 정보이고 시의적절하지 않음",
                "unacceptable": "완전히 구식이고 부적절한 정보"
            }
        }

    def _setup_evaluation_prompts(self):
        """평가용 프롬프트들을 설정합니다."""

        self.main_evaluation_prompt = """
당신은 전문적인 응답 평가자입니다. ReAct (Reasoning + Acting) 방식을 사용하여 단계적으로 응답을 평가해주세요.

**평가 대상**:
질문: {question}
응답: {answer}

**평가 차원** (각 10점 만점):
1. **관련성 (Relevance)**: 질문과의 관련성 및 핵심 포인트 다루기
2. **정확성 (Accuracy)**: 정보의 정확성 및 신뢰성
3. **완전성 (Completeness)**: 답변의 완전성 및 포괄성
4. **명확성 (Clarity)**: 답변의 명확성 및 이해하기 쉬운 정도
5. **실용성 (Usefulness)**: 실무에서의 활용 가능성 및 실용성
6. **시의성 (Timeliness)**: 정보의 시의적절성 및 최신성

**평가 단계**:
1. **Reasoning**: 각 차원별로 왜 해당 점수를 주는지 근거와 함께 분석
2. **Acting**: 구체적인 점수 부여 및 개선 제안

**출력 형식** (JSON):
```json
{{
    "step_by_step_reasoning": {{
        "relevance": {{
            "analysis": "관련성 분석 내용",
            "evidence": ["근거1", "근거2"],
            "score": 8,
            "reasoning": "점수 부여 이유"
        }},
        "accuracy": {{...}},
        "completeness": {{...}},
        "clarity": {{...}},
        "usefulness": {{...}},
        "timeliness": {{...}}
    }},
    "overall_assessment": {{
        "total_score": 48,
        "grade": "B+",
        "strengths": ["강점1", "강점2"],
        "weaknesses": ["약점1", "약점2"],
        "improvement_suggestions": ["개선사항1", "개선사항2"]
    }},
    "evaluation_summary": "전반적인 평가 요약"
}}
```

신중하고 객관적으로 평가해주세요.
"""

        self.dimension_focus_prompt = """
**{dimension_name}** 차원에 집중하여 다음 응답을 평가해주세요:

**평가 기준**: {criteria_description}

**질문**: {question}
**응답**: {answer}

**세부 평가 기준**:
- 우수 (9-10점): {excellent_criteria}
- 좋음 (7-8점): {good_criteria}
- 수용 가능 (5-6점): {acceptable_criteria}
- 부족 (3-4점): {poor_criteria}
- 매우 부족 (1-2점): {very_poor_criteria}
- 수용 불가 (0점): {unacceptable_criteria}

구체적인 근거와 함께 점수를 매기고, 개선 방안을 제시해주세요.

출력 형식:
```json
{{
    "score": 7,
    "reasoning": "점수 부여의 구체적 근거",
    "evidence": ["근거가 되는 구체적 사례1", "근거가 되는 구체적 사례2"],
    "suggestions": ["개선 제안1", "개선 제안2"]
}}
```
"""

    def _setup_grading_system(self):
        """등급 체계를 설정합니다."""
        self.grade_thresholds = {
            'A+': 57,  # 95% 이상
            'A':  54,  # 90% 이상
            'B+': 51,  # 85% 이상
            'B':  48,  # 80% 이상
            'C+': 42,  # 70% 이상
            'C':  36,  # 60% 이상
            'D':  30,  # 50% 이상
            'F':  0    # 50% 미만
        }

    def evaluate(self, question: str, answer: str, **kwargs) -> ReActEvaluationResult:
        """응답을 ReAct 방식으로 평가합니다."""
        try:
            # 1. 전체 평가 수행
            main_evaluation = self._perform_main_evaluation(question, answer)

            # 2. 차원별 세부 평가
            dimension_evaluations = self._perform_dimension_evaluations(question, answer)

            # 3. 결과 통합
            result = self._integrate_evaluation_results(
                question, answer, main_evaluation, dimension_evaluations, **kwargs
            )

            return result

        except Exception as e:
            logger.error(f"ReAct 평가 실패: {e}")
            # 기본 평가 반환
            return self._create_fallback_evaluation(question, answer, str(e))

    def _perform_main_evaluation(self, question: str, answer: str) -> Dict[str, Any]:
        """전체적인 평가를 수행합니다."""
        try:
            evaluation_result = self.llm.invoke(
                self.main_evaluation_prompt.format(
                    question=question,
                    answer=answer
                )
            )

            # JSON 파싱 시도
            try:
                parsed_result = json.loads(evaluation_result.content)
                return parsed_result
            except json.JSONDecodeError:
                logger.warning("메인 평가 JSON 파싱 실패, 텍스트 분석 사용")
                return self._parse_text_evaluation(evaluation_result.content)

        except Exception as e:
            logger.error(f"메인 평가 실패: {e}")
            return self._create_basic_evaluation(question, answer)

    def _perform_dimension_evaluations(
        self,
        question: str,
        answer: str
    ) -> Dict[EvaluationDimension, DimensionEvaluation]:
        """각 차원별 세부 평가를 수행합니다."""
        dimension_evaluations = {}

        for dimension in EvaluationDimension:
            try:
                criteria = self.evaluation_criteria[dimension]

                evaluation_result = self.llm.invoke(
                    self.dimension_focus_prompt.format(
                        dimension_name=dimension.value,
                        criteria_description=criteria["description"],
                        question=question,
                        answer=answer,
                        excellent_criteria=criteria["excellent"],
                        good_criteria=criteria["good"],
                        acceptable_criteria=criteria["acceptable"],
                        poor_criteria=criteria["poor"],
                        very_poor_criteria=criteria["very_poor"],
                        unacceptable_criteria=criteria["unacceptable"]
                    )
                )

                # JSON 파싱
                try:
                    parsed_result = json.loads(evaluation_result.content)

                    dimension_evaluations[dimension] = DimensionEvaluation(
                        dimension=dimension,
                        score=parsed_result.get('score', 5),
                        reasoning=parsed_result.get('reasoning', ''),
                        evidence=parsed_result.get('evidence', []),
                        suggestions=parsed_result.get('suggestions', [])
                    )

                except json.JSONDecodeError:
                    # 텍스트에서 점수 추출 시도
                    score = self._extract_score_from_text(evaluation_result.content)
                    dimension_evaluations[dimension] = DimensionEvaluation(
                        dimension=dimension,
                        score=score,
                        reasoning=evaluation_result.content[:200] + "...",
                        evidence=[],
                        suggestions=[]
                    )

            except Exception as e:
                logger.error(f"차원 {dimension.value} 평가 실패: {e}")
                dimension_evaluations[dimension] = DimensionEvaluation(
                    dimension=dimension,
                    score=5,  # 기본 점수
                    reasoning=f"평가 실패: {str(e)}",
                    evidence=[],
                    suggestions=[]
                )

        return dimension_evaluations

    def _integrate_evaluation_results(
        self,
        question: str,
        answer: str,
        main_evaluation: Dict[str, Any],
        dimension_evaluations: Dict[EvaluationDimension, DimensionEvaluation],
        **kwargs
    ) -> ReActEvaluationResult:
        """평가 결과들을 통합합니다."""

        # 총점 계산
        total_score = sum(eval_result.score for eval_result in dimension_evaluations.values())

        # 등급 계산
        grade = self._calculate_grade(total_score)

        # 강점과 약점 추출
        strengths, weaknesses = self._extract_strengths_weaknesses(dimension_evaluations)

        # 개선 제안 통합
        improvement_suggestions = self._integrate_suggestions(dimension_evaluations)

        # 평가 추론 생성
        evaluation_reasoning = self._generate_evaluation_reasoning(
            dimension_evaluations, total_score, grade
        )

        return ReActEvaluationResult(
            question=question,
            answer=answer,
            dimension_scores=dimension_evaluations,
            overall_score=total_score,
            grade=grade,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvement_suggestions,
            evaluation_reasoning=evaluation_reasoning,
            metadata={
                'evaluation_method': 'ReAct',
                'llm_model': self.llm.model_name,
                'evaluation_version': '1.0',
                **kwargs
            },
            timestamp=datetime.now()
        )

    def _calculate_grade(self, total_score: int) -> str:
        """총점을 기반으로 등급을 계산합니다."""
        for grade, threshold in self.grade_thresholds.items():
            if total_score >= threshold:
                return grade
        return 'F'

    def _extract_strengths_weaknesses(
        self,
        dimension_evaluations: Dict[EvaluationDimension, DimensionEvaluation]
    ) -> Tuple[List[str], List[str]]:
        """강점과 약점을 추출합니다."""
        strengths = []
        weaknesses = []

        for dimension, evaluation in dimension_evaluations.items():
            if evaluation.score >= 8:
                strengths.append(f"{dimension.value}: {evaluation.reasoning[:50]}...")
            elif evaluation.score <= 5:
                weaknesses.append(f"{dimension.value}: {evaluation.reasoning[:50]}...")

        return strengths, weaknesses

    def _integrate_suggestions(
        self,
        dimension_evaluations: Dict[EvaluationDimension, DimensionEvaluation]
    ) -> List[str]:
        """개선 제안들을 통합합니다."""
        all_suggestions = []

        for evaluation in dimension_evaluations.values():
            all_suggestions.extend(evaluation.suggestions)

        # 중복 제거 및 상위 5개 선택
        unique_suggestions = list(set(all_suggestions))
        return unique_suggestions[:5]

    def _generate_evaluation_reasoning(
        self,
        dimension_evaluations: Dict[EvaluationDimension, DimensionEvaluation],
        total_score: int,
        grade: str
    ) -> str:
        """평가 추론을 생성합니다."""
        reasoning_parts = []

        reasoning_parts.append(f"총 {total_score}/60점으로 {grade} 등급에 해당합니다.")

        # 최고/최저 차원 언급
        scores = [(dim.value, eval_result.score) for dim, eval_result in dimension_evaluations.items()]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_dimension, best_score = scores[0]
        worst_dimension, worst_score = scores[-1]

        reasoning_parts.append(f"가장 우수한 영역은 {best_dimension}({best_score}점)이며, ")
        reasoning_parts.append(f"가장 개선이 필요한 영역은 {worst_dimension}({worst_score}점)입니다.")

        # 전반적인 품질 평가
        if total_score >= 54:
            reasoning_parts.append("전반적으로 매우 우수한 응답입니다.")
        elif total_score >= 48:
            reasoning_parts.append("전반적으로 좋은 품질의 응답입니다.")
        elif total_score >= 36:
            reasoning_parts.append("기본적인 수준은 만족하나 개선이 필요합니다.")
        else:
            reasoning_parts.append("상당한 개선이 필요한 응답입니다.")

        return " ".join(reasoning_parts)

    def _parse_text_evaluation(self, text: str) -> Dict[str, Any]:
        """텍스트 평가 결과를 파싱합니다."""
        # 기본 구조 반환
        return {
            "step_by_step_reasoning": {},
            "overall_assessment": {
                "total_score": 30,
                "grade": "C",
                "strengths": ["텍스트 분석으로 인한 제한적 평가"],
                "weaknesses": ["JSON 파싱 실패"],
                "improvement_suggestions": ["평가 형식 개선 필요"]
            },
            "evaluation_summary": "평가 파싱에 문제가 있어 제한적으로 평가되었습니다."
        }

    def _create_basic_evaluation(self, question: str, answer: str) -> Dict[str, Any]:
        """기본 평가를 생성합니다."""
        return {
            "step_by_step_reasoning": {},
            "overall_assessment": {
                "total_score": 25,
                "grade": "C-",
                "strengths": ["기본적인 응답 제공"],
                "weaknesses": ["평가 시스템 오류"],
                "improvement_suggestions": ["평가 시스템 점검 필요"]
            },
            "evaluation_summary": "평가 시스템 오류로 인한 기본 평가입니다."
        }

    def _extract_score_from_text(self, text: str) -> int:
        """텍스트에서 점수를 추출합니다."""
        import re

        # 점수 패턴 찾기
        score_patterns = [
            r'점수[:\s]*(\d+)',
            r'score[:\s]*(\d+)',
            r'(\d+)\s*점',
            r'(\d+)/10'
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    score = int(matches[0])
                    if 0 <= score <= 10:
                        return score
                except ValueError:
                    continue

        return 5  # 기본 점수

    def _create_fallback_evaluation(
        self,
        question: str,
        answer: str,
        error_message: str
    ) -> ReActEvaluationResult:
        """폴백 평가를 생성합니다."""
        # 모든 차원에 대해 기본 평가 생성
        dimension_evaluations = {}
        for dimension in EvaluationDimension:
            dimension_evaluations[dimension] = DimensionEvaluation(
                dimension=dimension,
                score=5,
                reasoning=f"평가 오류로 인한 기본 점수: {error_message}",
                evidence=[],
                suggestions=["평가 시스템 점검 필요"]
            )

        return ReActEvaluationResult(
            question=question,
            answer=answer,
            dimension_scores=dimension_evaluations,
            overall_score=30,
            grade="C",
            strengths=["기본적인 응답 제공"],
            weaknesses=["평가 시스템 오류"],
            improvement_suggestions=["평가 시스템 점검 및 개선"],
            evaluation_reasoning=f"평가 시스템 오류: {error_message}",
            metadata={'error': True, 'error_message': error_message},
            timestamp=datetime.now()
        )

    def batch_evaluate(
        self,
        qa_pairs: List[Tuple[str, str]],
        **kwargs
    ) -> List[ReActEvaluationResult]:
        """여러 질문-답변 쌍을 일괄 평가합니다."""
        results = []

        for i, (question, answer) in enumerate(qa_pairs):
            logger.info(f"평가 진행 중: {i+1}/{len(qa_pairs)}")
            result = self.evaluate(question, answer, batch_index=i, **kwargs)
            results.append(result)

        return results

    def get_evaluation_statistics(
        self,
        evaluations: List[ReActEvaluationResult]
    ) -> Dict[str, Any]:
        """평가 결과의 통계를 계산합니다."""
        if not evaluations:
            return {}

        # 총점 통계
        total_scores = [eval_result.overall_score for eval_result in evaluations]
        avg_score = sum(total_scores) / len(total_scores)

        # 차원별 평균 점수
        dimension_averages = {}
        for dimension in EvaluationDimension:
            dimension_scores = [
                eval_result.dimension_scores[dimension].score
                for eval_result in evaluations
            ]
            dimension_averages[dimension.value] = sum(dimension_scores) / len(dimension_scores)

        # 등급 분포
        grade_distribution = {}
        for eval_result in evaluations:
            grade = eval_result.grade
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

        return {
            'total_evaluations': len(evaluations),
            'average_score': round(avg_score, 2),
            'score_range': (min(total_scores), max(total_scores)),
            'dimension_averages': dimension_averages,
            'grade_distribution': grade_distribution,
            'success_rate': len([e for e in evaluations if e.overall_score >= 36]) / len(evaluations)
        }


# 편의 함수들
def create_react_evaluator(config: Optional[CoolStayConfig] = None) -> ReActEvaluationAgent:
    """ReAct 평가 에이전트를 생성합니다."""
    return ReActEvaluationAgent(config)


def evaluate_simple(
    question: str,
    answer: str,
    config: Optional[CoolStayConfig] = None
) -> ReActEvaluationResult:
    """간단한 응답 평가 함수"""
    evaluator = ReActEvaluationAgent(config)
    return evaluator.evaluate(question, answer)


def convert_to_dict(evaluation: ReActEvaluationResult) -> Dict[str, Any]:
    """평가 결과를 딕셔너리로 변환합니다."""
    result = asdict(evaluation)

    # Enum과 datetime 처리
    result['dimension_scores'] = {
        dim.value: asdict(eval_result)
        for dim, eval_result in evaluation.dimension_scores.items()
    }

    # dimension 필드 처리
    for dim_result in result['dimension_scores'].values():
        dim_result['dimension'] = dim_result['dimension'].value

    result['timestamp'] = evaluation.timestamp.isoformat()

    return result