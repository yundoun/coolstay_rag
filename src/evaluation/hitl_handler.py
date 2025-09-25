"""
CoolStay RAG Human-in-the-Loop (HITL) 핸들러 모듈

인간 피드백을 수집하고 처리하여 시스템 성능을 지속적으로 개선하는 모듈입니다.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid

from ..core.config import CoolStayConfig
from .react_evaluator import ReActEvaluationResult, ReActEvaluationAgent

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """피드백 유형"""
    RATING = "rating"              # 평점 피드백
    CORRECTION = "correction"      # 수정 피드백
    PREFERENCE = "preference"      # 선호도 피드백
    SUGGESTION = "suggestion"      # 제안 피드백
    VALIDATION = "validation"      # 검증 피드백


class FeedbackSource(Enum):
    """피드백 소스"""
    USER_INTERFACE = "user_interface"    # 사용자 인터페이스
    API_ENDPOINT = "api_endpoint"        # API 엔드포인트
    BATCH_REVIEW = "batch_review"        # 일괄 검토
    EXPERT_REVIEW = "expert_review"      # 전문가 검토


class InteractionContext(Enum):
    """상호작용 컨텍스트"""
    REAL_TIME = "real_time"        # 실시간 상호작용
    ASYNCHRONOUS = "asynchronous"  # 비동기 상호작용
    BATCH_PROCESS = "batch_process" # 배치 프로세스


@dataclass
class HumanFeedback:
    """인간 피드백"""
    feedback_id: str
    question: str
    original_answer: str
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    feedback_data: Dict[str, Any]  # 피드백 내용
    context: Dict[str, Any]        # 컨텍스트 정보
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class HITLSession:
    """HITL 세션"""
    session_id: str
    start_time: datetime
    interaction_context: InteractionContext
    feedbacks: List[HumanFeedback]
    metadata: Dict[str, Any]
    end_time: Optional[datetime] = None


@dataclass
class FeedbackAnalysis:
    """피드백 분석 결과"""
    total_feedbacks: int
    feedback_distribution: Dict[FeedbackType, int]
    average_rating: Optional[float]
    common_issues: List[str]
    improvement_areas: List[str]
    positive_aspects: List[str]
    trend_analysis: Dict[str, Any]
    recommendations: List[str]


class HITLInterface:
    """
    Human-in-the-Loop 인터페이스

    인간 피드백을 실시간으로 수집하고 처리하는 인터페이스입니다.
    """

    def __init__(
        self,
        config: Optional[CoolStayConfig] = None,
        feedback_callbacks: Optional[Dict[FeedbackType, Callable]] = None
    ):
        """
        Args:
            config: CoolStay 설정 객체
            feedback_callbacks: 피드백 타입별 콜백 함수들
        """
        self.config = config or CoolStayConfig()
        self.feedback_callbacks = feedback_callbacks or {}

        # 피드백 저장소
        self.feedback_storage: List[HumanFeedback] = []
        self.active_sessions: Dict[str, HITLSession] = {}

        # 평가 에이전트
        self.react_evaluator = ReActEvaluationAgent(config)

        # 인터페이스 설정
        self._setup_interface_config()

        logger.info("HITLInterface 초기화 완료")

    def _setup_interface_config(self):
        """인터페이스 설정을 초기화합니다."""
        self.interface_config = {
            'rating_scale': (1, 5),  # 평점 범위
            'feedback_timeout': 300,  # 피드백 대기 시간 (초)
            'max_concurrent_sessions': 10,  # 최대 동시 세션
            'auto_save_interval': 60,  # 자동 저장 간격 (초)
            'feedback_validation': True,  # 피드백 검증 활성화
        }

        self.prompt_templates = {
            'rating_request': """
다음 응답에 대해 평가해주세요:

질문: {question}
응답: {answer}

1-5점 척도로 평가해주세요:
1: 매우 부족
2: 부족
3: 보통
4: 좋음
5: 매우 좋음

평점: ___
이유: ___________________
""",
            'correction_request': """
다음 응답에서 수정이 필요한 부분을 알려주세요:

질문: {question}
원본 응답: {answer}

수정할 부분:
수정된 내용:
수정 이유:
""",
            'preference_request': """
다음 두 응답 중 어느 것이 더 좋은지 선택해주세요:

질문: {question}

응답 A: {answer_a}
응답 B: {answer_b}

선택: A / B
이유: ___________________
"""
        }

    async def start_hitl_session(
        self,
        session_id: Optional[str] = None,
        context: InteractionContext = InteractionContext.REAL_TIME,
        **metadata
    ) -> HITLSession:
        """HITL 세션을 시작합니다."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        session = HITLSession(
            session_id=session_id,
            start_time=datetime.now(),
            interaction_context=context,
            feedbacks=[],
            metadata=metadata
        )

        self.active_sessions[session_id] = session
        logger.info(f"HITL 세션 시작: {session_id}")

        return session

    async def end_hitl_session(self, session_id: str) -> HITLSession:
        """HITL 세션을 종료합니다."""
        if session_id not in self.active_sessions:
            raise ValueError(f"활성 세션을 찾을 수 없음: {session_id}")

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()

        # 세션 피드백을 전체 저장소에 추가
        self.feedback_storage.extend(session.feedbacks)

        # 세션 정리
        del self.active_sessions[session_id]

        logger.info(f"HITL 세션 종료: {session_id}, 피드백 수: {len(session.feedbacks)}")
        return session

    async def collect_rating_feedback(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout: Optional[int] = None,
        **context
    ) -> HumanFeedback:
        """평점 피드백을 수집합니다."""
        feedback_id = str(uuid.uuid4())
        timeout = timeout or self.interface_config['feedback_timeout']

        # 사용자에게 평점 요청
        prompt = self.prompt_templates['rating_request'].format(
            question=question,
            answer=answer
        )

        print("\n" + "="*50)
        print("📊 평점 피드백 요청")
        print("="*50)
        print(prompt)
        print("="*50)

        try:
            # 실제 구현에서는 웹 인터페이스나 API를 통해 입력 받음
            # 여기서는 시뮬레이션을 위해 기본값 사용
            rating_data = await self._simulate_rating_input(timeout)

            feedback = HumanFeedback(
                feedback_id=feedback_id,
                question=question,
                original_answer=answer,
                feedback_type=FeedbackType.RATING,
                feedback_source=FeedbackSource.USER_INTERFACE,
                feedback_data=rating_data,
                context=context,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id
            )

            # 세션에 추가
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].feedbacks.append(feedback)

            # 콜백 실행
            if FeedbackType.RATING in self.feedback_callbacks:
                await self.feedback_callbacks[FeedbackType.RATING](feedback)

            logger.info(f"평점 피드백 수집됨: {rating_data.get('rating', 'N/A')}")
            return feedback

        except asyncio.TimeoutError:
            logger.warning(f"평점 피드백 시간초과: {feedback_id}")
            return self._create_timeout_feedback(feedback_id, question, answer, FeedbackType.RATING)

    async def collect_correction_feedback(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **context
    ) -> HumanFeedback:
        """수정 피드백을 수집합니다."""
        feedback_id = str(uuid.uuid4())

        prompt = self.prompt_templates['correction_request'].format(
            question=question,
            answer=answer
        )

        print("\n" + "="*50)
        print("✏️ 수정 피드백 요청")
        print("="*50)
        print(prompt)
        print("="*50)

        try:
            correction_data = await self._simulate_correction_input()

            feedback = HumanFeedback(
                feedback_id=feedback_id,
                question=question,
                original_answer=answer,
                feedback_type=FeedbackType.CORRECTION,
                feedback_source=FeedbackSource.USER_INTERFACE,
                feedback_data=correction_data,
                context=context,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id
            )

            # 세션에 추가
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].feedbacks.append(feedback)

            # 콜백 실행
            if FeedbackType.CORRECTION in self.feedback_callbacks:
                await self.feedback_callbacks[FeedbackType.CORRECTION](feedback)

            logger.info(f"수정 피드백 수집됨: {feedback_id}")
            return feedback

        except Exception as e:
            logger.error(f"수정 피드백 수집 실패: {e}")
            return self._create_error_feedback(feedback_id, question, answer, FeedbackType.CORRECTION, str(e))

    async def collect_preference_feedback(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **context
    ) -> HumanFeedback:
        """선호도 피드백을 수집합니다."""
        feedback_id = str(uuid.uuid4())

        prompt = self.prompt_templates['preference_request'].format(
            question=question,
            answer_a=answer_a,
            answer_b=answer_b
        )

        print("\n" + "="*50)
        print("🔄 선호도 피드백 요청")
        print("="*50)
        print(prompt)
        print("="*50)

        try:
            preference_data = await self._simulate_preference_input()

            feedback = HumanFeedback(
                feedback_id=feedback_id,
                question=question,
                original_answer=answer_a if preference_data.get('choice') == 'A' else answer_b,
                feedback_type=FeedbackType.PREFERENCE,
                feedback_source=FeedbackSource.USER_INTERFACE,
                feedback_data=preference_data,
                context={**context, 'answer_a': answer_a, 'answer_b': answer_b},
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id
            )

            # 세션에 추가
            if session_id and session_id in self.active_sessions:
                self.active_sessions[session_id].feedbacks.append(feedback)

            # 콜백 실행
            if FeedbackType.PREFERENCE in self.feedback_callbacks:
                await self.feedback_callbacks[FeedbackType.PREFERENCE](feedback)

            logger.info(f"선호도 피드백 수집됨: {preference_data.get('choice', 'N/A')}")
            return feedback

        except Exception as e:
            logger.error(f"선호도 피드백 수집 실패: {e}")
            return self._create_error_feedback(feedback_id, question, answer_a, FeedbackType.PREFERENCE, str(e))

    async def _simulate_rating_input(self, timeout: int) -> Dict[str, Any]:
        """평점 입력 시뮬레이션 (실제 구현에서는 실제 사용자 입력)"""
        # 시뮬레이션: 3-5 사이의 랜덤 평점
        import random
        await asyncio.sleep(1)  # 입력 대기 시뮬레이션

        rating = random.randint(3, 5)
        reasons = {
            3: "보통 수준의 답변입니다.",
            4: "좋은 답변이지만 개선의 여지가 있습니다.",
            5: "매우 만족스러운 답변입니다."
        }

        return {
            'rating': rating,
            'reason': reasons.get(rating, "평가 완료"),
            'scale': self.interface_config['rating_scale'],
            'simulated': True
        }

    async def _simulate_correction_input(self) -> Dict[str, Any]:
        """수정 입력 시뮬레이션"""
        await asyncio.sleep(1)

        # 시뮬레이션 수정 데이터
        corrections = [
            {
                'section': '정책 설명 부분',
                'original': '기존 내용',
                'corrected': '수정된 내용',
                'reason': '더 정확한 정보로 수정'
            },
            {
                'section': '절차 안내',
                'original': '기존 절차',
                'corrected': '업데이트된 절차',
                'reason': '최신 정책 반영'
            }
        ]

        import random
        selected_correction = random.choice(corrections)

        return {
            'corrections': [selected_correction],
            'overall_feedback': '전반적으로 좋으나 일부 수정이 필요합니다.',
            'simulated': True
        }

    async def _simulate_preference_input(self) -> Dict[str, Any]:
        """선호도 입력 시뮬레이션"""
        await asyncio.sleep(1)

        import random
        choice = random.choice(['A', 'B'])
        reasons = {
            'A': '첫 번째 응답이 더 상세하고 명확합니다.',
            'B': '두 번째 응답이 더 간결하고 실용적입니다.'
        }

        return {
            'choice': choice,
            'reason': reasons[choice],
            'confidence': random.uniform(0.6, 1.0),
            'simulated': True
        }

    def _create_timeout_feedback(
        self,
        feedback_id: str,
        question: str,
        answer: str,
        feedback_type: FeedbackType
    ) -> HumanFeedback:
        """시간초과 피드백을 생성합니다."""
        return HumanFeedback(
            feedback_id=feedback_id,
            question=question,
            original_answer=answer,
            feedback_type=feedback_type,
            feedback_source=FeedbackSource.USER_INTERFACE,
            feedback_data={'status': 'timeout', 'reason': '사용자 응답 시간 초과'},
            context={'timeout': True},
            timestamp=datetime.now()
        )

    def _create_error_feedback(
        self,
        feedback_id: str,
        question: str,
        answer: str,
        feedback_type: FeedbackType,
        error_message: str
    ) -> HumanFeedback:
        """오류 피드백을 생성합니다."""
        return HumanFeedback(
            feedback_id=feedback_id,
            question=question,
            original_answer=answer,
            feedback_type=feedback_type,
            feedback_source=FeedbackSource.USER_INTERFACE,
            feedback_data={'status': 'error', 'error': error_message},
            context={'error': True},
            timestamp=datetime.now()
        )

    def analyze_feedback_trends(
        self,
        time_window: Optional[timedelta] = None
    ) -> FeedbackAnalysis:
        """피드백 트렌드를 분석합니다."""
        if time_window:
            cutoff_time = datetime.now() - time_window
            feedbacks = [f for f in self.feedback_storage if f.timestamp >= cutoff_time]
        else:
            feedbacks = self.feedback_storage

        if not feedbacks:
            return FeedbackAnalysis(
                total_feedbacks=0,
                feedback_distribution={},
                average_rating=None,
                common_issues=[],
                improvement_areas=[],
                positive_aspects=[],
                trend_analysis={},
                recommendations=[]
            )

        # 피드백 분포
        feedback_distribution = {}
        for feedback in feedbacks:
            feedback_type = feedback.feedback_type
            feedback_distribution[feedback_type] = feedback_distribution.get(feedback_type, 0) + 1

        # 평균 평점 계산
        ratings = []
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING:
                rating = feedback.feedback_data.get('rating')
                if rating is not None:
                    ratings.append(rating)

        average_rating = sum(ratings) / len(ratings) if ratings else None

        # 공통 이슈 및 개선 영역
        common_issues = self._extract_common_issues(feedbacks)
        improvement_areas = self._extract_improvement_areas(feedbacks)
        positive_aspects = self._extract_positive_aspects(feedbacks)

        # 트렌드 분석
        trend_analysis = self._analyze_trends(feedbacks)

        # 개선 권고사항
        recommendations = self._generate_recommendations(
            feedbacks, average_rating, common_issues, improvement_areas
        )

        return FeedbackAnalysis(
            total_feedbacks=len(feedbacks),
            feedback_distribution=feedback_distribution,
            average_rating=average_rating,
            common_issues=common_issues,
            improvement_areas=improvement_areas,
            positive_aspects=positive_aspects,
            trend_analysis=trend_analysis,
            recommendations=recommendations
        )

    def _extract_common_issues(self, feedbacks: List[HumanFeedback]) -> List[str]:
        """공통 이슈를 추출합니다."""
        issues = []

        # 낮은 평점의 이유 수집
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING:
                rating = feedback.feedback_data.get('rating', 0)
                if rating <= 3:
                    reason = feedback.feedback_data.get('reason', '')
                    if reason:
                        issues.append(reason)

            # 수정 피드백에서 이슈 추출
            elif feedback.feedback_type == FeedbackType.CORRECTION:
                corrections = feedback.feedback_data.get('corrections', [])
                for correction in corrections:
                    if 'reason' in correction:
                        issues.append(correction['reason'])

        # 중복 제거 및 빈도 기반 정렬 (간단한 구현)
        unique_issues = list(set(issues))
        return unique_issues[:5]  # 상위 5개

    def _extract_improvement_areas(self, feedbacks: List[HumanFeedback]) -> List[str]:
        """개선 영역을 추출합니다."""
        areas = []

        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.CORRECTION:
                corrections = feedback.feedback_data.get('corrections', [])
                for correction in corrections:
                    section = correction.get('section', '')
                    if section:
                        areas.append(section)

        # 빈도 기반 정렬
        unique_areas = list(set(areas))
        return unique_areas[:5]

    def _extract_positive_aspects(self, feedbacks: List[HumanFeedback]) -> List[str]:
        """긍정적 측면을 추출합니다."""
        aspects = []

        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING:
                rating = feedback.feedback_data.get('rating', 0)
                if rating >= 4:
                    reason = feedback.feedback_data.get('reason', '')
                    if reason:
                        aspects.append(reason)

        unique_aspects = list(set(aspects))
        return unique_aspects[:5]

    def _analyze_trends(self, feedbacks: List[HumanFeedback]) -> Dict[str, Any]:
        """트렌드를 분석합니다."""
        # 간단한 트렌드 분석
        if len(feedbacks) < 2:
            return {}

        # 시간별 평점 변화
        rating_timeline = []
        for feedback in feedbacks:
            if feedback.feedback_type == FeedbackType.RATING:
                rating = feedback.feedback_data.get('rating')
                if rating is not None:
                    rating_timeline.append({
                        'timestamp': feedback.timestamp.isoformat(),
                        'rating': rating
                    })

        # 최근 vs 이전 성능
        if len(rating_timeline) >= 4:
            recent_ratings = [r['rating'] for r in rating_timeline[-len(rating_timeline)//2:]]
            earlier_ratings = [r['rating'] for r in rating_timeline[:len(rating_timeline)//2]]

            recent_avg = sum(recent_ratings) / len(recent_ratings)
            earlier_avg = sum(earlier_ratings) / len(earlier_ratings)

            trend = "improving" if recent_avg > earlier_avg else "declining" if recent_avg < earlier_avg else "stable"
        else:
            trend = "insufficient_data"

        return {
            'rating_timeline': rating_timeline,
            'performance_trend': trend,
            'total_feedback_sessions': len(set(f.session_id for f in feedbacks if f.session_id))
        }

    def _generate_recommendations(
        self,
        feedbacks: List[HumanFeedback],
        average_rating: Optional[float],
        common_issues: List[str],
        improvement_areas: List[str]
    ) -> List[str]:
        """개선 권고사항을 생성합니다."""
        recommendations = []

        # 평점 기반 권고
        if average_rating is not None:
            if average_rating < 3.0:
                recommendations.append("전반적인 응답 품질 개선이 시급히 필요합니다.")
            elif average_rating < 4.0:
                recommendations.append("응답 품질을 더욱 향상시킬 여지가 있습니다.")

        # 이슈 기반 권고
        if common_issues:
            recommendations.append(f"가장 빈번한 이슈 해결: {', '.join(common_issues[:2])}")

        # 개선 영역 기반 권고
        if improvement_areas:
            recommendations.append(f"우선 개선 영역: {', '.join(improvement_areas[:2])}")

        # 피드백 수집 관련 권고
        if len(feedbacks) < 10:
            recommendations.append("더 많은 사용자 피드백 수집이 필요합니다.")

        return recommendations[:5]

    async def export_feedback_data(
        self,
        format: str = 'json',
        time_window: Optional[timedelta] = None
    ) -> str:
        """피드백 데이터를 내보냅니다."""
        if time_window:
            cutoff_time = datetime.now() - time_window
            feedbacks = [f for f in self.feedback_storage if f.timestamp >= cutoff_time]
        else:
            feedbacks = self.feedback_storage

        if format.lower() == 'json':
            return json.dumps([asdict(f) for f in feedbacks], default=str, indent=2)
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """세션 요약을 반환합니다."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # 종료된 세션에서 검색
            session_feedbacks = [f for f in self.feedback_storage if f.session_id == session_id]
            if not session_feedbacks:
                return {}

            return {
                'session_id': session_id,
                'total_feedbacks': len(session_feedbacks),
                'feedback_types': list(set(f.feedback_type.value for f in session_feedbacks)),
                'time_span': {
                    'start': min(f.timestamp for f in session_feedbacks).isoformat(),
                    'end': max(f.timestamp for f in session_feedbacks).isoformat()
                }
            }

        return {
            'session_id': session.session_id,
            'start_time': session.start_time.isoformat(),
            'duration': str(datetime.now() - session.start_time),
            'total_feedbacks': len(session.feedbacks),
            'feedback_types': list(set(f.feedback_type.value for f in session.feedbacks)),
            'context': session.interaction_context.value,
            'metadata': session.metadata
        }


# 편의 함수들
def create_hitl_interface(
    config: Optional[CoolStayConfig] = None,
    feedback_callbacks: Optional[Dict[FeedbackType, Callable]] = None
) -> HITLInterface:
    """HITL 인터페이스를 생성합니다."""
    return HITLInterface(config, feedback_callbacks)


async def collect_simple_rating(
    question: str,
    answer: str,
    config: Optional[CoolStayConfig] = None
) -> HumanFeedback:
    """간단한 평점 수집 함수"""
    hitl = HITLInterface(config)
    return await hitl.collect_rating_feedback(question, answer)


def create_feedback_dashboard(feedbacks: List[HumanFeedback]) -> Dict[str, Any]:
    """피드백 대시보드 데이터를 생성합니다."""
    if not feedbacks:
        return {'message': '피드백 데이터가 없습니다.'}

    # 기본 통계
    total_feedbacks = len(feedbacks)
    feedback_types = [f.feedback_type.value for f in feedbacks]
    type_counts = {t: feedback_types.count(t) for t in set(feedback_types)}

    # 평점 통계
    ratings = []
    for f in feedbacks:
        if f.feedback_type == FeedbackType.RATING:
            rating = f.feedback_data.get('rating')
            if rating is not None:
                ratings.append(rating)

    rating_stats = {
        'count': len(ratings),
        'average': sum(ratings) / len(ratings) if ratings else None,
        'distribution': {i: ratings.count(i) for i in range(1, 6)} if ratings else {}
    }

    # 최근 활동
    recent_activity = sorted(feedbacks, key=lambda f: f.timestamp, reverse=True)[:5]

    return {
        'total_feedbacks': total_feedbacks,
        'feedback_type_distribution': type_counts,
        'rating_statistics': rating_stats,
        'recent_activity': [
            {
                'id': f.feedback_id,
                'type': f.feedback_type.value,
                'timestamp': f.timestamp.isoformat(),
                'preview': str(f.feedback_data)[:50] + '...'
            }
            for f in recent_activity
        ],
        'generated_at': datetime.now().isoformat()
    }