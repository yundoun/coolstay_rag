"""
CoolStay RAG 시스템 통합 테스트

Phase 2 완료 후 전체 시스템의 통합 동작을 테스트합니다.
의존성 문제가 있어도 가능한 범위에서 테스트를 수행합니다.
"""

import asyncio
import logging
import time
from pathlib import Path
import sys
from typing import Optional, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTester:
    """통합 테스트 실행기"""

    def __init__(self):
        self.test_results = []
        self.dependency_issues = []

    def test_core_modules(self) -> bool:
        """핵심 모듈 테스트"""
        print("="*60)
        print("🔧 핵심 모듈 통합 테스트")
        print("="*60)

        try:
            from src.core.config import CoolStayConfig
            config = CoolStayConfig()

            print("✅ CoolStayConfig 로드 성공")
            print(f"  - 도메인 수: {len(config.get_domains())}")
            print(f"  - 데이터 디렉토리: {config.data_dir}")
            print(f"  - 컬렉션: {list(config.collection_names.keys())}")

            return True

        except Exception as e:
            print(f"❌ 핵심 모듈 테스트 실패: {e}")
            return False

    def test_llm_module(self) -> bool:
        """LLM 모듈 테스트"""
        print("\n" + "="*60)
        print("🤖 LLM 모듈 테스트")
        print("="*60)

        try:
            from src.core.llm import get_default_llm, CoolStayLLM

            # 기본 LLM 가져오기 테스트
            llm = get_default_llm()
            print("✅ 기본 LLM 인스턴스 생성 성공")
            print(f"  - 모델: {llm.model_name}")

            # 간단한 테스트 쿼리
            test_question = "안녕하세요. 간단한 인사말로 답변해주세요."
            response = llm.invoke(test_question)

            if response and response.content:
                print("✅ LLM 응답 생성 성공")
                print(f"  - 응답 길이: {len(response.content)}자")
                print(f"  - 응답 미리보기: {response.content[:100]}...")
                return True
            else:
                print("❌ LLM 응답 생성 실패")
                return False

        except Exception as e:
            print(f"❌ LLM 모듈 테스트 실패: {e}")
            if "langchain" in str(e).lower():
                self.dependency_issues.append("langchain 관련 패키지")
            return False

    def test_routing_module(self) -> bool:
        """라우팅 모듈 테스트"""
        print("\n" + "="*60)
        print("🔄 라우팅 모듈 테스트")
        print("="*60)

        try:
            from src.routing.question_analyzer import QuestionAnalyzer
            from src.routing.response_integrator import ResponseIntegrator

            # 질문 분석기 테스트
            analyzer = QuestionAnalyzer()
            print("✅ QuestionAnalyzer 생성 성공")

            # 응답 통합기 테스트
            integrator = ResponseIntegrator()
            print("✅ ResponseIntegrator 생성 성공")

            return True

        except Exception as e:
            print(f"❌ 라우팅 모듈 테스트 실패: {e}")
            if "langchain" in str(e).lower():
                self.dependency_issues.append("langchain 관련 패키지")
            return False

    def test_evaluation_module(self) -> bool:
        """평가 모듈 테스트"""
        print("\n" + "="*60)
        print("📊 평가 모듈 테스트")
        print("="*60)

        try:
            from src.evaluation import ReActEvaluationAgent, HITLInterface

            # ReAct 평가 에이전트 테스트
            evaluator = ReActEvaluationAgent()
            print("✅ ReActEvaluationAgent 생성 성공")

            # HITL 인터페이스 테스트
            hitl = HITLInterface()
            print("✅ HITLInterface 생성 성공")

            return True

        except Exception as e:
            print(f"❌ 평가 모듈 테스트 실패: {e}")
            if "langchain" in str(e).lower():
                self.dependency_issues.append("langchain 관련 패키지")
            return False

    async def test_basic_pipeline(self) -> bool:
        """기본 파이프라인 테스트"""
        print("\n" + "="*60)
        print("🔄 기본 파이프라인 테스트")
        print("="*60)

        try:
            from src.pipeline import IntegratedRAGPipeline, PipelineConfig

            # 의존성 문제로 인해 실제 초기화는 건너뛰고 클래스 생성만 테스트
            pipeline_config = PipelineConfig(
                enable_evaluation=False,
                enable_hitl=False,
                enable_web_search=False
            )

            pipeline = IntegratedRAGPipeline(pipeline_config=pipeline_config)
            print("✅ IntegratedRAGPipeline 인스턴스 생성 성공")

            # 시스템 상태 확인
            status = await pipeline.get_system_status()
            print(f"✅ 시스템 상태 조회 성공")
            print(f"  - 초기화 상태: {status.get('initialized', False)}")
            print(f"  - 파이프라인 모드: {status.get('pipeline_mode', 'N/A')}")

            return True

        except Exception as e:
            print(f"❌ 파이프라인 테스트 실패: {e}")
            if "langchain" in str(e).lower():
                self.dependency_issues.append("langchain 관련 패키지")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 통합 테스트 실행"""
        print("🚀 CoolStay RAG 시스템 통합 테스트 시작")
        print("Phase 2 모듈화 완료 후 전체 시스템 검증")
        print("="*80)

        start_time = time.time()

        # 개별 테스트 실행
        test_results = []

        test_results.append(("핵심 모듈", self.test_core_modules()))
        test_results.append(("LLM 모듈", self.test_llm_module()))
        test_results.append(("라우팅 모듈", self.test_routing_module()))
        test_results.append(("평가 모듈", self.test_evaluation_module()))
        test_results.append(("파이프라인", await self.test_basic_pipeline()))

        execution_time = time.time() - start_time

        # 결과 요약
        print("\n" + "="*80)
        print("📊 통합 테스트 결과 요약")
        print("="*80)

        success_count = sum(1 for _, result in test_results if result)
        total_count = len(test_results)

        for test_name, result in test_results:
            status = "✅ 성공" if result else "❌ 실패"
            print(f"{status}: {test_name}")

        print(f"\n📈 전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"⏱️  실행 시간: {execution_time:.2f}초")

        # 의존성 문제 분석
        if self.dependency_issues:
            print(f"\n⚠️  의존성 문제 감지:")
            unique_issues = list(set(self.dependency_issues))
            for issue in unique_issues:
                print(f"   - {issue}")

            print(f"\n💡 해결 방법:")
            print(f"   1. 가상환경 생성: python3 -m venv venv")
            print(f"   2. 가상환경 활성화: source venv/bin/activate")
            print(f"   3. 의존성 설치: pip install -r requirements.txt")

        # 최종 평가
        if success_count == total_count:
            print(f"\n🎉 모든 통합 테스트 통과!")
            print(f"✨ Phase 2 모듈화가 성공적으로 완료되어 시스템이 안정적입니다.")
            overall_status = "완전 성공"
        elif success_count >= total_count * 0.7:
            print(f"\n✅ 대부분의 통합 테스트 통과!")
            print(f"🔧 일부 의존성 문제만 해결하면 완전한 시스템이 됩니다.")
            overall_status = "부분 성공"
        else:
            print(f"\n⚠️  통합 테스트에서 여러 문제 감지")
            print(f"🔍 시스템 점검이 필요합니다.")
            overall_status = "추가 작업 필요"

        print(f"\n🚀 다음 단계:")
        print(f"   1. 의존성 문제 해결 후 재테스트")
        print(f"   2. Phase 3 Streamlit 웹 앱 개발 시작")
        print(f"   3. 배포 준비 및 성능 최적화")

        return {
            "overall_status": overall_status,
            "success_rate": success_count / total_count,
            "test_results": dict(test_results),
            "dependency_issues": unique_issues if hasattr(self, 'dependency_issues') else [],
            "execution_time": execution_time,
            "recommendations": [
                "가상환경에서 requirements.txt 설치",
                "Phase 3 Streamlit 앱 개발 시작",
                "성능 벤치마크 테스트 수행"
            ]
        }

async def main():
    """메인 실행 함수"""
    tester = IntegrationTester()
    results = await tester.run_all_tests()
    return results

if __name__ == "__main__":
    results = asyncio.run(main())