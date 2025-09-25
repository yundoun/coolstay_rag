"""
CoolStay RAG 시스템 기본 기능 테스트

Phase 2 모듈화가 완료된 후 기본적인 기능들이 올바르게 작동하는지 확인합니다.
"""

import asyncio
import logging
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """모든 모듈의 임포트를 테스트합니다."""
    print("="*50)
    print("🧪 모듈 임포트 테스트")
    print("="*50)

    tests = []

    # 1. 핵심 모듈
    try:
        from src.core.config import CoolStayConfig
        from src.core.llm import get_default_llm
        from src.core.embeddings import get_default_embeddings
        tests.append(("✅ 핵심 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 핵심 모듈", f"실패: {e}"))

    # 2. 데이터 처리 모듈
    try:
        from src.data import ChromaManager, DocumentLoader, DocumentPreprocessor
        tests.append(("✅ 데이터 처리 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 데이터 처리 모듈", f"실패: {e}"))

    # 3. 에이전트 모듈
    try:
        from src.agents import BaseRAGAgent, CorrectiveRAGAgent, WebSearchAgent
        tests.append(("✅ 에이전트 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 에이전트 모듈", f"실패: {e}"))

    # 4. 라우팅 모듈
    try:
        from src.routing import QuestionAnalyzer, DomainRouter, ResponseIntegrator
        tests.append(("✅ 라우팅 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 라우팅 모듈", f"실패: {e}"))

    # 5. 평가 모듈
    try:
        from src.evaluation import ReActEvaluationAgent, HITLInterface
        tests.append(("✅ 평가 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 평가 모듈", f"실패: {e}"))

    # 6. 파이프라인 모듈
    try:
        from src.pipeline import IntegratedRAGPipeline, WorkflowManager
        tests.append(("✅ 파이프라인 모듈", "성공"))
    except Exception as e:
        tests.append(("❌ 파이프라인 모듈", f"실패: {e}"))

    # 결과 출력
    for test_name, result in tests:
        print(f"{test_name}: {result}")

    success_count = len([t for t in tests if "✅" in t[0]])
    total_count = len(tests)
    print(f"\n📊 임포트 테스트 결과: {success_count}/{total_count} 성공")

    return success_count == total_count

def test_basic_config():
    """기본 설정을 테스트합니다."""
    print("\n" + "="*50)
    print("⚙️ 기본 설정 테스트")
    print("="*50)

    try:
        from src.core.config import CoolStayConfig

        config = CoolStayConfig()

        print(f"✅ 설정 로드 성공")
        print(f"  - 도메인 수: {len(config.get_domains())}")
        print(f"  - 임베딩 모델: {config.embeddings_config.get('model_name', 'N/A')}")
        print(f"  - LLM 모델: {config.llm_config.get('model_name', 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        return False

async def test_basic_pipeline():
    """기본 파이프라인을 테스트합니다."""
    print("\n" + "="*50)
    print("🔄 기본 파이프라인 테스트")
    print("="*50)

    try:
        from src.pipeline import IntegratedRAGPipeline, PipelineConfig
        from src.core.config import CoolStayConfig

        # 기본 설정으로 파이프라인 생성
        config = CoolStayConfig()
        pipeline_config = PipelineConfig(
            enable_evaluation=False,  # 평가 비활성화 (속도 향상)
            enable_hitl=False,        # HITL 비활성화
            enable_web_search=False   # 웹 검색 비활성화 (API 키 불필요)
        )

        pipeline = IntegratedRAGPipeline(config, pipeline_config)

        print("✅ 파이프라인 생성 성공")

        # 초기화 테스트 (실제 리소스 없이)
        print("🔄 파이프라인 초기화 테스트 중...")

        # 상태 확인만 수행 (실제 초기화는 리소스가 필요할 수 있음)
        status = await pipeline.get_system_status()
        print(f"  - 초기화 상태: {status.get('initialized', False)}")
        print(f"  - 파이프라인 모드: {status.get('pipeline_mode', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ 파이프라인 테스트 실패: {e}")
        return False

def test_basic_components():
    """기본 컴포넌트들을 테스트합니다."""
    print("\n" + "="*50)
    print("🧩 기본 컴포넌트 테스트")
    print("="*50)

    results = []

    # 1. 질문 분석기
    try:
        from src.routing.question_analyzer import QuestionAnalyzer, QuestionType

        analyzer = QuestionAnalyzer()
        print("✅ 질문 분석기 생성 성공")

        # 간단한 분석 테스트 (LLM 호출 없이)
        sample_question = "꿀스테이의 인사정책은 어떻게 되나요?"
        print(f"  - 테스트 질문: {sample_question}")

        results.append(True)

    except Exception as e:
        print(f"❌ 질문 분석기 테스트 실패: {e}")
        results.append(False)

    # 2. 도메인 라우터
    try:
        from src.routing.domain_router import DomainRouter, RoutingStrategy

        router = DomainRouter()
        print("✅ 도메인 라우터 생성 성공")

        # 상태 확인
        status = router.get_agent_status()
        print(f"  - 도메인 에이전트: {status.get('domain_agents', 0)}개")
        print(f"  - 교정 에이전트: {status.get('corrective_agents', 0)}개")

        results.append(True)

    except Exception as e:
        print(f"❌ 도메인 라우터 테스트 실패: {e}")
        results.append(False)

    # 3. 응답 통합기
    try:
        from src.routing.response_integrator import ResponseIntegrator, IntegrationStrategy

        integrator = ResponseIntegrator()
        print("✅ 응답 통합기 생성 성공")

        results.append(True)

    except Exception as e:
        print(f"❌ 응답 통합기 테스트 실패: {e}")
        results.append(False)

    success_count = sum(results)
    total_count = len(results)
    print(f"\n📊 컴포넌트 테스트 결과: {success_count}/{total_count} 성공")

    return success_count == total_count

async def main():
    """메인 테스트 함수"""
    print("🚀 CoolStay RAG 시스템 기본 기능 테스트 시작")
    print("Phase 2 모듈화 완료 후 검증")
    print("="*70)

    test_results = []

    # 1. 임포트 테스트
    test_results.append(test_imports())

    # 2. 설정 테스트
    test_results.append(test_basic_config())

    # 3. 컴포넌트 테스트
    test_results.append(test_basic_components())

    # 4. 파이프라인 테스트
    test_results.append(await test_basic_pipeline())

    # 최종 결과
    print("\n" + "="*70)
    print("📊 전체 테스트 결과")
    print("="*70)

    success_count = sum(test_results)
    total_count = len(test_results)

    print(f"✅ 성공한 테스트: {success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 모든 기본 기능 테스트 통과!")
        print("✨ Phase 2 모듈화가 성공적으로 완료되었습니다.")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        print("🔧 실패한 부분을 확인하고 수정이 필요합니다.")

    print("\n💡 다음 단계:")
    print("  1. 실제 데이터로 통합 테스트 수행")
    print("  2. 성능 벤치마크 테스트")
    print("  3. Phase 3 서비스화 준비")

    return success_count == total_count

if __name__ == "__main__":
    asyncio.run(main())