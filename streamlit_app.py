"""
CoolStay RAG 시스템 Streamlit 웹 애플리케이션

Phase 3: 서비스화 - 사용자 친화적인 웹 인터페이스 제공
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Dict, Any, List

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="🍯 CoolStay RAG Assistant",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FFB800 0%, #FFA000 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitRAGApp:
    """Streamlit RAG 애플리케이션"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False

        if 'system_status' not in st.session_state:
            st.session_state.system_status = None

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'evaluation_counter' not in st.session_state:
            st.session_state.evaluation_counter = 0

    def check_system_dependencies(self) -> Dict[str, Any]:
        """시스템 의존성을 확인합니다"""
        status = {
            "overall_status": "unknown",
            "modules": {},
            "dependencies": {},
            "recommendations": [],
            "vectordb_init": None
        }

        # 핵심 모듈 확인
        try:
            from src.core.config import CoolStayConfig
            config = CoolStayConfig()
            status["modules"]["core"] = {"status": "✅", "message": "정상"}
            status["config"] = {
                "domains": len(config.get_domains()),
                "data_dir": str(config.data_dir)
            }

            # 벡터 DB 자동 초기화 (Streamlit Cloud에서)
            try:
                from src.utils.cloud_init import initialize_vectordb_if_needed, is_cloud_environment

                if is_cloud_environment():
                    st.info("☁️ Streamlit Cloud 환경 감지: 벡터 DB 초기화 중...")
                    init_result = initialize_vectordb_if_needed(config)
                    status["vectordb_init"] = init_result

                    if init_result["initialized"]:
                        st.success(f"✅ {init_result['message']}")
                    elif init_result["errors"]:
                        st.warning(f"⚠️ 일부 초기화 실패: {len(init_result['errors'])}개 오류")
            except Exception as e:
                st.warning(f"⚠️ 벡터 DB 자동 초기화 실패: {str(e)}")

        except Exception as e:
            status["modules"]["core"] = {"status": "❌", "message": f"오류: {str(e)}"}

        # LLM 모듈 확인
        try:
            from src.core.llm import get_default_llm
            llm = get_default_llm()
            if llm.is_initialized:
                status["modules"]["llm"] = {"status": "✅", "message": f"모델: {llm.model_name}"}
            else:
                status["modules"]["llm"] = {"status": "⚠️", "message": "초기화 대기 중"}
        except Exception as e:
            status["modules"]["llm"] = {"status": "❌", "message": f"오류: {str(e)}"}
            if "langchain" in str(e).lower():
                status["dependencies"]["langchain"] = "❌ 설치 필요"

        # 평가 모듈 확인
        try:
            from src.evaluation import ReActEvaluationAgent
            evaluator = ReActEvaluationAgent()
            status["modules"]["evaluation"] = {"status": "✅", "message": "ReAct 평가 시스템"}
        except Exception as e:
            status["modules"]["evaluation"] = {"status": "❌", "message": f"오류: {str(e)}"}

        # 전체 상태 결정
        module_statuses = [m["status"] for m in status["modules"].values()]
        if all("✅" in s for s in module_statuses):
            status["overall_status"] = "완전 정상"
        elif any("✅" in s for s in module_statuses):
            status["overall_status"] = "부분 정상"
        else:
            status["overall_status"] = "오류"

        # 권장사항 생성
        if "langchain" in status.get("dependencies", {}):
            status["recommendations"].append("pip install langchain-community langchain-ollama")

        if status["overall_status"] != "완전 정상":
            status["recommendations"].append("가상환경에서 requirements.txt 설치")

        return status

    def display_header(self):
        """헤더 표시"""
        st.markdown("""
        <div class="main-header">
            <h1>🍯 CoolStay RAG Assistant</h1>
            <p>AI 기반 다중 에이전트 질문 응답 시스템</p>
        </div>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        """사이드바 표시"""
        with st.sidebar:
            st.markdown("## 🛠️ 시스템 상태")

            # 시스템 상태 확인 버튼
            if st.button("🔄 상태 확인", type="primary"):
                with st.spinner("시스템 상태 확인 중..."):
                    st.session_state.system_status = self.check_system_dependencies()

            # 시스템 상태 표시
            if st.session_state.system_status:
                status = st.session_state.system_status

                # 전체 상태
                st.markdown(f"**전체 상태**: {status['overall_status']}")

                # 모듈별 상태
                st.markdown("### 📦 모듈 상태")
                for module_name, module_status in status["modules"].items():
                    st.markdown(f"- **{module_name}**: {module_status['status']} {module_status['message']}")

                # 설정 정보
                if "config" in status:
                    st.markdown("### ⚙️ 설정 정보")
                    st.markdown(f"- 도메인 수: {status['config']['domains']}개")

                # 권장사항
                if status.get("recommendations"):
                    st.markdown("### 💡 권장사항")
                    for rec in status["recommendations"]:
                        st.markdown(f"- {rec}")

            st.markdown("---")

            # 시스템 정보
            st.markdown("## 📋 시스템 정보")
            st.markdown(f"""
            - **프로젝트**: CoolStay Multi-Agent RAG
            - **버전**: Phase 3 (서비스화)
            - **개발 상태**: 프로토타입
            - **지원 도메인**: 7개
            """)

            st.markdown("---")

            # 사용 가이드
            st.markdown("## 📚 사용 가이드")
            with st.expander("시작하기"):
                st.markdown("""
                1. **시스템 상태 확인**: 상태 확인 버튼 클릭
                2. **질문 입력**: 메인 화면에서 질문 입력
                3. **응답 확인**: AI가 생성한 응답 검토
                4. **피드백 제공**: 응답 품질 평가
                """)

            with st.expander("지원 도메인"):
                st.markdown("""
                - **HR 정책**: 인사 관련 정책 및 절차
                - **기술 정책**: 기술 표준 및 가이드라인
                - **아키텍처**: 시스템 아키텍처 설계
                - **컴포넌트**: 소프트웨어 컴포넌트
                - **배포**: 배포 및 운영 절차
                - **개발**: 개발 프로세스 및 방법론
                - **비즈니스 정책**: 비즈니스 규정 및 프로세스
                """)

    def display_chat_interface(self):
        """채팅 인터페이스 표시"""
        st.markdown("## 💬 AI 어시스턴트와 채팅")

        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # 메타데이터가 있는 경우 표시
                if message.get("metadata"):
                    with st.expander("응답 세부 정보"):
                        st.json(message["metadata"])

        # 질문 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            print(f"DEBUG: 질문 입력 받음 - {prompt}")

            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("응답 생성 중..."):
                    print(f"DEBUG: process_question 호출 시작 - 질문: {prompt}")
                    response = self.process_question(prompt)
                    print(f"DEBUG: process_question 완료 - 응답 타입: {type(response)}")

                st.markdown(response["content"])

                # 응답을 세션에 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "metadata": response.get("metadata", {})
                })

                # 응답 평가 UI
                if response.get("success", True):
                    st.session_state.evaluation_counter += 1
                    self.display_response_evaluation(prompt, response["content"], st.session_state.evaluation_counter)

    def process_question(self, question: str) -> Dict[str, Any]:
        """질문을 처리합니다"""
        print(f"DEBUG: process_question 메소드 시작 - 질문: {question}")
        try:
            print("DEBUG: 시스템 상태 확인 시작")
            # 시스템 상태 확인
            if not st.session_state.system_status:
                print("DEBUG: system_status가 None이므로 check_system_dependencies 호출")
                st.session_state.system_status = self.check_system_dependencies()
            else:
                print("DEBUG: 기존 system_status 사용")

            status = st.session_state.system_status
            print(f"DEBUG: 시스템 상태 - overall_status: {status['overall_status']}")

            if status["overall_status"] == "완전 정상":
                print("DEBUG: 완전 정상 - RAG 파이프라인 사용")
                # 실제 RAG 파이프라인 사용
                return self._process_with_rag_pipeline(question)
            elif status["overall_status"] == "부분 정상":
                print("DEBUG: 부분 정상 - 제한적 기능 사용")
                # 제한적 기능으로 처리
                return self._process_with_limited_functionality(question, status)
            else:
                print(f"DEBUG: 오류 상태 ({status['overall_status']}) - 오류 진단 응답")
                # 오류 진단 및 안내
                return self._generate_error_diagnosis_response(question, status)

        except Exception as e:
            return {
                "content": f"⚠️ 질문 처리 중 예상치 못한 오류가 발생했습니다: {str(e)}",
                "success": False,
                "metadata": {"error": str(e), "error_type": "unexpected"}
            }

    def _process_with_rag_pipeline(self, question: str) -> Dict[str, Any]:
        """완전한 RAG 파이프라인으로 처리"""
        try:
            # 실제 IntegratedRAGPipeline 사용
            import asyncio
            from src.pipeline.rag_pipeline import IntegratedRAGPipeline, PipelineConfig
            from src.core.config import CoolStayConfig

            # RAG 파이프라인 초기화 (캐싱)
            if not hasattr(self, '_rag_pipeline') or self._rag_pipeline is None:
                config = CoolStayConfig()
                pipeline_config = PipelineConfig(
                    enable_evaluation=False,  # 웹앱에서는 빠른 응답을 위해 평가 비활성화
                    enable_hitl=False,       # HITL도 비활성화
                    enable_web_search=True,
                    enable_corrective_rag=True,
                    enable_quality_checks=True,  # 품질 검증 활성화
                    min_confidence_threshold=0.9,  # 높은 신뢰도 요구 (강제로 재구성 유발)
                    min_quality_threshold=0.95,   # 높은 품질 요구 (강제로 재구성 유발)
                    max_concurrent_agents=3   # 동시 실행 에이전트 수 제한
                )
                self._rag_pipeline = IntegratedRAGPipeline(config, pipeline_config)

            # 비동기 처리를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # 파이프라인 초기화
                if not self._rag_pipeline.is_initialized:
                    print("DEBUG: 파이프라인 초기화 시작")
                    print(f"DEBUG: config 타입: {type(config)}")
                    initialization_success = loop.run_until_complete(
                        self._rag_pipeline.initialize()
                    )
                    print(f"DEBUG: 초기화 결과: {initialization_success}")
                    print(f"DEBUG: 초기화 오류: {self._rag_pipeline.initialization_error}")
                    if not initialization_success:
                        import traceback
                        traceback.print_exc()
                        raise Exception(f"파이프라인 초기화 실패: {self._rag_pipeline.initialization_error}")

                # 질문 처리
                print(f"DEBUG: 질문 처리 시작: {question}")
                result = loop.run_until_complete(
                    self._rag_pipeline.process_question(question)
                )
                print(f"DEBUG: 질문 처리 완료")
                print(f"DEBUG: result type: {type(result)}")
                # 결과 상세 분석 로그
                print(f"""
╔══════════════════════════════════════════════════════════════╗
║ [결과 분석] Pipeline Result 상세 정보                         ║
╚══════════════════════════════════════════════════════════════╝
📋 기본 정보:
   - 질문: {result.question}
   - 성공 여부: {'✅ 성공' if result.success else '❌ 실패'}
   - 신뢰도: {result.confidence:.2f}
   - 처리 시간: {result.execution_time:.2f}초
   - 파이프라인 모드: {result.pipeline_mode.value if hasattr(result, 'pipeline_mode') else 'N/A'}

🎯 라우팅 정보:
   - 전략: {result.routing_result.routing_decision.strategy.value if hasattr(result, 'routing_result') and result.routing_result else 'N/A'}
   - 활성 에이전트: {list(result.routing_result.agent_responses.keys()) if hasattr(result, 'routing_result') and result.routing_result else 'N/A'}
   - 웹 검색 수행: {'✅' if hasattr(result, 'routing_result') and result.routing_result and result.routing_result.web_response else '❌'}

🔄 응답 통합:
   - 통합 전략: {result.integrated_response.integration_strategy.value if hasattr(result, 'integrated_response') and result.integrated_response else 'N/A'}
   - 기여 에이전트: {result.integrated_response.contributing_agents if hasattr(result, 'integrated_response') and result.integrated_response else 'N/A'}
   - 품질 점수: {result.integrated_response.quality_metrics if hasattr(result, 'integrated_response') and result.integrated_response else 'N/A'}

📝 최종 답변 (처음 200자):
   {result.final_answer[:200]}{'...' if len(result.final_answer) > 200 else ''}

🏁 완료된 단계: {[stage.value for stage in result.stages_completed] if hasattr(result, 'stages_completed') else 'N/A'}
""")

                # 각 에이전트별 상세 응답 (옵션)
                if hasattr(result, 'routing_result') and result.routing_result and result.routing_result.agent_responses:
                    print("🤖 에이전트별 응답 요약:")
                    for agent_name, agent_response in result.routing_result.agent_responses.items():
                        status_icon = "✅" if agent_response.status.value == "ready" else "❌"
                        confidence = f"{agent_response.confidence_score:.2f}" if agent_response.confidence_score else "N/A"
                        answer_preview = agent_response.answer[:100].replace('\n', ' ') + "..." if len(agent_response.answer) > 100 else agent_response.answer.replace('\n', ' ')
                        print(f"   - {agent_name}: {status_icon} (신뢰도: {confidence}) {answer_preview}")

                if hasattr(result, 'routing_result') and result.routing_result and result.routing_result.web_response:
                    web_response = result.routing_result.web_response
                    web_status = "✅" if web_response.status.value == "ready" else "❌"
                    web_preview = web_response.answer[:100].replace('\n', ' ') + "..." if len(web_response.answer) > 100 else web_response.answer.replace('\n', ' ')
                    print(f"   - 웹검색: {web_status} {web_preview}")

                print("═" * 66)

                # 결과 포맷팅
                return {
                    "content": f"✨ **CoolStay RAG 시스템 응답**\n\n**질문**: {question}\n\n**답변**: {result.final_answer}\n\n**신뢰도**: {result.confidence:.1%}\n**처리 시간**: {result.execution_time:.2f}초",
                    "success": result.success,
                    "metadata": {
                        "pipeline_mode": result.pipeline_mode.value,
                        "processing_time": result.execution_time,
                        "confidence": result.confidence,
                        "stages_completed": [stage.value for stage in result.stages_completed],
                        "routing_info": {
                            "selected_agents": len(result.routing_result.agent_responses) if result.routing_result else 0,
                            "domains_used": list(result.routing_result.agent_responses.keys()) if result.routing_result else []
                        }
                    }
                }

            finally:
                loop.close()

        except Exception as e:
            # 오류 발생 시 상세 진단 제공
            error_message = str(e)

            # 구체적인 오류 유형에 따른 처리
            if "API" in error_message or "key" in error_message.lower():
                return {
                    "content": f"🔑 **API 키 설정 오류**\n\n질문: '{question}'\n\n❌ API 키가 설정되지 않았거나 유효하지 않습니다.\n\n**해결 방법**:\n1. `.env` 파일에서 `OPENAI_API_KEY` 확인\n2. 유효한 OpenAI API 키인지 확인\n3. 웹앱 재시작\n\n**현재 오류**: {error_message}",
                    "success": False,
                    "metadata": {"error_type": "api_key", "error": error_message}
                }
            elif "import" in error_message.lower() or "module" in error_message.lower():
                return {
                    "content": f"📦 **모듈 임포트 오류**\n\n질문: '{question}'\n\n❌ 필요한 모듈을 불러올 수 없습니다.\n\n**해결 방법**:\n1. 가상환경이 활성화되어 있는지 확인\n2. `pip install -r requirements.txt` 재실행\n3. 웹앱 재시작\n\n**현재 오류**: {error_message}",
                    "success": False,
                    "metadata": {"error_type": "import_error", "error": error_message}
                }
            else:
                return {
                    "content": f"⚠️ **RAG 파이프라인 오류**\n\n질문: '{question}'\n\n❌ RAG 시스템 처리 중 오류가 발생했습니다.\n\n**오류 세부사항**: {error_message}\n\n**대안**: 제한적 기능을 사용해보거나, 시스템 상태를 다시 확인해주세요.",
                    "success": False,
                    "metadata": {"error_type": "pipeline_error", "error": error_message}
                }

    def _process_with_limited_functionality(self, question: str, status: Dict[str, Any]) -> Dict[str, Any]:
        """제한적 기능으로 처리"""
        try:
            from src.routing.question_analyzer import QuestionAnalyzer

            analyzer = QuestionAnalyzer()
            # 실제로는 LLM 호출이 필요하지만, 현재 상태 기반으로 안내

            working_modules = [name for name, info in status["modules"].items() if "✅" in info["status"]]
            failing_modules = [name for name, info in status["modules"].items() if "❌" in info["status"]]

            return {
                "content": f"""🔍 **시스템 상태 기반 응답**

**질문**: {question}

**현재 시스템 상태**:
- ✅ **정상 작동 모듈**: {', '.join(working_modules) if working_modules else '없음'}
- ❌ **오류 모듈**: {', '.join(failing_modules) if failing_modules else '없음'}

**제한적 기능 안내**:
- 일부 핵심 모듈이 정상 작동하여 기본적인 질문 분석은 가능합니다.
- 하지만 완전한 RAG 응답을 생성하기 위해서는 모든 의존성이 필요합니다.

**해결 방법**:
{self._generate_troubleshooting_steps(status)}

💡 **참고**: 전체 기능을 사용하려면 사이드바의 권장사항을 따라주세요.
""",
                "success": True,
                "metadata": {
                    "pipeline_mode": "limited",
                    "working_modules": working_modules,
                    "failing_modules": failing_modules,
                    "question_length": len(question)
                }
            }
        except Exception as e:
            return self._generate_error_diagnosis_response(question, status)

    def _generate_error_diagnosis_response(self, question: str, status: Dict[str, Any]) -> Dict[str, Any]:
        """오류 진단 및 안내 응답 생성"""
        # 주요 오류 원인 분석
        error_causes = []
        solutions = []

        # 모듈별 오류 분석
        failing_modules = []
        for module_name, module_info in status.get("modules", {}).items():
            if "❌" in module_info["status"]:
                failing_modules.append({
                    "name": module_name,
                    "message": module_info["message"]
                })

                # 구체적인 오류 원인 분석
                if "langchain" in module_info["message"].lower():
                    error_causes.append("LangChain 관련 패키지가 설치되지 않음")
                    solutions.append("pip install langchain-community langchain-ollama")
                elif "api" in module_info["message"].lower():
                    error_causes.append("API 키가 설정되지 않았거나 유효하지 않음")
                    solutions.append(".env 파일에서 OPENAI_API_KEY 확인")
                elif "초기화" in module_info["message"]:
                    error_causes.append(f"{module_name} 모듈 초기화 실패")

        # 의존성 문제 분석
        dependency_issues = status.get("dependencies", {})
        for dep, dep_status in dependency_issues.items():
            if "❌" in dep_status:
                error_causes.append(f"{dep} 패키지 누락")
                solutions.append(f"pip install {dep}")

        return {
            "content": f"""❌ **시스템 오류 진단**

**질문**: {question}

**⚠️ 현재 응답을 생성할 수 없는 이유**:

**주요 오류 모듈**:
{self._format_failing_modules(failing_modules)}

**감지된 문제**:
{self._format_error_causes(error_causes)}

**🔧 해결 방법**:
{self._format_solutions(solutions)}

**📋 체크리스트**:
□ 가상환경 활성화: `source venv/bin/activate`
□ 의존성 설치: `pip install -r requirements.txt`
□ API 키 설정: `.env` 파일에 `OPENAI_API_KEY=your-key` 추가
□ 시스템 재시작: 웹앱 재시작 후 "🔄 상태 확인" 클릭

**💡 도움말**:
- 사이드바에서 "🔄 상태 확인"을 클릭하여 실시간 시스템 상태를 확인하세요
- `INSTALLATION_GUIDE.md` 파일을 참조하여 단계별 설치를 진행하세요
- 모든 모듈이 ✅ 상태가 되면 정상적인 AI 응답을 받을 수 있습니다

🤖 **CoolStay RAG 시스템 정보**:
이 시스템이 정상 작동하면 7개 전문 도메인의 지식을 바탕으로 정확한 답변을 제공합니다.
""",
            "success": False,
            "metadata": {
                "pipeline_mode": "error_diagnosis",
                "failing_modules": [m["name"] for m in failing_modules],
                "error_causes": error_causes,
                "solutions": solutions,
                "question_length": len(question)
            }
        }

    def _format_failing_modules(self, failing_modules: List[Dict[str, str]]) -> str:
        """실패한 모듈 포맷팅"""
        if not failing_modules:
            return "- 모든 모듈이 정상 작동 중"

        formatted = []
        for module in failing_modules:
            formatted.append(f"- **{module['name']}**: {module['message']}")
        return "\n".join(formatted)

    def _format_error_causes(self, causes: List[str]) -> str:
        """오류 원인 포맷팅"""
        if not causes:
            return "- 구체적인 오류 원인을 특정할 수 없음"

        formatted = []
        for i, cause in enumerate(set(causes), 1):  # 중복 제거
            formatted.append(f"{i}. {cause}")
        return "\n".join(formatted)

    def _format_solutions(self, solutions: List[str]) -> str:
        """해결방법 포맷팅"""
        if not solutions:
            return "1. INSTALLATION_GUIDE.md 참조\n2. 가상환경에서 requirements.txt 설치"

        formatted = []
        for i, solution in enumerate(set(solutions), 1):  # 중복 제거
            formatted.append(f"{i}. {solution}")
        return "\n".join(formatted)

    def _generate_troubleshooting_steps(self, status: Dict[str, Any]) -> str:
        """문제 해결 단계 생성"""
        steps = []

        # 의존성 문제가 있는 경우
        if status.get("dependencies"):
            steps.append("1. 가상환경에서 `pip install -r requirements.txt` 실행")

        # LLM 모듈 문제가 있는 경우
        llm_status = status.get("modules", {}).get("llm", {})
        if "❌" in llm_status.get("status", ""):
            steps.append("2. .env 파일에서 OPENAI_API_KEY 확인")

        # 일반적인 해결책
        steps.append("3. 웹앱 재시작 후 '🔄 상태 확인' 클릭")

        return "\n".join(steps) if steps else "사이드바의 권장사항을 확인해주세요."

    def display_response_evaluation(self, question: str, response: str, evaluation_id: int):
        """응답 평가 UI 표시"""
        st.markdown("---")
        st.markdown("### 📊 응답 평가")

        col1, col2, col3 = st.columns(3)

        with col1:
            rating = st.slider("전체 만족도", 1, 5, 3, key=f"rating_{evaluation_id}")

        with col2:
            relevance = st.selectbox("관련성", ["높음", "보통", "낮음"], index=1, key=f"relevance_{evaluation_id}")

        with col3:
            usefulness = st.selectbox("실용성", ["유용함", "보통", "유용하지 않음"], index=1, key=f"usefulness_{evaluation_id}")

        feedback = st.text_area("추가 피드백", placeholder="개선 사항이나 추가 의견을 입력해주세요...", key=f"feedback_{evaluation_id}")

        if st.button("💾 피드백 저장", key=f"save_feedback_{evaluation_id}"):
            # 피드백 저장
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response[:100] + "...",
                "rating": rating,
                "relevance": relevance,
                "usefulness": usefulness,
                "feedback": feedback
            }

            # 세션 상태에 피드백 저장
            if "feedbacks" not in st.session_state:
                st.session_state.feedbacks = []

            st.session_state.feedbacks.append(feedback_data)

            st.success("✅ 피드백이 저장되었습니다. 감사합니다!")

    def display_analytics(self):
        """분석 정보 표시"""
        st.markdown("## 📈 사용 분석")

        if hasattr(st.session_state, 'feedbacks') and st.session_state.feedbacks:
            feedbacks = st.session_state.feedbacks

            col1, col2, col3 = st.columns(3)

            with col1:
                avg_rating = sum(f["rating"] for f in feedbacks) / len(feedbacks)
                st.metric("평균 만족도", f"{avg_rating:.1f}/5")

            with col2:
                total_questions = len(st.session_state.messages) // 2  # user + assistant pairs
                st.metric("총 질문 수", total_questions)

            with col3:
                if feedbacks:
                    latest_feedback = feedbacks[-1]
                    st.metric("최근 평가", f"{latest_feedback['rating']}/5")

            # 피드백 히스토리
            st.markdown("### 최근 피드백")
            for i, feedback in enumerate(reversed(feedbacks[-5:])):  # 최근 5개
                with st.expander(f"피드백 {len(feedbacks)-i}: {feedback['question'][:50]}..."):
                    st.json(feedback)
        else:
            st.info("아직 피드백 데이터가 없습니다. 질문을 하고 평가해주세요!")

    def run(self):
        """애플리케이션 실행"""
        self.display_header()
        self.display_sidebar()

        # 메인 콘텐츠
        tab1, tab2 = st.tabs(["💬 채팅", "📈 분석"])

        with tab1:
            self.display_chat_interface()

        with tab2:
            self.display_analytics()

        # 푸터
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            🍯 CoolStay RAG Assistant | Phase 3 서비스화 데모 |
            Built with ❤️ using Streamlit
        </div>
        """, unsafe_allow_html=True)


# 애플리케이션 실행
if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()