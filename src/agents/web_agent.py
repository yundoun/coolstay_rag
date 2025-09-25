"""
CoolStay RAG 시스템 웹 검색 에이전트 모듈

이 모듈은 실시간 웹 검색을 통해 최신 정보를 제공하는 에이전트입니다.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults

from ..core.config import config, get_model_config
from ..core.llm import CoolStayLLM, get_default_llm
from .base_agent import AgentResponse, AgentStatus

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """웹 검색 전용 에이전트"""

    def __init__(self, llm: Optional[CoolStayLLM] = None, max_results: int = 5):
        """
        웹 검색 에이전트 초기화

        Args:
            llm: 사용할 LLM 인스턴스
            max_results: 최대 검색 결과 수
        """
        self.llm = llm or get_default_llm()
        self.max_results = max_results
        self.domain = "web_search"
        self.agent_name = "웹 검색 전문가"
        self.description = "실시간 웹 검색을 통한 최신 정보 제공"
        self.status = AgentStatus.INITIALIZED

        # Tavily API 키 확인 및 초기화
        self._initialize_web_search()

        # 프롬프트 설정
        self._setup_prompts()

        # 답변 생성 체인 구성
        self._build_answer_chain()

        self.status = AgentStatus.READY if self.web_search_tool else AgentStatus.ERROR

    def _initialize_web_search(self):
        """웹 검색 도구 초기화"""
        try:
            tavily_config = get_model_config("tavily")

            if not tavily_config.api_key:
                logger.warning("Tavily API 키가 설정되지 않았습니다.")
                self.web_search_tool = None
                return

            self.web_search_tool = TavilySearchResults(
                max_results=self.max_results,
                api_key=tavily_config.api_key
            )

            # 테스트 검색
            try:
                test_results = self.web_search_tool.invoke({"query": "test"})
                logger.info("✅ 웹 검색 도구 초기화 완료")
            except Exception as e:
                logger.warning(f"웹 검색 테스트 실패: {e}")
                self.web_search_tool = None

        except Exception as e:
            logger.error(f"웹 검색 도구 초기화 실패: {e}")
            self.web_search_tool = None

    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        self.web_answer_prompt = ChatPromptTemplate.from_template("""
당신은 웹 검색을 통해 최신 정보를 제공하는 전문가입니다.

**역할:**
- 웹 검색 결과를 종합하여 정확하고 유용한 정보 제공
- 최신 정보임을 명시하고 신뢰할 수 있는 출처 인용
- 여러 소스의 정보가 다를 경우 이를 명시
- 정보의 신뢰성을 고려한 답변 생성

**질문:** {question}

**웹 검색 결과:**
{search_results}

**답변 지침:**
1. 검색 결과를 종합하여 질문에 대한 정확한 답변 제공
2. 최신 정보임을 명시하고 출처 언급 (URL 포함)
3. 여러 소스의 정보가 다를 경우 이를 명시하고 각각의 관점 제시
4. 신뢰할 수 있는 정보를 우선적으로 사용
5. 불확실하거나 상충하는 정보가 있다면 명시적으로 표현
6. 답변은 한국어로 작성
7. 검색 결과가 부족하거나 관련성이 낮다면 솔직히 표현

**답변:**
""")

        self.search_query_optimizer = ChatPromptTemplate.from_template("""
다음 질문을 웹 검색에 최적화된 검색어로 변환해주세요.

**원래 질문:** {original_question}

**최적화 지침:**
1. 핵심 키워드 추출
2. 검색 효율성을 높일 수 있는 용어 사용
3. 불필요한 조사/어미 제거
4. 구체적이고 명확한 검색어로 변환
5. 여러 개념이 있다면 관련 키워드 포함

**최적화된 검색어 (한국어 또는 영어):**
""")

    def _build_answer_chain(self):
        """답변 생성 체인 구성"""
        if not self.llm.is_initialized:
            logger.warning("웹 검색 에이전트: LLM이 초기화되지 않음")
            self.answer_chain = None
            self.query_optimizer_chain = None
            return

        try:
            # 답변 생성 체인
            self.answer_chain = (
                self.web_answer_prompt
                | self.llm.llm
                | StrOutputParser()
            )

            # 검색어 최적화 체인
            self.query_optimizer_chain = (
                self.search_query_optimizer
                | self.llm.llm
                | StrOutputParser()
            )

            logger.info("✅ 웹 검색 에이전트 체인 구성 완료")

        except Exception as e:
            logger.error(f"❌ 웹 검색 에이전트 체인 구성 실패: {e}")
            self.answer_chain = None
            self.query_optimizer_chain = None

    def optimize_search_query(self, question: str) -> str:
        """검색어 최적화"""
        if not self.query_optimizer_chain:
            return question

        try:
            optimized_query = self.query_optimizer_chain.invoke({
                "original_question": question
            })
            return optimized_query.strip()
        except Exception as e:
            logger.error(f"검색어 최적화 실패: {e}")
            return question

    def search_web(self, query: str, optimize_query: bool = True) -> List[Dict[str, Any]]:
        """웹 검색 수행"""
        if not self.web_search_tool:
            logger.warning("웹 검색 도구가 초기화되지 않았습니다.")
            return []

        try:
            # 검색어 최적화
            if optimize_query:
                search_query = self.optimize_search_query(query)
                logger.info(f"검색어 최적화: '{query}' → '{search_query}'")
            else:
                search_query = query

            # 웹 검색 실행
            results = self.web_search_tool.invoke({"query": search_query})

            if isinstance(results, list):
                logger.info(f"웹 검색 완료: {len(results)}개 결과")
                return results
            else:
                logger.warning("웹 검색 결과 형식이 예상과 다릅니다.")
                return []

        except Exception as e:
            logger.error(f"웹 검색 실패: {e}")
            return []

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """검색 결과 포맷팅"""
        if not results:
            return "검색 결과가 없습니다."

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "제목 없음")
            content = result.get("content", "내용 없음")
            url = result.get("url", "URL 없음")

            # 내용 길이 제한
            if len(content) > 300:
                content = content[:300] + "..."

            formatted.append(
                f"**검색 결과 {i}:**\n"
                f"제목: {title}\n"
                f"내용: {content}\n"
                f"출처: {url}"
            )

        return "\n\n".join(formatted)

    def generate_web_answer(self, question: str, search_results: List[Dict[str, Any]]) -> str:
        """웹 검색 결과 기반 답변 생성"""
        if not self.answer_chain:
            return "웹 검색 답변 생성 기능을 사용할 수 없습니다."

        if not search_results:
            return "웹 검색에서 관련 정보를 찾을 수 없었습니다. 다른 키워드로 다시 시도해보세요."

        try:
            formatted_results = self.format_search_results(search_results)

            answer = self.answer_chain.invoke({
                "question": question,
                "search_results": formatted_results
            })

            return answer

        except Exception as e:
            logger.error(f"웹 검색 답변 생성 실패: {e}")
            # 폴백: 검색 결과를 단순 포맷팅하여 반환
            fallback_answer = f"웹 검색을 통해 다음 정보를 찾았습니다:\n\n{formatted_results}"
            return fallback_answer

    def process_query(self, question: str, **kwargs) -> AgentResponse:
        """웹 검색 기반 질문 처리"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        try:
            # 웹 검색 수행
            search_results = self.search_web(question)

            if not search_results:
                answer = "죄송합니다. 웹 검색에서 관련 정보를 찾을 수 없습니다."
                source_docs = []
            else:
                # 답변 생성
                answer = self.generate_web_answer(question, search_results)

                # Document 형태로 변환
                source_docs = []
                for result in search_results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "source": "web_search",
                            "search_query": question,
                            "domain": self.domain
                        }
                    )
                    source_docs.append(doc)

            processing_time = time.time() - start_time

            response = AgentResponse(
                answer=answer,
                source_documents=source_docs,
                domain=self.domain,
                agent_name=self.agent_name,
                processing_time=processing_time,
                status=AgentStatus.READY,
                metadata={
                    'question': question,
                    'search_results_count': len(search_results),
                    'web_search_used': True,
                    'search_sources': [r.get("url", "") for r in search_results],
                    'query_optimized': True
                }
            )

            self.status = AgentStatus.READY
            return response

        except Exception as e:
            self.status = AgentStatus.ERROR
            error_msg = str(e)
            logger.error(f"웹 검색 에이전트 처리 실패: {error_msg}")

            return AgentResponse(
                answer=f"웹 검색 중 오류가 발생했습니다: {error_msg}",
                source_documents=[],
                domain=self.domain,
                agent_name=self.agent_name,
                processing_time=time.time() - start_time,
                status=AgentStatus.ERROR,
                error=error_msg
            )

    def get_status(self) -> Dict[str, Any]:
        """웹 검색 에이전트 상태 정보"""
        return {
            'domain': self.domain,
            'agent_name': self.agent_name,
            'description': self.description,
            'status': self.status.value,
            'web_search_available': self.web_search_tool is not None,
            'llm_available': self.llm.is_initialized if self.llm else False,
            'answer_chain_ready': self.answer_chain is not None,
            'query_optimizer_ready': self.query_optimizer_chain is not None,
            'max_results': self.max_results,
            'tavily_api_configured': bool(os.getenv("TAVILY_API_KEY"))
        }

    def health_check(self) -> Dict[str, Any]:
        """웹 검색 에이전트 헬스 체크"""
        health = {
            'overall_status': 'healthy',
            'checks': {}
        }

        # 1. Tavily API 키 확인
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        health['checks']['tavily_api_key'] = {
            'status': 'pass' if tavily_api_key else 'fail',
            'message': 'Tavily API 키 설정됨' if tavily_api_key else 'Tavily API 키 없음'
        }

        # 2. 웹 검색 도구 확인
        health['checks']['web_search_tool'] = {
            'status': 'pass' if self.web_search_tool else 'fail',
            'message': '웹 검색 도구 사용 가능' if self.web_search_tool else '웹 검색 도구 초기화 실패'
        }

        # 3. LLM 확인
        health['checks']['llm'] = {
            'status': 'pass' if self.llm and self.llm.is_initialized else 'fail',
            'message': 'LLM 연결됨' if self.llm and self.llm.is_initialized else 'LLM 연결 안됨'
        }

        # 4. 답변 체인 확인
        health['checks']['answer_chain'] = {
            'status': 'pass' if self.answer_chain else 'fail',
            'message': '답변 생성 체인 구성됨' if self.answer_chain else '답변 생성 체인 구성 안됨'
        }

        # 전체 상태 결정
        failed_checks = sum(1 for check in health['checks'].values() if check['status'] == 'fail')
        if failed_checks == 0:
            health['overall_status'] = 'healthy'
        elif failed_checks < len(health['checks']):
            health['overall_status'] = 'degraded'
        else:
            health['overall_status'] = 'unhealthy'

        return health

    def is_relevant_question(self, question: str) -> bool:
        """웹 검색이 필요한 질문인지 판단"""
        # 최신 정보, 시간 관련, 뉴스 관련 키워드
        web_search_keywords = [
            "최신", "뉴스", "현재", "오늘", "2024", "업데이트",
            "동향", "트렌드", "변화", "발표", "출시", "런칭"
        ]

        question_lower = question.lower()
        matches = sum(1 for keyword in web_search_keywords if keyword in question_lower)
        return matches > 0

    def __str__(self) -> str:
        return f"WebSearchAgent(status={self.status.value})"

    def __repr__(self) -> str:
        return f"WebSearchAgent(agent_name='{self.agent_name}', status={self.status})"


# 편의 함수들
def create_web_agent(llm: Optional[CoolStayLLM] = None, max_results: int = 5) -> WebSearchAgent:
    """웹 검색 에이전트 생성 편의 함수"""
    return WebSearchAgent(llm, max_results)


def is_web_search_available() -> bool:
    """웹 검색 기능 사용 가능 여부 확인"""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    return bool(tavily_api_key)


def search_web_simple(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """간단한 웹 검색 편의 함수"""
    agent = create_web_agent(max_results=max_results)

    if agent.status == AgentStatus.ERROR:
        logger.warning("웹 검색 에이전트를 초기화할 수 없습니다.")
        return []

    return agent.search_web(query, optimize_query=True)


if __name__ == "__main__":
    # 웹 검색 에이전트 테스트
    print("🌐 CoolStay 웹 검색 에이전트 테스트")
    print("=" * 50)

    # 웹 검색 에이전트 생성
    agent = create_web_agent(max_results=3)
    status = agent.get_status()

    print(f"📊 웹 검색 에이전트 상태:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 헬스 체크
    health = agent.health_check()
    print(f"\n🏥 헬스 체크: {health['overall_status']}")

    for check_name, check_result in health['checks'].items():
        status_icon = "✅" if check_result['status'] == 'pass' else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    # 웹 검색 테스트
    if health['overall_status'] in ['healthy', 'degraded']:
        print(f"\n🔍 웹 검색 테스트:")
        test_question = "2024년 AI 기술 트렌드"
        print(f"   질문: {test_question}")

        # 검색어 최적화 테스트
        optimized_query = agent.optimize_search_query(test_question)
        print(f"   최적화된 검색어: {optimized_query}")

        # 웹 검색 실행
        search_results = agent.search_web(test_question)
        print(f"   검색 결과: {len(search_results)}개")

        if search_results:
            # 첫 번째 결과만 출력
            first_result = search_results[0]
            print(f"   첫 번째 결과:")
            print(f"     제목: {first_result.get('title', 'N/A')}")
            print(f"     출처: {first_result.get('url', 'N/A')}")

        # 질문 처리 테스트
        print(f"\n💬 전체 질문 처리 테스트:")
        response = agent.process_query(test_question)
        print(f"   답변: {response.answer[:200]}...")
        print(f"   처리 시간: {response.processing_time:.2f}초")
        print(f"   소스 문서: {len(response.source_documents)}개")

    else:
        print(f"\n❌ 웹 검색 에이전트가 정상 상태가 아니어서 테스트를 건너뜁니다.")
        print(f"   Tavily API 키를 .env 파일에 설정해주세요: TAVILY_API_KEY=your_key")