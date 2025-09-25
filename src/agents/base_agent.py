"""
CoolStay RAG 시스템 기본 에이전트 모듈

이 모듈은 RAG 에이전트의 기본 클래스와 공통 기능을 제공합니다.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..core.config import config, get_domain_config
from ..core.llm import get_default_llm, CoolStayLLM
from ..vectorstore import ChromaManager, MultiDomainRetriever

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """에이전트 상태"""
    INITIALIZED = "initialized"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentResponse:
    """에이전트 응답 결과"""
    answer: str
    source_documents: List[Document]
    domain: str
    agent_name: str
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    status: AgentStatus = AgentStatus.READY
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseRAGAgent:
    """기본 RAG 에이전트 클래스"""

    def __init__(self, domain: str, llm: Optional[CoolStayLLM] = None,
                 chroma_manager: Optional[ChromaManager] = None):
        """
        기본 RAG 에이전트 초기화

        Args:
            domain: 에이전트가 담당할 도메인
            llm: 사용할 LLM 인스턴스
            chroma_manager: ChromaDB 관리자
        """
        self.domain = domain
        self.llm = llm or get_default_llm()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.status = AgentStatus.INITIALIZED

        # 도메인 설정 로드
        try:
            self.domain_config = get_domain_config(domain)
            self.agent_name = f"{domain.title()} 전문가"
            self.description = self.domain_config.description
            self.keywords = self.domain_config.keywords
        except ValueError:
            logger.error(f"알 수 없는 도메인: {domain}")
            self.domain_config = None
            self.agent_name = f"{domain} 에이전트"
            self.description = f"{domain} 관련 질문 처리"
            self.keywords = []

        # 벡터 저장소 로드
        self.vectorstore = self.chroma_manager.get_vectorstore(domain)
        self.retriever = None

        # 프롬프트 템플릿 설정
        self._setup_prompts()

        # RAG 체인 구성
        self._build_rag_chain()

        # 초기화 완료
        self.status = AgentStatus.READY if self.vectorstore else AgentStatus.ERROR

    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""
        self.system_prompt = f"""당신은 꿀스테이의 {self.agent_name}입니다.

**전문 분야**: {self.description}
**핵심 키워드**: {', '.join(self.keywords)}

**역할과 책임**:
1. {self.description} 관련 질문에 대해 정확하고 상세한 답변 제공
2. 제공된 컨텍스트 정보를 기반으로 한 신뢰할 수 있는 답변
3. 불확실한 정보에 대한 명시적 표현
4. 실무에 도움이 되는 구체적인 가이드라인 제공

**답변 원칙**:
- 질문에 직접적으로 답변하세요
- 제공된 문서의 정보를 우선적으로 활용하세요
- 구체적인 예시나 절차가 있다면 포함하세요
- 확실하지 않은 정보는 "문서에 따르면..." 등으로 표현하세요
- 답변은 한국어로 작성하세요
- 전문적이면서도 이해하기 쉽게 설명하세요"""

        self.rag_prompt_template = ChatPromptTemplate.from_template("""
{system_prompt}

**질문**: {question}

**관련 문서**:
{context}

**답변**:
""")

    def _build_rag_chain(self):
        """RAG 체인 구성"""
        if not self.vectorstore or not self.llm.is_initialized:
            logger.warning(f"{self.domain} 에이전트: 벡터 저장소 또는 LLM이 초기화되지 않음")
            self.rag_chain = None
            return

        try:
            # 검색기 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            # 문서 포맷팅 함수
            def format_docs(docs):
                if not docs:
                    return "관련 문서를 찾을 수 없습니다."

                formatted = []
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content
                    metadata = doc.metadata

                    # 헤더 정보 추출
                    header_info = []
                    for key in ["Header 1", "Header 2", "Header 3"]:
                        if key in metadata and metadata[key]:
                            header_info.append(metadata[key])

                    header_str = " > ".join(header_info) if header_info else "섹션 정보 없음"

                    formatted.append(f"**문서 {i}** (출처: {header_str})\n{content}")

                return "\n\n".join(formatted)

            # RAG 체인 구성
            self.rag_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "system_prompt": lambda _: self.system_prompt
                }
                | self.rag_prompt_template
                | self.llm.llm
                | StrOutputParser()
            )

            logger.info(f"✅ {self.domain} 에이전트 RAG 체인 구성 완료")

        except Exception as e:
            logger.error(f"❌ {self.domain} 에이전트 RAG 체인 구성 실패: {e}")
            self.rag_chain = None

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """문서 검색"""
        if not self.vectorstore:
            logger.warning(f"{self.domain} 에이전트: 벡터 저장소가 없습니다")
            return []

        try:
            documents = self.vectorstore.similarity_search(query, k=k)
            return documents
        except Exception as e:
            logger.error(f"{self.domain} 문서 검색 실패: {e}")
            return []

    def generate_answer(self, question: str) -> str:
        """답변 생성"""
        if not self.rag_chain:
            return f"죄송합니다. {self.agent_name}이 현재 사용할 수 없습니다. 시스템 관리자에게 문의해주세요."

        try:
            answer = self.rag_chain.invoke(question)
            return answer
        except Exception as e:
            logger.error(f"{self.domain} 답변 생성 실패: {e}")
            return f"죄송합니다. {self.description} 관련 질문 처리 중 오류가 발생했습니다."

    def process_query(self, question: str, **kwargs) -> AgentResponse:
        """질문 처리 (메인 인터페이스)"""
        start_time = time.time()
        self.status = AgentStatus.BUSY

        try:
            # 문서 검색
            source_documents = self.retrieve_documents(question)

            # 답변 생성
            answer = self.generate_answer(question)

            processing_time = time.time() - start_time

            # 응답 생성
            response = AgentResponse(
                answer=answer,
                source_documents=source_documents,
                domain=self.domain,
                agent_name=self.agent_name,
                processing_time=processing_time,
                status=AgentStatus.READY,
                metadata={
                    'question': question,
                    'retrieved_docs_count': len(source_documents),
                    'domain_description': self.description,
                    'keywords_matched': self._count_keyword_matches(question)
                }
            )

            self.status = AgentStatus.READY
            return response

        except Exception as e:
            self.status = AgentStatus.ERROR
            error_msg = str(e)
            logger.error(f"{self.domain} 에이전트 처리 실패: {error_msg}")

            return AgentResponse(
                answer=f"죄송합니다. 질문 처리 중 오류가 발생했습니다: {error_msg}",
                source_documents=[],
                domain=self.domain,
                agent_name=self.agent_name,
                processing_time=time.time() - start_time,
                status=AgentStatus.ERROR,
                error=error_msg
            )

    def _count_keyword_matches(self, question: str) -> int:
        """질문에서 키워드 매칭 수 계산"""
        if not self.keywords:
            return 0

        question_lower = question.lower()
        matches = sum(1 for keyword in self.keywords if keyword.lower() in question_lower)
        return matches

    def is_relevant_question(self, question: str, threshold: int = 1) -> bool:
        """질문이 이 에이전트와 관련 있는지 판단"""
        keyword_matches = self._count_keyword_matches(question)
        return keyword_matches >= threshold

    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보 반환"""
        return {
            'domain': self.domain,
            'agent_name': self.agent_name,
            'description': self.description,
            'status': self.status.value,
            'vectorstore_available': self.vectorstore is not None,
            'llm_available': self.llm.is_initialized if self.llm else False,
            'rag_chain_ready': self.rag_chain is not None,
            'document_count': self.chroma_manager.get_domain_document_count(self.domain) if self.chroma_manager else 0,
            'keywords': self.keywords
        }

    def health_check(self) -> Dict[str, Any]:
        """에이전트 헬스 체크"""
        health = {
            'overall_status': 'healthy',
            'checks': {}
        }

        # 1. 벡터 저장소 확인
        if self.vectorstore:
            try:
                doc_count = self.vectorstore._collection.count()
                health['checks']['vectorstore'] = {
                    'status': 'pass',
                    'message': f'벡터 저장소 정상 ({doc_count}개 문서)'
                }
            except Exception as e:
                health['checks']['vectorstore'] = {
                    'status': 'fail',
                    'message': f'벡터 저장소 오류: {e}'
                }
        else:
            health['checks']['vectorstore'] = {
                'status': 'fail',
                'message': '벡터 저장소 없음'
            }

        # 2. LLM 확인
        if self.llm and self.llm.is_initialized:
            health['checks']['llm'] = {
                'status': 'pass',
                'message': 'LLM 연결됨'
            }
        else:
            health['checks']['llm'] = {
                'status': 'fail',
                'message': 'LLM 연결 안됨'
            }

        # 3. RAG 체인 확인
        health['checks']['rag_chain'] = {
            'status': 'pass' if self.rag_chain else 'fail',
            'message': 'RAG 체인 구성됨' if self.rag_chain else 'RAG 체인 구성 안됨'
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

    def __str__(self) -> str:
        return f"BaseRAGAgent(domain={self.domain}, status={self.status.value})"

    def __repr__(self) -> str:
        return f"BaseRAGAgent(domain='{self.domain}', agent_name='{self.agent_name}', status={self.status})"


# 편의 함수들
def create_agent(domain: str, llm: Optional[CoolStayLLM] = None,
                chroma_manager: Optional[ChromaManager] = None) -> BaseRAGAgent:
    """RAG 에이전트 생성 편의 함수"""
    return BaseRAGAgent(domain, llm, chroma_manager)


def create_all_domain_agents(llm: Optional[CoolStayLLM] = None,
                            chroma_manager: Optional[ChromaManager] = None) -> Dict[str, BaseRAGAgent]:
    """모든 도메인 에이전트 생성"""
    agents = {}

    for domain in config.domain_list:
        try:
            agent = BaseRAGAgent(domain, llm, chroma_manager)
            if agent.status != AgentStatus.ERROR:
                agents[domain] = agent
                logger.info(f"✅ {domain} 에이전트 생성 완료")
            else:
                logger.warning(f"⚠️ {domain} 에이전트 생성 실패")
        except Exception as e:
            logger.error(f"❌ {domain} 에이전트 생성 중 오류: {e}")

    logger.info(f"🎉 도메인 에이전트 생성 완료: {len(agents)}/{len(config.domain_list)}개")
    return agents


def get_agent_status_summary(agents: Dict[str, BaseRAGAgent]) -> Dict[str, Any]:
    """에이전트들의 상태 요약"""
    summary = {
        'total_agents': len(agents),
        'healthy_agents': 0,
        'degraded_agents': 0,
        'unhealthy_agents': 0,
        'agent_details': {}
    }

    for domain, agent in agents.items():
        health = agent.health_check()
        summary['agent_details'][domain] = {
            'status': health['overall_status'],
            'agent_name': agent.agent_name,
            'document_count': agent.chroma_manager.get_domain_document_count(domain)
        }

        if health['overall_status'] == 'healthy':
            summary['healthy_agents'] += 1
        elif health['overall_status'] == 'degraded':
            summary['degraded_agents'] += 1
        else:
            summary['unhealthy_agents'] += 1

    return summary


if __name__ == "__main__":
    # 기본 에이전트 테스트
    print("🤖 CoolStay 기본 에이전트 테스트")
    print("=" * 50)

    # 단일 에이전트 테스트
    test_domain = "hr_policy"
    print(f"🔍 {test_domain} 에이전트 생성 중...")

    agent = create_agent(test_domain)
    status = agent.get_status()

    print(f"📊 에이전트 상태:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 헬스 체크
    health = agent.health_check()
    print(f"\n🏥 헬스 체크: {health['overall_status']}")

    for check_name, check_result in health['checks'].items():
        status_icon = "✅" if check_result['status'] == 'pass' else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    # 질문 처리 테스트
    if health['overall_status'] in ['healthy', 'degraded']:
        print(f"\n💬 질문 처리 테스트:")
        test_question = "휴가는 어떻게 신청하나요?"
        print(f"   질문: {test_question}")

        response = agent.process_query(test_question)
        print(f"   답변: {response.answer[:200]}...")
        print(f"   처리 시간: {response.processing_time:.2f}초")
        print(f"   검색된 문서: {len(response.source_documents)}개")
    else:
        print(f"\n❌ 에이전트가 정상 상태가 아니어서 테스트를 건너뜁니다.")

    print(f"\n🎯 모든 도메인 에이전트 생성 테스트:")
    all_agents = create_all_domain_agents()
    summary = get_agent_status_summary(all_agents)

    print(f"   총 에이전트: {summary['total_agents']}개")
    print(f"   정상: {summary['healthy_agents']}개")
    print(f"   부분 문제: {summary['degraded_agents']}개")
    print(f"   문제 있음: {summary['unhealthy_agents']}개")