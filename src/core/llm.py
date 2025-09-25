"""
CoolStay RAG 시스템 LLM (대언어모델) 관리 모듈

이 모듈은 OpenAI의 ChatGPT 모델을 관리하고 RAG 시스템에서 사용할 수 있는
통합된 인터페이스를 제공합니다.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from .config import config, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 응답 결과"""
    content: str
    model: str
    usage_tokens: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    success: bool = True


class CoolStayLLM:
    """CoolStay RAG 시스템용 LLM 래퍼 클래스"""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        LLM 초기화

        Args:
            model_config: 모델 설정. None인 경우 기본 설정 사용
        """
        self.config = model_config or config.openai_config
        self.llm: Optional[ChatOpenAI] = None
        self.is_initialized = False
        self.initialization_error = None

        # 초기화 시도
        self.initialize()

    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        return self.config.name

    def initialize(self) -> bool:
        """LLM 초기화"""
        try:
            # API 키 검증
            if not self.config.api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

            if not self.config.api_key.startswith('sk-'):
                raise ValueError("올바르지 않은 OpenAI API 키 형식입니다.")

            # ChatOpenAI 인스턴스 생성
            self.llm = ChatOpenAI(
                model=self.config.name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key
            )

            # 연결 테스트
            test_response = self._test_connection()
            if test_response.success:
                self.is_initialized = True
                logger.info(f"✅ LLM 초기화 성공: {self.config.name}")
                return True
            else:
                raise Exception(f"연결 테스트 실패: {test_response.error}")

        except Exception as e:
            self.initialization_error = str(e)
            self.is_initialized = False
            logger.error(f"❌ LLM 초기화 실패: {e}")
            return False

    def _test_connection(self) -> LLMResponse:
        """LLM 연결 테스트"""
        try:
            start_time = time.time()
            response = self.llm.invoke([HumanMessage(content="Hello")])
            response_time = time.time() - start_time

            return LLMResponse(
                content=response.content,
                model=self.config.name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.config.name,
                error=str(e),
                success=False
            )

    def invoke(self, messages: List[Any]) -> LLMResponse:
        """메시지로 LLM 호출"""
        if not self.is_initialized:
            return LLMResponse(
                content="",
                model=self.config.name,
                error=f"LLM이 초기화되지 않았습니다: {self.initialization_error}",
                success=False
            )

        try:
            start_time = time.time()
            response = self.llm.invoke(messages)
            response_time = time.time() - start_time

            return LLMResponse(
                content=response.content,
                model=self.config.name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            logger.error(f"LLM 호출 실패: {e}")
            return LLMResponse(
                content="",
                model=self.config.name,
                error=str(e),
                success=False
            )

    def invoke_with_prompt(self, prompt_template: str, **kwargs) -> LLMResponse:
        """프롬프트 템플릿으로 LLM 호출"""
        if not self.is_initialized:
            return LLMResponse(
                content="",
                model=self.config.name,
                error=f"LLM이 초기화되지 않았습니다: {self.initialization_error}",
                success=False
            )

        try:
            # 프롬프트 포맷팅
            formatted_prompt = prompt_template.format(**kwargs)

            start_time = time.time()
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            response_time = time.time() - start_time

            return LLMResponse(
                content=response.content,
                model=self.config.name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            logger.error(f"프롬프트 기반 LLM 호출 실패: {e}")
            return LLMResponse(
                content="",
                model=self.config.name,
                error=str(e),
                success=False
            )

    def get_chain_with_parser(self, prompt_template: ChatPromptTemplate, parser_type: str = "str"):
        """체인과 파서를 함께 반환"""
        if not self.is_initialized:
            raise RuntimeError(f"LLM이 초기화되지 않았습니다: {self.initialization_error}")

        if parser_type == "json":
            parser = JsonOutputParser()
        else:
            parser = StrOutputParser()

        return prompt_template | self.llm | parser

    def create_rag_answer_chain(self, agent_name: str, description: str) -> Any:
        """RAG 답변 생성용 체인 생성"""
        if not self.is_initialized:
            raise RuntimeError(f"LLM이 초기화되지 않았습니다: {self.initialization_error}")

        prompt = ChatPromptTemplate.from_template("""
        당신은 꿀스테이의 {agent_name}입니다.
        전문 분야: {description}

        **역할:**
        - {description} 관련 질문에 대해 정확하고 상세한 답변 제공
        - 제공된 컨텍스트 정보를 기반으로 답변
        - 불확실한 정보는 명시적으로 표현

        **질문:** {question}

        **관련 문서:**
        {context}

        **답변 지침:**
        1. 질문에 직접적으로 답변하세요
        2. 제공된 문서의 정보를 우선적으로 활용하세요
        3. 구체적인 예시나 절차가 있다면 포함하세요
        4. 확실하지 않은 정보는 "문서에 따르면..." 등으로 표현하세요
        5. 답변은 한국어로 작성하세요

        **답변:**
        """)

        return prompt.partial(agent_name=agent_name, description=description) | self.llm | StrOutputParser()

    def create_quality_evaluation_chain(self) -> Any:
        """품질 평가용 체인 생성"""
        if not self.is_initialized:
            raise RuntimeError(f"LLM이 초기화되지 않았습니다: {self.initialization_error}")

        prompt = ChatPromptTemplate.from_template("""
        당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.

        **평가 기준:**
        1. **관련성**: 질문과 답변이 얼마나 관련이 있는가?
        2. **정확성**: 제공된 정보가 얼마나 정확한가?
        3. **완성도**: 질문에 대한 답변이 얼마나 완전한가?
        4. **확신도**: 답변의 신뢰도는 얼마나 높은가?

        **질문:** {question}

        **제공된 컨텍스트:**
        {context}

        **생성된 답변:**
        {answer}

        **평가 요청:**
        각 기준에 대해 0.0~1.0 점수를 매기고, 전체적인 품질 등급(excellent/good/fair/poor)을 결정하세요.
        개선이 필요한지 판단하고 그 이유를 설명하세요.

        다음 JSON 형식으로 응답하세요:
        {{
            "overall_quality": "excellent|good|fair|poor",
            "relevance_score": 0.0-1.0,
            "accuracy_score": 0.0-1.0,
            "completeness_score": 0.0-1.0,
            "confidence_score": 0.0-1.0,
            "reasoning": "평가 이유",
            "needs_improvement": true|false
        }}
        """)

        return prompt | self.llm | JsonOutputParser()

    def create_query_improvement_chain(self, description: str) -> Any:
        """쿼리 개선용 체인 생성"""
        if not self.is_initialized:
            raise RuntimeError(f"LLM이 초기화되지 않았습니다: {self.initialization_error}")

        prompt = ChatPromptTemplate.from_template("""
        다음 질문에 대한 검색 결과가 만족스럽지 않습니다.
        더 나은 검색을 위해 질문을 개선해주세요.

        **원래 질문:** {original_question}
        **검색된 컨텍스트:** {context}
        **품질 평가:** {quality_feedback}

        **요청:**
        - 더 구체적이고 명확한 검색 쿼리로 개선
        - {description} 도메인에 특화된 용어 사용
        - 여러 관점에서 접근할 수 있도록 확장

        **개선된 질문:**
        """)

        return prompt.partial(description=description) | self.llm | StrOutputParser()

    def create_web_search_answer_chain(self) -> Any:
        """웹 검색 답변 생성용 체인 생성"""
        if not self.is_initialized:
            raise RuntimeError(f"LLM이 초기화되지 않았습니다: {self.initialization_error}")

        prompt = ChatPromptTemplate.from_template("""
        당신은 웹 검색을 통해 최신 정보를 제공하는 전문가입니다.

        **질문:** {question}

        **웹 검색 결과:**
        {search_results}

        **답변 지침:**
        1. 검색 결과를 종합하여 질문에 대한 정확한 답변 제공
        2. 최신 정보임을 명시하고 출처 언급
        3. 여러 소스의 정보가 다를 경우 이를 명시
        4. 신뢰할 수 있는 정보 우선 사용
        5. 답변은 한국어로 작성

        **답변:**
        """)

        return prompt | self.llm | StrOutputParser()

    def get_status(self) -> Dict[str, Any]:
        """LLM 상태 정보 반환"""
        return {
            "initialized": self.is_initialized,
            "model": self.config.name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_key_set": bool(self.config.api_key),
            "initialization_error": self.initialization_error
        }


class LLMManager:
    """여러 LLM 인스턴스를 관리하는 매니저 클래스"""

    def __init__(self):
        self.llm_instances: Dict[str, CoolStayLLM] = {}

    def get_llm(self, llm_type: str = "default") -> CoolStayLLM:
        """LLM 인스턴스 반환 (싱글톤 패턴)"""
        if llm_type not in self.llm_instances:
            if llm_type == "default":
                self.llm_instances[llm_type] = CoolStayLLM()
            else:
                # 다른 설정이 필요한 경우 확장 가능
                self.llm_instances[llm_type] = CoolStayLLM()

        return self.llm_instances[llm_type]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 LLM 인스턴스 상태 반환"""
        return {
            llm_type: llm.get_status()
            for llm_type, llm in self.llm_instances.items()
        }

    def test_all_connections(self) -> Dict[str, bool]:
        """모든 LLM 연결 테스트"""
        results = {}
        for llm_type, llm in self.llm_instances.items():
            if llm.is_initialized:
                test_response = llm._test_connection()
                results[llm_type] = test_response.success
            else:
                results[llm_type] = False
        return results


# 전역 LLM 매니저 인스턴스
llm_manager = LLMManager()


# 편의 함수들
def get_default_llm() -> CoolStayLLM:
    """기본 LLM 인스턴스 반환"""
    return llm_manager.get_llm("default")


def create_rag_chain(agent_name: str, description: str) -> Any:
    """RAG 체인 생성 편의 함수"""
    llm = get_default_llm()
    return llm.create_rag_answer_chain(agent_name, description)


def create_quality_evaluator() -> Any:
    """품질 평가 체인 생성 편의 함수"""
    llm = get_default_llm()
    return llm.create_quality_evaluation_chain()


def test_llm_connection() -> bool:
    """LLM 연결 테스트 편의 함수"""
    llm = get_default_llm()
    if llm.is_initialized:
        test_response = llm._test_connection()
        return test_response.success
    return False


if __name__ == "__main__":
    # LLM 테스트
    print("🤖 CoolStay LLM 모듈 테스트")
    print("=" * 50)

    # 기본 LLM 가져오기
    llm = get_default_llm()
    status = llm.get_status()

    print(f"📊 LLM 상태:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 연결 테스트
    if llm.is_initialized:
        print(f"\n🔍 연결 테스트 중...")
        test_response = llm._test_connection()

        if test_response.success:
            print(f"✅ 연결 성공!")
            print(f"   - 응답 시간: {test_response.response_time:.2f}초")
            print(f"   - 응답 내용: {test_response.content}")
        else:
            print(f"❌ 연결 실패: {test_response.error}")
    else:
        print(f"\n❌ LLM 초기화 실패: {status['initialization_error']}")