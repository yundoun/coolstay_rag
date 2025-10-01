"""
CoolStay RAG 시스템 임베딩 모델 관리 모듈

이 모듈은 Ollama의 bge-m3 임베딩 모델을 관리하고 RAG 시스템에서 사용할 수 있는
통합된 인터페이스를 제공합니다.
"""

import os
import logging
import time
import requests
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

try:
    from langchain_ollama import OllamaEmbeddings
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from .config import config, ModelConfig, get_env_var

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResponse:
    """임베딩 응답 결과"""
    embeddings: Optional[List[List[float]]] = None
    dimension: Optional[int] = None
    model: Optional[str] = None
    response_time: Optional[float] = None
    error: Optional[str] = None
    success: bool = True


class CoolStayEmbeddings:
    """CoolStay RAG 시스템용 임베딩 래퍼 클래스"""

    def __init__(self, model_config: Optional[ModelConfig] = None, use_openai: bool = None):
        """
        임베딩 모델 초기화

        Args:
            model_config: 모델 설정. None인 경우 기본 설정 사용
            use_openai: OpenAI 임베딩 사용 여부. None인 경우 자동 감지 (Streamlit Cloud에서는 자동으로 OpenAI 사용)
        """
        self.config = model_config or config.embedding_config

        # OpenAI 사용 여부 결정 (환경 변수 또는 자동 감지)
        if use_openai is None:
            # EMBEDDING_PROVIDER 환경 변수로 제어 가능
            provider = get_env_var("EMBEDDING_PROVIDER", "auto").lower()
            if provider == "openai":
                self.use_openai = True
            elif provider == "ollama":
                self.use_openai = False
            else:
                # 자동 감지: Ollama가 설치되어있지 않거나 클라우드 환경이면 OpenAI 사용
                # HAS_OLLAMA가 False이거나 클라우드 환경이면 OpenAI 사용
                is_cloud = self._is_cloud_environment()
                if is_cloud:
                    self.use_openai = True
                    logger.info("☁️ 클라우드 환경 감지: OpenAI 임베딩 사용")
                elif not HAS_OLLAMA:
                    self.use_openai = True
                    logger.info("📦 Ollama 미설치: OpenAI 임베딩 사용")
                else:
                    # Ollama 서버 연결 확인
                    self.use_openai = not self._check_ollama_connection()
        else:
            self.use_openai = use_openai

        self.embeddings = None
        self.is_initialized = False
        self.initialization_error = None
        self.dimension = None

        # 초기화 시도
        self.initialize()

    def _is_cloud_environment(self) -> bool:
        """클라우드 환경(Streamlit Cloud) 감지"""
        # Streamlit Cloud에서는 /mount/src/ 경로 사용
        import sys
        import os

        # 여러 방법으로 클라우드 환경 감지
        indicators = [
            '/mount/src/' in str(Path(__file__).resolve()),  # 파일 경로 확인
            '/mount/src/' in os.getcwd(),  # 작업 디렉토리 확인
            os.getenv('STREAMLIT_SHARING_MODE') is not None,  # Streamlit 환경 변수
        ]

        return any(indicators)

    def _check_ollama_connection(self) -> bool:
        """Ollama 서버 실제 연결 가능 여부 확인"""
        try:
            response = requests.get(f"{self.config.base_url}/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False

    def initialize(self) -> bool:
        """임베딩 모델 초기화"""
        try:
            if self.use_openai:
                # OpenAI 임베딩 사용
                openai_api_key = get_env_var("OPENAI_API_KEY", "")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",  # 저렴하고 빠른 모델
                    openai_api_key=openai_api_key
                )
                model_name = "text-embedding-3-small"
                self.dimension = 1536  # text-embedding-3-small의 차원

            else:
                # Ollama 임베딩 사용
                if not HAS_OLLAMA:
                    raise ImportError("langchain_ollama 패키지가 설치되지 않았습니다.")

                # Ollama 서버 연결 확인
                if not self._check_ollama_server():
                    raise ConnectionError("Ollama 서버에 연결할 수 없습니다. 'ollama serve' 명령으로 서버를 시작해주세요.")

                # 모델 존재 확인
                if not self._check_model_exists():
                    raise ValueError(f"모델 '{self.config.name}'이 설치되지 않았습니다. 'ollama pull {self.config.name}' 명령으로 설치해주세요.")

                # OllamaEmbeddings 인스턴스 생성
                self.embeddings = OllamaEmbeddings(
                    model=self.config.name,
                    base_url=self.config.base_url
                )
                model_name = self.config.name

            # 연결 테스트
            test_response = self._test_connection()
            if test_response.success:
                self.is_initialized = True
                if not self.dimension:
                    self.dimension = test_response.dimension
                provider = "OpenAI" if self.use_openai else "Ollama"
                logger.info(f"✅ 임베딩 모델 초기화 성공: {model_name} ({provider}, {self.dimension}차원)")
                return True
            else:
                raise Exception(f"연결 테스트 실패: {test_response.error}")

        except Exception as e:
            self.initialization_error = str(e)
            self.is_initialized = False
            logger.error(f"❌ 임베딩 모델 초기화 실패: {e}")
            return False

    def _check_ollama_server(self) -> bool:
        """Ollama 서버 연결 확인"""
        if self.use_openai:
            return True  # OpenAI 사용 시 Ollama 서버 체크 불필요
        try:
            response = requests.get(f"{self.config.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _check_model_exists(self) -> bool:
        """설치된 모델 확인"""
        if self.use_openai:
            return True  # OpenAI 사용 시 모델 체크 불필요
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "").split(":")[0] for model in models]
                return self.config.name in model_names
            return False
        except requests.exceptions.RequestException:
            return False

    def _test_connection(self) -> EmbeddingResponse:
        """임베딩 모델 연결 테스트"""
        try:
            test_text = "임베딩 테스트"
            start_time = time.time()
            test_embedding = self.embeddings.embed_query(test_text)
            response_time = time.time() - start_time

            model_name = "OpenAI" if self.use_openai else self.config.name

            return EmbeddingResponse(
                embeddings=[test_embedding],
                dimension=len(test_embedding),
                model=model_name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            model_name = "OpenAI" if self.use_openai else self.config.name
            return EmbeddingResponse(
                error=str(e),
                model=model_name,
                success=False
            )

    def embed_query(self, text: str) -> EmbeddingResponse:
        """단일 쿼리 임베딩"""
        if not self.is_initialized:
            return EmbeddingResponse(
                error=f"임베딩 모델이 초기화되지 않았습니다: {self.initialization_error}",
                model=self.config.name,
                success=False
            )

        try:
            start_time = time.time()
            embedding = self.embeddings.embed_query(text)
            response_time = time.time() - start_time

            return EmbeddingResponse(
                embeddings=[embedding],
                dimension=len(embedding),
                model=self.config.name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            logger.error(f"쿼리 임베딩 실패: {e}")
            return EmbeddingResponse(
                error=str(e),
                model=self.config.name,
                success=False
            )

    def embed_documents(self, texts: List[str]) -> EmbeddingResponse:
        """다중 문서 임베딩"""
        if not self.is_initialized:
            return EmbeddingResponse(
                error=f"임베딩 모델이 초기화되지 않았습니다: {self.initialization_error}",
                model=self.config.name,
                success=False
            )

        try:
            start_time = time.time()
            embeddings = self.embeddings.embed_documents(texts)
            response_time = time.time() - start_time

            dimension = len(embeddings[0]) if embeddings else 0

            return EmbeddingResponse(
                embeddings=embeddings,
                dimension=dimension,
                model=self.config.name,
                response_time=response_time,
                success=True
            )

        except Exception as e:
            logger.error(f"문서 임베딩 실패: {e}")
            return EmbeddingResponse(
                error=str(e),
                model=self.config.name,
                success=False
            )

    def embed_documents_with_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """메타데이터가 포함된 문서들 임베딩"""
        if not documents:
            return {
                "embeddings": [],
                "texts": [],
                "metadata": [],
                "success": True,
                "message": "빈 문서 리스트"
            }

        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]

        response = self.embed_documents(texts)

        if response.success:
            return {
                "embeddings": response.embeddings,
                "texts": texts,
                "metadata": metadata,
                "dimension": response.dimension,
                "model": response.model,
                "response_time": response.response_time,
                "success": True,
                "document_count": len(documents)
            }
        else:
            return {
                "embeddings": [],
                "texts": [],
                "metadata": [],
                "error": response.error,
                "success": False
            }

    def get_langchain_embeddings(self):
        """LangChain 호환 임베딩 객체 반환"""
        if not self.is_initialized:
            logger.warning("임베딩 모델이 초기화되지 않았습니다.")
            return None
        return self.embeddings

    def get_status(self) -> Dict[str, Any]:
        """임베딩 모델 상태 정보 반환"""
        status = {
            "initialized": self.is_initialized,
            "model": self.config.name,
            "base_url": self.config.base_url,
            "dimension": self.dimension,
            "initialization_error": self.initialization_error
        }

        # 서버 상태 확인
        if self.is_initialized:
            status["server_connected"] = self._check_ollama_server()
            status["model_available"] = self._check_model_exists()
        else:
            status["server_connected"] = False
            status["model_available"] = False

        return status

    def benchmark_performance(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """임베딩 성능 벤치마크"""
        if not self.is_initialized:
            return {
                "success": False,
                "error": "임베딩 모델이 초기화되지 않았습니다."
            }

        if sample_texts is None:
            sample_texts = [
                "CoolStay는 혁신적인 숙박 플랫폼입니다.",
                "개발팀은 React와 Node.js를 사용합니다.",
                "CI/CD 파이프라인을 통해 자동 배포를 수행합니다.",
                "사용자 경험을 위해 UI/UX 가이드라인을 준수합니다.",
                "보안 정책에 따라 코드 리뷰를 진행합니다."
            ]

        results = {}

        # 단일 쿼리 성능 테스트
        single_response = self.embed_query(sample_texts[0])
        results["single_query"] = {
            "success": single_response.success,
            "response_time": single_response.response_time,
            "dimension": single_response.dimension
        }

        # 다중 문서 성능 테스트
        multiple_response = self.embed_documents(sample_texts)
        results["multiple_documents"] = {
            "success": multiple_response.success,
            "document_count": len(sample_texts),
            "total_response_time": multiple_response.response_time,
            "avg_response_time": multiple_response.response_time / len(sample_texts) if multiple_response.response_time else 0,
            "dimension": multiple_response.dimension
        }

        results["model"] = self.config.name
        results["overall_success"] = single_response.success and multiple_response.success

        return results


class EmbeddingManager:
    """여러 임베딩 인스턴스를 관리하는 매니저 클래스"""

    def __init__(self):
        self.embedding_instances: Dict[str, CoolStayEmbeddings] = {}

    def get_embeddings(self, embedding_type: str = "default") -> CoolStayEmbeddings:
        """임베딩 인스턴스 반환 (싱글톤 패턴)"""
        if embedding_type not in self.embedding_instances:
            if embedding_type == "default":
                self.embedding_instances[embedding_type] = CoolStayEmbeddings()
            else:
                # 다른 설정이 필요한 경우 확장 가능
                self.embedding_instances[embedding_type] = CoolStayEmbeddings()

        return self.embedding_instances[embedding_type]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 임베딩 인스턴스 상태 반환"""
        return {
            embedding_type: embedding.get_status()
            for embedding_type, embedding in self.embedding_instances.items()
        }

    def test_all_connections(self) -> Dict[str, bool]:
        """모든 임베딩 연결 테스트"""
        results = {}
        for embedding_type, embedding in self.embedding_instances.items():
            if embedding.is_initialized:
                test_response = embedding._test_connection()
                results[embedding_type] = test_response.success
            else:
                results[embedding_type] = False
        return results


# 전역 임베딩 매니저 인스턴스
embedding_manager = EmbeddingManager()


# 편의 함수들
def get_default_embeddings() -> CoolStayEmbeddings:
    """기본 임베딩 인스턴스 반환"""
    return embedding_manager.get_embeddings("default")


def get_langchain_embeddings():
    """LangChain 호환 임베딩 객체 반환"""
    embeddings = get_default_embeddings()
    return embeddings.get_langchain_embeddings()


def test_embedding_connection() -> bool:
    """임베딩 연결 테스트 편의 함수"""
    embeddings = get_default_embeddings()
    if embeddings.is_initialized:
        test_response = embeddings._test_connection()
        return test_response.success
    return False


def embed_query(text: str) -> List[float]:
    """단일 쿼리 임베딩 편의 함수"""
    embeddings = get_default_embeddings()
    response = embeddings.embed_query(text)

    if response.success and response.embeddings:
        return response.embeddings[0]
    else:
        raise RuntimeError(f"임베딩 실패: {response.error}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """다중 문서 임베딩 편의 함수"""
    embeddings = get_default_embeddings()
    response = embeddings.embed_documents(texts)

    if response.success and response.embeddings:
        return response.embeddings
    else:
        raise RuntimeError(f"임베딩 실패: {response.error}")


if __name__ == "__main__":
    # 임베딩 모듈 테스트
    print("🔗 CoolStay 임베딩 모듈 테스트")
    print("=" * 50)

    # 기본 임베딩 가져오기
    embeddings = get_default_embeddings()
    status = embeddings.get_status()

    print(f"📊 임베딩 상태:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 연결 테스트
    if embeddings.is_initialized:
        print(f"\n🔍 연결 테스트 중...")
        test_response = embeddings._test_connection()

        if test_response.success:
            print(f"✅ 연결 성공!")
            print(f"   - 모델: {test_response.model}")
            print(f"   - 차원: {test_response.dimension}")
            print(f"   - 응답 시간: {test_response.response_time:.3f}초")

            # 성능 벤치마크
            print(f"\n⚡ 성능 벤치마크 실행...")
            benchmark_result = embeddings.benchmark_performance()

            if benchmark_result["overall_success"]:
                print(f"   - 단일 쿼리: {benchmark_result['single_query']['response_time']:.3f}초")
                print(f"   - 다중 문서 (5개): {benchmark_result['multiple_documents']['total_response_time']:.3f}초")
                print(f"   - 평균 문서당: {benchmark_result['multiple_documents']['avg_response_time']:.3f}초")
            else:
                print(f"   ❌ 벤치마크 실패")
        else:
            print(f"❌ 연결 실패: {test_response.error}")
    else:
        print(f"\n❌ 임베딩 초기화 실패: {status['initialization_error']}")
        print(f"\n💡 해결 방법:")
        print(f"   1. Ollama 서버 시작: ollama serve")
        print(f"   2. 모델 설치: ollama pull bge-m3")
        print(f"   3. 서버 상태 확인: curl {config.embedding_config.base_url}/api/version")