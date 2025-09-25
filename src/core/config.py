"""
CoolStay RAG 시스템 설정 관리 모듈

이 모듈은 CoolStay Multi-Agent RAG 시스템의 모든 설정을 중앙에서 관리합니다.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


@dataclass
class DomainConfig:
    """도메인별 설정 정보"""
    file: str
    description: str
    keywords: List[str]


@dataclass
class ChunkingConfig:
    """청킹 전략 설정"""
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    headers_to_split: List[tuple]


@dataclass
class ModelConfig:
    """모델 설정 정보"""
    name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None


class CoolStayConfig:
    """CoolStay RAG 시스템 중앙 설정 관리 클래스"""

    def __init__(self):
        self._setup_paths()
        self._setup_domains()
        self._setup_models()
        self._setup_chunking()
        self._setup_vector_store()
        self._setup_evaluation()

    def _setup_paths(self):
        """파일 경로 설정"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.notebooks_dir = self.project_root / "notebooks"
        self.task_management_dir = self.project_root / "task_management"
        self.logs_dir = self.task_management_dir / "logs"

        # ChromaDB 저장 경로
        self.chroma_db_dir = self.project_root / "chroma_db"

    def _setup_domains(self):
        """도메인 설정"""
        self.domain_config: Dict[str, DomainConfig] = {
            "hr_policy": DomainConfig(
                file="HR_Policy_Guide.md",
                description="인사정책, 근무시간, 휴가, 급여, 복리후생",
                keywords=["근무시간", "휴가", "급여", "복리후생", "인사", "채용", "평가"]
            ),
            "tech_policy": DomainConfig(
                file="Tech_Policy_Guide.md",
                description="기술정책, 개발환경, 코딩표준, 보안정책",
                keywords=["개발", "기술", "코딩", "보안", "테스트", "배포", "인프라"]
            ),
            "architecture": DomainConfig(
                file="Architecture_Guide.md",
                description="CMS 아키텍처, 시스템설계, 레이어구조",
                keywords=["아키텍처", "시스템", "설계", "구조", "레이어", "모듈"]
            ),
            "component": DomainConfig(
                file="Component_Guide.md",
                description="컴포넌트 가이드라인, UI/UX 표준",
                keywords=["컴포넌트", "UI", "UX", "디자인", "인터페이스", "사용자"]
            ),
            "deployment": DomainConfig(
                file="Deployment_Guide.md",
                description="배포프로세스, CI/CD, 환경관리",
                keywords=["배포", "CI/CD", "환경", "빌드", "릴리스", "운영"]
            ),
            "development": DomainConfig(
                file="Development_Process_Guide.md",
                description="개발프로세스, 워크플로우, 협업규칙",
                keywords=["프로세스", "워크플로우", "협업", "스프린트", "애자일"]
            ),
            "business_policy": DomainConfig(
                file="Business_Policy_Guide.md",
                description="비즈니스정책, 운영규칙, 의사결정",
                keywords=["비즈니스", "정책", "운영", "의사결정", "전략"]
            )
        }

        # 도메인 리스트 (순서 보장)
        self.domain_list = list(self.domain_config.keys())

        # 웹 검색 에이전트 설정
        self.web_search_config = {
            "description": "실시간 웹 검색, 최신 정보 조회",
            "keywords": ["최신", "뉴스", "업데이트", "검색", "웹"]
        }

    def _setup_models(self):
        """모델 설정"""
        # OpenAI 모델 설정
        self.openai_config = ModelConfig(
            name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=0.1,
            max_tokens=2000
        )

        # 임베딩 모델 설정 (Ollama)
        self.embedding_config = ModelConfig(
            name="bge-m3",
            api_key="",  # Ollama는 API 키 불필요
            base_url="http://localhost:11434"
        )

        # Tavily 웹 검색 설정
        self.tavily_config = ModelConfig(
            name="tavily-search",
            api_key=os.getenv("TAVILY_API_KEY", "")
        )

    def _setup_chunking(self):
        """청킹 전략 설정"""
        # 기본 청킹 설정 (RecursiveCharacterTextSplitter)
        self.chunking_config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            headers_to_split=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]
        )

        # 도메인별 최적 청킹 전략 (노트북에서 실험한 결과)
        self.optimal_chunking_strategies = {
            "hr_policy": {"strategy": "size_based", "chunk_size": 1500},
            "tech_policy": {"strategy": "size_based", "chunk_size": 1000},
            "architecture": {"strategy": "size_based", "chunk_size": 1500},
            "component": {"strategy": "size_based", "chunk_size": 1000},
            "deployment": {"strategy": "size_based", "chunk_size": 1000},
            "development": {"strategy": "size_based", "chunk_size": 1000},
            "business_policy": {"strategy": "size_based", "chunk_size": 1000}
        }

    def _setup_vector_store(self):
        """벡터 저장소 설정"""
        self.vector_store_config = {
            "chroma_db_impl": "duckdb+parquet",
            "anonymized_telemetry": False,
            "embedding_dimension": 1024,  # bge-m3 dimension
            "similarity_threshold": 0.7,
            "max_results": 5
        }

        # 도메인별 컬렉션 이름
        self.collection_names = {
            domain: f"coolstay_{domain}"
            for domain in self.domain_list
        }

    def _setup_evaluation(self):
        """평가 시스템 설정"""
        # ReAct 평가 차원 (6차원 60점 만점)
        self.evaluation_dimensions = {
            "relevance": {
                "name": "관련성",
                "description": "질문과 답변의 관련성",
                "max_score": 10
            },
            "accuracy": {
                "name": "정확성",
                "description": "정보의 정확성과 사실 여부",
                "max_score": 10
            },
            "completeness": {
                "name": "완전성",
                "description": "답변의 완전성과 포괄성",
                "max_score": 10
            },
            "clarity": {
                "name": "명확성",
                "description": "답변의 명확성과 이해도",
                "max_score": 10
            },
            "usefulness": {
                "name": "유용성",
                "description": "실무적 활용 가능성",
                "max_score": 10
            },
            "coherence": {
                "name": "일관성",
                "description": "논리적 일관성과 구조",
                "max_score": 10
            }
        }

        # 품질 임계값
        self.quality_thresholds = {
            "corrective_rag": 6.0,  # 10점 만점 중 6점 미만 시 재시도
            "hitl_required": 7.5,   # 7.5점 미만 시 HITL 트리거
            "excellent": 8.5        # 8.5점 이상 우수 답변
        }

        # HITL 설정
        self.hitl_config = {
            "max_iterations": 3,
            "timeout_seconds": 300,
            "auto_approve_threshold": 8.5
        }

    def get_domain_file_path(self, domain: str) -> Path:
        """도메인 파일 경로 반환"""
        if domain not in self.domain_config:
            raise ValueError(f"Unknown domain: {domain}")

        return self.data_dir / self.domain_config[domain].file

    def get_collection_name(self, domain: str) -> str:
        """도메인별 ChromaDB 컬렉션 이름 반환"""
        if domain not in self.collection_names:
            raise ValueError(f"Unknown domain: {domain}")

        return self.collection_names[domain]

    def get_optimal_chunk_size(self, domain: str) -> int:
        """도메인별 최적 청크 크기 반환"""
        if domain not in self.optimal_chunking_strategies:
            return self.chunking_config.chunk_size

        return self.optimal_chunking_strategies[domain]["chunk_size"]

    def validate_api_keys(self) -> Dict[str, bool]:
        """API 키 유효성 검사"""
        validation = {
            "openai": bool(self.openai_config.api_key),
            "tavily": bool(self.tavily_config.api_key),
        }
        return validation

    def get_environment_info(self) -> Dict[str, Any]:
        """환경 정보 반환"""
        return {
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "chroma_db_dir": str(self.chroma_db_dir),
            "domains_count": len(self.domain_list),
            "api_keys_valid": self.validate_api_keys(),
            "python_version": os.sys.version
        }


# 전역 설정 인스턴스
config = CoolStayConfig()


# 편의 함수들
def get_domain_config(domain: str) -> DomainConfig:
    """특정 도메인 설정 반환"""
    return config.domain_config[domain]


def get_all_domains() -> List[str]:
    """모든 도메인 리스트 반환"""
    return config.domain_list.copy()


def get_model_config(model_type: str) -> ModelConfig:
    """모델 설정 반환"""
    model_configs = {
        "openai": config.openai_config,
        "embedding": config.embedding_config,
        "tavily": config.tavily_config
    }

    if model_type not in model_configs:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_configs[model_type]


def is_development_mode() -> bool:
    """개발 모드 여부 확인"""
    return os.getenv("ENVIRONMENT", "development").lower() == "development"


def get_log_level() -> str:
    """로그 레벨 반환"""
    return os.getenv("LOG_LEVEL", "INFO").upper()


if __name__ == "__main__":
    # 설정 테스트
    print("🔧 CoolStay RAG 시스템 설정")
    print("=" * 50)

    env_info = config.get_environment_info()
    for key, value in env_info.items():
        print(f"{key}: {value}")

    print(f"\n📁 도메인: {len(config.domain_list)}개")
    for domain in config.domain_list:
        domain_config = config.domain_config[domain]
        print(f"  - {domain}: {domain_config.description}")

    print(f"\n🔑 API 키 상태:")
    api_validation = config.validate_api_keys()
    for api, is_valid in api_validation.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {api}: {'설정됨' if is_valid else '미설정'}")