"""
CoolStay RAG 시스템 ChromaDB 벡터 저장소 관리 모듈

이 모듈은 ChromaDB 벡터 저장소의 생성, 로딩, 검색 기능을 제공합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..core.config import config, get_domain_config
from ..core.embeddings import get_langchain_embeddings

logger = logging.getLogger(__name__)


class ChromaManager:
    """ChromaDB 벡터 저장소 관리 클래스"""

    def __init__(self, embeddings=None, chroma_db_dir: Optional[Path] = None):
        """
        ChromaDB 관리자 초기화

        Args:
            embeddings: 임베딩 함수. None인 경우 기본 임베딩 사용
            chroma_db_dir: ChromaDB 저장 디렉토리. None인 경우 기본 설정 사용
        """
        self.embeddings = embeddings or get_langchain_embeddings()
        self.chroma_db_dir = chroma_db_dir or config.chroma_db_dir
        self.vectorstores: Dict[str, Chroma] = {}
        self.initialization_stats: Dict[str, Dict[str, Any]] = {}

        # ChromaDB 디렉토리 생성
        self.chroma_db_dir.mkdir(exist_ok=True)

    def create_domain_vectorstore(self, domain: str, documents: List[Document],
                                force_recreate: bool = False) -> Optional[Chroma]:
        """도메인별 벡터 저장소 생성"""
        if not documents:
            logger.warning(f"도메인 {domain}에 문서가 없습니다.")
            return None

        try:
            collection_name = config.get_collection_name(domain)
            persist_directory = str(self.chroma_db_dir / domain)

            # 기존 저장소 확인
            persist_path = Path(persist_directory)
            if persist_path.exists() and not force_recreate:
                logger.info(f"기존 벡터 저장소 발견: {domain} (force_recreate=False)")
                return self.load_domain_vectorstore(domain)

            # 새 벡터 저장소 생성
            start_time = time.time()

            if self.embeddings is None:
                raise ValueError("임베딩 함수가 초기화되지 않았습니다.")

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )

            creation_time = time.time() - start_time

            # 통계 저장
            self.initialization_stats[domain] = {
                'status': 'created',
                'document_count': len(documents),
                'collection_name': collection_name,
                'persist_directory': persist_directory,
                'creation_time': creation_time,
                'timestamp': time.time()
            }

            # 캐시에 저장
            self.vectorstores[domain] = vectorstore

            logger.info(f"✅ {domain} 벡터 저장소 생성 완료: {len(documents)}개 문서, {creation_time:.2f}초")
            return vectorstore

        except Exception as e:
            logger.error(f"❌ {domain} 벡터 저장소 생성 실패: {e}")
            self.initialization_stats[domain] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }
            return None

    def load_domain_vectorstore(self, domain: str) -> Optional[Chroma]:
        """기존 도메인 벡터 저장소 로딩"""
        # 캐시에서 확인
        if domain in self.vectorstores:
            return self.vectorstores[domain]

        try:
            collection_name = config.get_collection_name(domain)
            persist_directory = str(self.chroma_db_dir / domain)

            if not Path(persist_directory).exists():
                logger.warning(f"벡터 저장소 없음: {persist_directory}")
                return None

            if self.embeddings is None:
                raise ValueError("임베딩 함수가 초기화되지 않았습니다.")

            start_time = time.time()

            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )

            # 연결 테스트
            count = vectorstore._collection.count()
            load_time = time.time() - start_time

            # 통계 저장
            self.initialization_stats[domain] = {
                'status': 'loaded',
                'document_count': count,
                'collection_name': collection_name,
                'persist_directory': persist_directory,
                'load_time': load_time,
                'timestamp': time.time()
            }

            # 캐시에 저장
            self.vectorstores[domain] = vectorstore

            logger.info(f"✅ {domain} 벡터 저장소 로딩 완료: {count}개 문서, {load_time:.2f}초")
            return vectorstore

        except Exception as e:
            logger.error(f"❌ {domain} 벡터 저장소 로딩 실패: {e}")
            self.initialization_stats[domain] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }
            return None

    def load_all_vectorstores(self) -> Dict[str, Chroma]:
        """모든 도메인 벡터 저장소 로딩"""
        logger.info("📊 모든 도메인 벡터 저장소 로딩 시작...")

        loaded_count = 0
        for domain in config.domain_list:
            vectorstore = self.load_domain_vectorstore(domain)
            if vectorstore:
                loaded_count += 1

        logger.info(f"✅ 벡터 저장소 로딩 완료: {loaded_count}/{len(config.domain_list)}개 도메인")
        return self.vectorstores.copy()

    def create_all_vectorstores(self, documents_by_domain: Dict[str, List[Document]],
                               force_recreate: bool = False) -> Dict[str, Chroma]:
        """모든 도메인 벡터 저장소 생성"""
        logger.info("🚀 모든 도메인 벡터 저장소 생성 시작...")

        created_count = 0
        total_documents = 0

        for domain, documents in documents_by_domain.items():
            vectorstore = self.create_domain_vectorstore(domain, documents, force_recreate)
            if vectorstore:
                created_count += 1
                total_documents += len(documents)

        logger.info(f"🎉 벡터 저장소 생성 완료: {created_count}/{len(documents_by_domain)}개 도메인, {total_documents}개 문서")
        return self.vectorstores.copy()

    def delete_domain_vectorstore(self, domain: str) -> bool:
        """도메인 벡터 저장소 삭제"""
        try:
            persist_directory = Path(self.chroma_db_dir / domain)

            # 메모리에서 제거
            if domain in self.vectorstores:
                del self.vectorstores[domain]

            # 통계에서 제거
            if domain in self.initialization_stats:
                del self.initialization_stats[domain]

            # 디스크에서 제거
            if persist_directory.exists():
                import shutil
                shutil.rmtree(persist_directory)
                logger.info(f"✅ {domain} 벡터 저장소 삭제 완료")
                return True
            else:
                logger.warning(f"삭제할 벡터 저장소가 없습니다: {domain}")
                return False

        except Exception as e:
            logger.error(f"❌ {domain} 벡터 저장소 삭제 실패: {e}")
            return False

    def get_vectorstore(self, domain: str) -> Optional[Chroma]:
        """도메인 벡터 저장소 반환 (로딩 시도 포함)"""
        # 캐시에서 확인
        if domain in self.vectorstores:
            return self.vectorstores[domain]

        # 로딩 시도
        return self.load_domain_vectorstore(domain)

    def search_domain(self, domain: str, query: str, k: int = 5,
                     search_type: str = "similarity") -> List[Document]:
        """도메인에서 문서 검색"""
        vectorstore = self.get_vectorstore(domain)

        if not vectorstore:
            logger.warning(f"도메인 {domain}의 벡터 저장소가 없습니다.")
            return []

        try:
            if search_type == "similarity":
                return vectorstore.similarity_search(query, k=k)
            elif search_type == "mmr":
                return vectorstore.max_marginal_relevance_search(query, k=k)
            else:
                raise ValueError(f"지원하지 않는 검색 타입: {search_type}")

        except Exception as e:
            logger.error(f"도메인 {domain} 검색 실패: {e}")
            return []

    def search_domain_with_scores(self, domain: str, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """도메인에서 유사도 점수와 함께 문서 검색"""
        vectorstore = self.get_vectorstore(domain)

        if not vectorstore:
            logger.warning(f"도메인 {domain}의 벡터 저장소가 없습니다.")
            return []

        try:
            return vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"도메인 {domain} 점수 검색 실패: {e}")
            return []

    def add_documents_to_domain(self, domain: str, documents: List[Document]) -> bool:
        """도메인에 문서 추가"""
        vectorstore = self.get_vectorstore(domain)

        if not vectorstore:
            logger.error(f"도메인 {domain}의 벡터 저장소가 없습니다.")
            return False

        try:
            vectorstore.add_documents(documents)
            logger.info(f"✅ {domain}에 {len(documents)}개 문서 추가 완료")
            return True
        except Exception as e:
            logger.error(f"❌ {domain}에 문서 추가 실패: {e}")
            return False

    def get_domain_document_count(self, domain: str) -> int:
        """도메인 문서 수 반환"""
        vectorstore = self.get_vectorstore(domain)

        if not vectorstore:
            return 0

        try:
            return vectorstore._collection.count()
        except Exception as e:
            logger.error(f"도메인 {domain} 문서 수 조회 실패: {e}")
            return 0

    def get_all_stats(self) -> Dict[str, Any]:
        """모든 벡터 저장소 통계 반환"""
        total_documents = 0
        loaded_domains = 0
        failed_domains = 0

        domain_stats = {}

        for domain in config.domain_list:
            if domain in self.initialization_stats:
                stats = self.initialization_stats[domain]
                domain_stats[domain] = stats

                if stats['status'] in ['created', 'loaded']:
                    total_documents += stats.get('document_count', 0)
                    loaded_domains += 1
                else:
                    failed_domains += 1
            else:
                domain_stats[domain] = {'status': 'not_processed'}
                failed_domains += 1

        return {
            'total_domains': len(config.domain_list),
            'loaded_domains': loaded_domains,
            'failed_domains': failed_domains,
            'total_documents': total_documents,
            'chroma_db_directory': str(self.chroma_db_dir),
            'embeddings_initialized': self.embeddings is not None,
            'domain_stats': domain_stats
        }

    def print_status_summary(self) -> None:
        """상태 요약 출력"""
        stats = self.get_all_stats()

        print("\n📊 ChromaDB 벡터 저장소 상태")
        print("=" * 60)
        print(f"📁 저장 위치: {stats['chroma_db_directory']}")
        print(f"🔗 임베딩 연결: {'✅ 연결됨' if stats['embeddings_initialized'] else '❌ 연결 안됨'}")
        print(f"📈 전체 통계:")
        print(f"   - 전체 도메인: {stats['total_domains']}개")
        print(f"   - 로딩 성공: {stats['loaded_domains']}개")
        print(f"   - 실패: {stats['failed_domains']}개")
        print(f"   - 총 문서: {stats['total_documents']:,}개")

        print(f"\n🏷️ 도메인별 상태:")
        for domain, domain_stat in stats['domain_stats'].items():
            status = domain_stat['status']
            if status in ['created', 'loaded']:
                doc_count = domain_stat.get('document_count', 0)
                time_taken = domain_stat.get('creation_time', domain_stat.get('load_time', 0))
                print(f"   ✅ {domain}: {doc_count:,}개 문서 ({time_taken:.2f}초)")
            elif status == 'failed':
                error = domain_stat.get('error', 'Unknown error')
                print(f"   ❌ {domain}: 실패 ({error})")
            else:
                print(f"   ⏳ {domain}: 미처리")

    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        health = {
            'overall_status': 'healthy',
            'checks': {}
        }

        # 1. 임베딩 연결 확인
        health['checks']['embeddings'] = {
            'status': 'pass' if self.embeddings is not None else 'fail',
            'message': '임베딩 연결됨' if self.embeddings is not None else '임베딩 연결 안됨'
        }

        # 2. ChromaDB 디렉토리 확인
        health['checks']['storage_directory'] = {
            'status': 'pass' if self.chroma_db_dir.exists() else 'fail',
            'message': f"저장 디렉토리 존재: {self.chroma_db_dir}"
        }

        # 3. 벡터 저장소 연결 확인
        connected_domains = 0
        total_domains = len(config.domain_list)

        for domain in config.domain_list:
            if domain in self.vectorstores:
                try:
                    self.vectorstores[domain]._collection.count()
                    connected_domains += 1
                except:
                    pass

        health['checks']['vectorstores'] = {
            'status': 'pass' if connected_domains > 0 else 'fail',
            'message': f"{connected_domains}/{total_domains} 도메인 연결됨"
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


# 편의 함수들
def create_vectorstore_for_domain(domain: str, documents: List[Document],
                                force_recreate: bool = False) -> Optional[Chroma]:
    """도메인 벡터 저장소 생성 편의 함수"""
    manager = ChromaManager()
    return manager.create_domain_vectorstore(domain, documents, force_recreate)


def load_vectorstore_for_domain(domain: str) -> Optional[Chroma]:
    """도메인 벡터 저장소 로딩 편의 함수"""
    manager = ChromaManager()
    return manager.load_domain_vectorstore(domain)


def load_all_vectorstores() -> Dict[str, Chroma]:
    """모든 벡터 저장소 로딩 편의 함수"""
    manager = ChromaManager()
    return manager.load_all_vectorstores()


def search_in_domain(domain: str, query: str, k: int = 5) -> List[Document]:
    """도메인 검색 편의 함수"""
    manager = ChromaManager()
    return manager.search_domain(domain, query, k)


if __name__ == "__main__":
    # ChromaDB 관리자 테스트
    print("🗄️ CoolStay ChromaDB 관리자 테스트")
    print("=" * 50)

    manager = ChromaManager()

    # 헬스 체크
    health = manager.health_check()
    print(f"🏥 헬스 체크: {health['overall_status']}")

    for check_name, check_result in health['checks'].items():
        status_icon = "✅" if check_result['status'] == 'pass' else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    # 기존 벡터 저장소 로딩 시도
    print(f"\n🔍 기존 벡터 저장소 확인...")
    vectorstores = manager.load_all_vectorstores()

    if vectorstores:
        print(f"✅ {len(vectorstores)}개 도메인 벡터 저장소 발견")
        manager.print_status_summary()

        # 검색 테스트
        test_domain = list(vectorstores.keys())[0]
        test_query = "테스트 검색 쿼리"

        print(f"\n🔍 검색 테스트: {test_domain}")
        results = manager.search_domain(test_domain, test_query, k=2)
        print(f"   검색 결과: {len(results)}개")
    else:
        print("❌ 기존 벡터 저장소가 없습니다.")
        print("   01_data_processing.ipynb와 02_vector_stores.ipynb를 먼저 실행해주세요.")