"""
CoolStay RAG 시스템 문서 검색 모듈

이 모듈은 벡터 저장소에서 문서를 검색하는 고급 기능을 제공합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..core.config import config, get_domain_config
from .chroma_manager import ChromaManager

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """검색 타입"""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximum Marginal Relevance
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"


class SearchScope(Enum):
    """검색 범위"""
    SINGLE_DOMAIN = "single_domain"
    MULTI_DOMAIN = "multi_domain"
    ALL_DOMAINS = "all_domains"
    SMART_ROUTING = "smart_routing"


@dataclass
class SearchResult:
    """검색 결과"""
    documents: List[Document]
    scores: Optional[List[float]] = None
    domain: Optional[str] = None
    query: Optional[str] = None
    search_type: Optional[SearchType] = None
    search_time: Optional[float] = None
    total_results: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalConfig:
    """검색 설정"""
    k: int = 5
    search_type: SearchType = SearchType.SIMILARITY
    score_threshold: float = 0.7
    mmr_diversity_score: float = 0.5
    max_results_per_domain: int = 10
    enable_reranking: bool = False


class DomainRetriever:
    """도메인별 문서 검색기"""

    def __init__(self, domain: str, chroma_manager: ChromaManager):
        """
        도메인 검색기 초기화

        Args:
            domain: 대상 도메인
            chroma_manager: ChromaDB 관리자
        """
        self.domain = domain
        self.chroma_manager = chroma_manager
        self.vectorstore = chroma_manager.get_vectorstore(domain)
        self.domain_config = get_domain_config(domain)

    def search(self, query: str, config: RetrievalConfig) -> SearchResult:
        """도메인에서 검색"""
        if not self.vectorstore:
            logger.warning(f"도메인 {self.domain}의 벡터 저장소가 없습니다.")
            return SearchResult(
                documents=[],
                domain=self.domain,
                query=query,
                search_type=config.search_type,
                total_results=0
            )

        start_time = time.time()

        try:
            # 검색 타입에 따른 검색 실행
            if config.search_type == SearchType.SIMILARITY:
                if config.score_threshold > 0:
                    results = self.vectorstore.similarity_search_with_score(query, k=config.k)
                    # 임계값 필터링
                    filtered_results = [(doc, score) for doc, score in results if score <= config.score_threshold]
                    documents = [doc for doc, score in filtered_results]
                    scores = [score for doc, score in filtered_results]
                else:
                    documents = self.vectorstore.similarity_search(query, k=config.k)
                    scores = None

            elif config.search_type == SearchType.MMR:
                documents = self.vectorstore.max_marginal_relevance_search(
                    query, k=config.k, lambda_mult=config.mmr_diversity_score
                )
                scores = None

            elif config.search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
                results = self.vectorstore.similarity_search_with_score(query, k=config.k * 2)
                filtered_results = [(doc, score) for doc, score in results if score <= config.score_threshold]
                documents = [doc for doc, score in filtered_results[:config.k]]
                scores = [score for doc, score in filtered_results[:config.k]]

            else:
                raise ValueError(f"지원하지 않는 검색 타입: {config.search_type}")

            search_time = time.time() - start_time

            # 결과에 도메인 정보 추가
            for doc in documents:
                if 'domain' not in doc.metadata:
                    doc.metadata['domain'] = self.domain

            return SearchResult(
                documents=documents,
                scores=scores,
                domain=self.domain,
                query=query,
                search_type=config.search_type,
                search_time=search_time,
                total_results=len(documents),
                metadata={
                    'domain_description': self.domain_config.description,
                    'domain_keywords': self.domain_config.keywords
                }
            )

        except Exception as e:
            logger.error(f"도메인 {self.domain} 검색 실패: {e}")
            return SearchResult(
                documents=[],
                domain=self.domain,
                query=query,
                search_type=config.search_type,
                search_time=time.time() - start_time,
                total_results=0,
                metadata={'error': str(e)}
            )

    def get_similar_documents(self, document: Document, k: int = 5) -> List[Document]:
        """유사한 문서 찾기"""
        if not self.vectorstore:
            return []

        try:
            return self.vectorstore.similarity_search(document.page_content, k=k)
        except Exception as e:
            logger.error(f"유사 문서 검색 실패 ({self.domain}): {e}")
            return []

    def is_available(self) -> bool:
        """검색기 사용 가능 여부"""
        return self.vectorstore is not None


class MultiDomainRetriever:
    """다중 도메인 문서 검색기"""

    def __init__(self, chroma_manager: ChromaManager):
        """
        다중 도메인 검색기 초기화

        Args:
            chroma_manager: ChromaDB 관리자
        """
        self.chroma_manager = chroma_manager
        self.domain_retrievers: Dict[str, DomainRetriever] = {}

        # 도메인별 검색기 초기화
        self._initialize_domain_retrievers()

    def _initialize_domain_retrievers(self):
        """도메인별 검색기 초기화"""
        for domain in config.domain_list:
            retriever = DomainRetriever(domain, self.chroma_manager)
            if retriever.is_available():
                self.domain_retrievers[domain] = retriever
                logger.info(f"✅ {domain} 검색기 초기화 완료")
            else:
                logger.warning(f"⚠️ {domain} 검색기 초기화 실패")

        logger.info(f"📊 다중 도메인 검색기: {len(self.domain_retrievers)}/{len(config.domain_list)}개 도메인 활성화")

    def search_single_domain(self, domain: str, query: str,
                           retrieval_config: RetrievalConfig) -> SearchResult:
        """단일 도메인에서 검색"""
        if domain not in self.domain_retrievers:
            logger.warning(f"도메인 {domain}의 검색기가 없습니다.")
            return SearchResult(documents=[], domain=domain, query=query, total_results=0)

        return self.domain_retrievers[domain].search(query, retrieval_config)

    def search_multiple_domains(self, domains: List[str], query: str,
                              retrieval_config: RetrievalConfig) -> Dict[str, SearchResult]:
        """다중 도메인에서 검색"""
        results = {}

        for domain in domains:
            if domain in self.domain_retrievers:
                result = self.domain_retrievers[domain].search(query, retrieval_config)
                results[domain] = result
            else:
                logger.warning(f"도메인 {domain}의 검색기가 없습니다.")
                results[domain] = SearchResult(
                    documents=[], domain=domain, query=query, total_results=0
                )

        return results

    def search_all_domains(self, query: str, retrieval_config: RetrievalConfig) -> Dict[str, SearchResult]:
        """모든 도메인에서 검색"""
        return self.search_multiple_domains(list(self.domain_retrievers.keys()), query, retrieval_config)

    def smart_search(self, query: str, retrieval_config: RetrievalConfig,
                    max_domains: int = 3) -> Tuple[Dict[str, SearchResult], List[str]]:
        """스마트 검색 (관련성 높은 도메인 자동 선택)"""
        # 1단계: 모든 도메인에서 간단한 검색 (k=2)
        quick_config = RetrievalConfig(
            k=2,
            search_type=retrieval_config.search_type,
            score_threshold=retrieval_config.score_threshold
        )

        preliminary_results = self.search_all_domains(query, quick_config)

        # 2단계: 도메인별 관련성 점수 계산
        domain_scores = {}
        for domain, result in preliminary_results.items():
            if result.documents and result.scores:
                # 평균 유사도 점수 (점수가 낮을수록 유사함)
                avg_score = sum(result.scores) / len(result.scores)
                domain_scores[domain] = avg_score
            elif result.documents:
                # 점수가 없으면 문서 개수로 판단
                domain_scores[domain] = 1.0 - (len(result.documents) / quick_config.k)
            else:
                domain_scores[domain] = 1.0  # 관련성 낮음

        # 3단계: 상위 도메인 선택
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1])
        selected_domains = [domain for domain, score in sorted_domains[:max_domains]]

        # 4단계: 선택된 도메인에서 상세 검색
        detailed_results = self.search_multiple_domains(selected_domains, query, retrieval_config)

        return detailed_results, selected_domains

    def aggregate_results(self, results: Dict[str, SearchResult],
                         max_total_results: int = 10) -> SearchResult:
        """검색 결과 통합"""
        all_documents = []
        all_scores = []
        search_times = []
        total_results = 0

        for domain, result in results.items():
            all_documents.extend(result.documents)
            total_results += result.total_results

            if result.scores:
                all_scores.extend(result.scores)

            if result.search_time:
                search_times.append(result.search_time)

        # 점수 기반 정렬 (점수가 있는 경우)
        if all_scores and len(all_scores) == len(all_documents):
            # 점수가 낮을수록 유사함 (distance)
            sorted_pairs = sorted(zip(all_documents, all_scores), key=lambda x: x[1])
            sorted_documents = [doc for doc, score in sorted_pairs[:max_total_results]]
            sorted_scores = [score for doc, score in sorted_pairs[:max_total_results]]
        else:
            # 점수가 없으면 도메인 순서대로
            sorted_documents = all_documents[:max_total_results]
            sorted_scores = None

        return SearchResult(
            documents=sorted_documents,
            scores=sorted_scores,
            query=list(results.values())[0].query if results else None,
            search_type=list(results.values())[0].search_type if results else None,
            search_time=max(search_times) if search_times else None,
            total_results=len(sorted_documents),
            metadata={
                'aggregated_from': list(results.keys()),
                'original_total_results': total_results,
                'domains_searched': len(results)
            }
        )

    def get_available_domains(self) -> List[str]:
        """사용 가능한 도메인 리스트"""
        return list(self.domain_retrievers.keys())

    def get_domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """도메인별 통계"""
        stats = {}
        for domain, retriever in self.domain_retrievers.items():
            document_count = self.chroma_manager.get_domain_document_count(domain)
            stats[domain] = {
                'available': retriever.is_available(),
                'document_count': document_count,
                'description': retriever.domain_config.description,
                'keywords': retriever.domain_config.keywords
            }
        return stats


class AdvancedRetriever:
    """고급 검색 기능 제공 클래스"""

    def __init__(self, multi_domain_retriever: MultiDomainRetriever):
        """
        고급 검색기 초기화

        Args:
            multi_domain_retriever: 다중 도메인 검색기
        """
        self.multi_domain_retriever = multi_domain_retriever

    def semantic_search(self, query: str, domains: Optional[List[str]] = None,
                       k: int = 5) -> SearchResult:
        """의미적 검색"""
        config = RetrievalConfig(
            k=k,
            search_type=SearchType.SIMILARITY,
            score_threshold=0.8
        )

        if domains:
            results = self.multi_domain_retriever.search_multiple_domains(domains, query, config)
        else:
            results, selected_domains = self.multi_domain_retriever.smart_search(query, config)

        return self.multi_domain_retriever.aggregate_results(results, k)

    def diverse_search(self, query: str, domains: Optional[List[str]] = None,
                      k: int = 5, diversity_score: float = 0.5) -> SearchResult:
        """다양성 검색 (MMR)"""
        config = RetrievalConfig(
            k=k,
            search_type=SearchType.MMR,
            mmr_diversity_score=diversity_score
        )

        if domains:
            results = self.multi_domain_retriever.search_multiple_domains(domains, query, config)
        else:
            results, selected_domains = self.multi_domain_retriever.smart_search(query, config, max_domains=5)

        return self.multi_domain_retriever.aggregate_results(results, k)

    def threshold_search(self, query: str, score_threshold: float = 0.7,
                        domains: Optional[List[str]] = None, max_results: int = 10) -> SearchResult:
        """임계값 기반 검색"""
        config = RetrievalConfig(
            k=max_results,
            search_type=SearchType.SIMILARITY_SCORE_THRESHOLD,
            score_threshold=score_threshold
        )

        if domains:
            results = self.multi_domain_retriever.search_multiple_domains(domains, query, config)
        else:
            results = self.multi_domain_retriever.search_all_domains(query, config)

        return self.multi_domain_retriever.aggregate_results(results, max_results)

    def find_similar_documents(self, reference_document: Document,
                              k: int = 5, exclude_same_domain: bool = False) -> List[Document]:
        """유사 문서 찾기"""
        query = reference_document.page_content
        ref_domain = reference_document.metadata.get('domain')

        if exclude_same_domain and ref_domain:
            # 같은 도메인 제외
            available_domains = self.multi_domain_retriever.get_available_domains()
            search_domains = [d for d in available_domains if d != ref_domain]
        else:
            search_domains = None

        result = self.semantic_search(query, domains=search_domains, k=k)
        return result.documents


# 편의 함수들
def create_retriever(chroma_manager: ChromaManager) -> MultiDomainRetriever:
    """다중 도메인 검색기 생성 편의 함수"""
    return MultiDomainRetriever(chroma_manager)


def search_documents(query: str, domains: Optional[List[str]] = None,
                    k: int = 5, search_type: str = "similarity") -> SearchResult:
    """문서 검색 편의 함수"""
    from .chroma_manager import ChromaManager

    chroma_manager = ChromaManager()
    retriever = MultiDomainRetriever(chroma_manager)
    advanced_retriever = AdvancedRetriever(retriever)

    if search_type == "similarity":
        return advanced_retriever.semantic_search(query, domains, k)
    elif search_type == "diverse":
        return advanced_retriever.diverse_search(query, domains, k)
    elif search_type == "threshold":
        return advanced_retriever.threshold_search(query, domains=domains, max_results=k)
    else:
        raise ValueError(f"지원하지 않는 검색 타입: {search_type}")


if __name__ == "__main__":
    # 검색기 테스트
    print("🔍 CoolStay 검색기 테스트")
    print("=" * 50)

    from .chroma_manager import ChromaManager

    # ChromaDB 관리자 및 검색기 초기화
    chroma_manager = ChromaManager()
    retriever = MultiDomainRetriever(chroma_manager)
    advanced_retriever = AdvancedRetriever(retriever)

    # 사용 가능한 도메인 확인
    available_domains = retriever.get_available_domains()
    print(f"📊 사용 가능한 도메인: {len(available_domains)}개")

    if available_domains:
        for domain in available_domains:
            doc_count = chroma_manager.get_domain_document_count(domain)
            print(f"   - {domain}: {doc_count:,}개 문서")

        # 검색 테스트
        test_query = "개발 워크플로우"
        print(f"\n🔍 검색 테스트: '{test_query}'")

        # 의미적 검색
        result = advanced_retriever.semantic_search(test_query, k=3)
        print(f"   📄 검색 결과: {len(result.documents)}개")
        print(f"   ⏱️  검색 시간: {result.search_time:.3f}초")

        if result.documents:
            for i, doc in enumerate(result.documents[:2], 1):
                domain = doc.metadata.get('domain', 'unknown')
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i}. [{domain}] {preview}...")
    else:
        print("❌ 사용 가능한 벡터 저장소가 없습니다.")
        print("   02_vector_stores.ipynb를 먼저 실행해주세요.")