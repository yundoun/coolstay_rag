"""
CoolStay RAG 시스템 문서 청킹 모듈

이 모듈은 마크다운 문서를 다양한 전략으로 청킹하는 기능을 제공합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..core.config import config, get_domain_config

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """청킹 전략 타입"""
    HEADER_BASED = "header_based"
    SIZE_BASED = "size_based"
    HYBRID = "hybrid"


@dataclass
class ChunkingResult:
    """청킹 결과"""
    documents: List[Document]
    strategy: ChunkingStrategy
    total_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    metadata: Dict[str, Any]


class MarkdownChunker:
    """마크다운 문서 청킹 클래스"""

    def __init__(self):
        self.chunking_config = config.chunking_config
        self.optimal_strategies = config.optimal_chunking_strategies

    def chunk_by_headers(self, content: str, domain: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """헤더 기반 청킹"""
        try:
            # 헤더 기반 분할기 설정
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.chunking_config.headers_to_split,
                strip_headers=False
            )

            # 헤더 기반 청킹 실행
            md_header_splits = markdown_splitter.split_text(content)

            # 메타데이터 강화
            enhanced_documents = []
            for i, doc in enumerate(md_header_splits):
                # 기본 메타데이터와 병합
                enhanced_metadata = {
                    **doc.metadata,
                    "domain": domain,
                    "chunk_type": ChunkingStrategy.HEADER_BASED.value,
                    "chunk_index": i,
                    "chunk_id": f"{domain}_header_{i}",
                    "content_length": len(doc.page_content),
                    "description": get_domain_config(domain).description,
                    "keywords": get_domain_config(domain).keywords,
                }

                # 추가 메타데이터 병합
                if metadata:
                    enhanced_metadata.update(metadata)

                enhanced_documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=enhanced_metadata
                    )
                )

            logger.info(f"✅ {domain} 헤더 기반 청킹 완료: {len(enhanced_documents)}개 청크")
            return enhanced_documents

        except Exception as e:
            logger.error(f"헤더 기반 청킹 실패 {domain}: {e}")
            return []

    def chunk_by_size(self, content: str, domain: str, chunk_size: Optional[int] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """크기 기반 청킹"""
        try:
            # 청크 크기 결정 (도메인별 최적화 또는 기본값)
            if chunk_size is None:
                chunk_size = config.get_optimal_chunk_size(domain)

            # RecursiveCharacterTextSplitter 설정
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=self.chunking_config.chunk_overlap,
                length_function=len,
                separators=self.chunking_config.separators
            )

            # 청킹 실행
            chunks = text_splitter.split_text(content)

            # Document 객체로 변환하며 메타데이터 추가
            documents = []
            for i, chunk in enumerate(chunks):
                enhanced_metadata = {
                    "domain": domain,
                    "chunk_type": ChunkingStrategy.SIZE_BASED.value,
                    "chunk_index": i,
                    "chunk_id": f"{domain}_size_{i}",
                    "chunk_size": len(chunk),
                    "target_chunk_size": chunk_size,
                    "content_length": len(chunk),
                    "description": get_domain_config(domain).description,
                    "keywords": get_domain_config(domain).keywords,
                }

                # 추가 메타데이터 병합
                if metadata:
                    enhanced_metadata.update(metadata)

                doc = Document(
                    page_content=chunk,
                    metadata=enhanced_metadata
                )
                documents.append(doc)

            logger.info(f"✅ {domain} 크기 기반 청킹 완료: {len(documents)}개 청크 (크기: {chunk_size})")
            return documents

        except Exception as e:
            logger.error(f"크기 기반 청킹 실패 {domain}: {e}")
            return []

    def chunk_hybrid(self, content: str, domain: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """하이브리드 청킹 (헤더 + 크기 기반)"""
        try:
            # 1단계: 헤더 기반 청킹
            header_chunks = self.chunk_by_headers(content, domain, metadata)

            # 2단계: 큰 청크들을 크기 기반으로 재분할
            final_documents = []
            max_size = config.get_optimal_chunk_size(domain)

            for i, doc in enumerate(header_chunks):
                if len(doc.page_content) <= max_size:
                    # 적절한 크기면 그대로 유지
                    doc.metadata["chunk_type"] = ChunkingStrategy.HYBRID.value
                    doc.metadata["sub_chunk_index"] = 0
                    final_documents.append(doc)
                else:
                    # 큰 청크는 추가 분할
                    size_chunks = self.chunk_by_size(
                        doc.page_content,
                        domain,
                        chunk_size=max_size,
                        metadata=doc.metadata
                    )

                    # 하이브리드 메타데이터 업데이트
                    for j, size_chunk in enumerate(size_chunks):
                        size_chunk.metadata.update({
                            "chunk_type": ChunkingStrategy.HYBRID.value,
                            "parent_chunk_index": i,
                            "sub_chunk_index": j,
                            "chunk_id": f"{domain}_hybrid_{i}_{j}"
                        })
                        final_documents.append(size_chunk)

            logger.info(f"✅ {domain} 하이브리드 청킹 완료: {len(final_documents)}개 청크")
            return final_documents

        except Exception as e:
            logger.error(f"하이브리드 청킹 실패 {domain}: {e}")
            return []

    def chunk_with_strategy(self, content: str, domain: str, strategy: ChunkingStrategy,
                           chunk_size: Optional[int] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """전략별 청킹 실행"""
        if strategy == ChunkingStrategy.HEADER_BASED:
            return self.chunk_by_headers(content, domain, metadata)
        elif strategy == ChunkingStrategy.SIZE_BASED:
            return self.chunk_by_size(content, domain, chunk_size, metadata)
        elif strategy == ChunkingStrategy.HYBRID:
            return self.chunk_hybrid(content, domain, metadata)
        else:
            raise ValueError(f"지원하지 않는 청킹 전략: {strategy}")

    def chunk_with_optimal_strategy(self, content: str, domain: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """도메인별 최적 전략으로 청킹"""
        optimal_config = self.optimal_strategies.get(domain, {})
        strategy_name = optimal_config.get("strategy", "size_based")
        chunk_size = optimal_config.get("chunk_size", config.chunking_config.chunk_size)

        try:
            strategy = ChunkingStrategy(strategy_name)
        except ValueError:
            logger.warning(f"잘못된 전략 {strategy_name}, size_based로 대체")
            strategy = ChunkingStrategy.SIZE_BASED

        return self.chunk_with_strategy(content, domain, strategy, chunk_size, metadata)

    def analyze_chunks(self, documents: List[Document], strategy: ChunkingStrategy) -> ChunkingResult:
        """청킹 결과 분석"""
        if not documents:
            return ChunkingResult(
                documents=documents,
                strategy=strategy,
                total_chunks=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                metadata={}
            )

        chunk_sizes = [len(doc.page_content) for doc in documents]

        return ChunkingResult(
            documents=documents,
            strategy=strategy,
            total_chunks=len(documents),
            avg_chunk_size=sum(chunk_sizes) / len(chunk_sizes),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            metadata={
                "domain": documents[0].metadata.get("domain", "unknown"),
                "total_content_length": sum(chunk_sizes),
                "chunk_distribution": self._analyze_chunk_distribution(chunk_sizes)
            }
        )

    def _analyze_chunk_distribution(self, chunk_sizes: List[int]) -> Dict[str, int]:
        """청크 크기 분포 분석"""
        distribution = {
            "very_small": 0,    # < 200
            "small": 0,         # 200-500
            "medium": 0,        # 500-1000
            "large": 0,         # 1000-1500
            "very_large": 0     # > 1500
        }

        for size in chunk_sizes:
            if size < 200:
                distribution["very_small"] += 1
            elif size < 500:
                distribution["small"] += 1
            elif size < 1000:
                distribution["medium"] += 1
            elif size < 1500:
                distribution["large"] += 1
            else:
                distribution["very_large"] += 1

        return distribution


class ChunkingExperiment:
    """청킹 전략 실험 클래스"""

    def __init__(self):
        self.chunker = MarkdownChunker()

    def compare_strategies(self, content: str, domain: str, chunk_sizes: List[int] = None) -> Dict[str, ChunkingResult]:
        """여러 전략 비교 실험"""
        if chunk_sizes is None:
            chunk_sizes = [800, 1000, 1200, 1500]

        results = {}

        # 헤더 기반 청킹
        try:
            header_docs = self.chunker.chunk_by_headers(content, domain)
            results["header_based"] = self.chunker.analyze_chunks(header_docs, ChunkingStrategy.HEADER_BASED)
        except Exception as e:
            logger.error(f"헤더 기반 실험 실패: {e}")

        # 다양한 크기로 크기 기반 청킹
        for size in chunk_sizes:
            try:
                size_docs = self.chunker.chunk_by_size(content, domain, chunk_size=size)
                results[f"size_{size}"] = self.chunker.analyze_chunks(size_docs, ChunkingStrategy.SIZE_BASED)
            except Exception as e:
                logger.error(f"크기 기반 실험 실패 (크기: {size}): {e}")

        # 하이브리드 청킹
        try:
            hybrid_docs = self.chunker.chunk_hybrid(content, domain)
            results["hybrid"] = self.chunker.analyze_chunks(hybrid_docs, ChunkingStrategy.HYBRID)
        except Exception as e:
            logger.error(f"하이브리드 실험 실패: {e}")

        return results

    def find_optimal_strategy(self, content: str, domain: str) -> Tuple[str, ChunkingResult]:
        """최적 청킹 전략 찾기"""
        results = self.compare_strategies(content, domain)

        if not results:
            return "size_1000", ChunkingResult([], ChunkingStrategy.SIZE_BASED, 0, 0.0, 0, 0, {})

        # 평가 기준으로 최적 전략 선택
        best_strategy = None
        best_score = float('inf')

        for strategy_name, result in results.items():
            if result.total_chunks == 0:
                continue

            # 점수 계산 (낮을수록 좋음)
            # 1. 청크 수가 너무 많거나 적으면 패널티
            chunk_count_penalty = abs(result.total_chunks - 10) * 0.5

            # 2. 평균 크기가 800-1200 범위에서 벗어나면 패널티
            size_penalty = max(0, abs(result.avg_chunk_size - 1000) - 200) * 0.01

            # 3. 크기 편차가 클수록 패널티
            variance_penalty = (result.max_chunk_size - result.min_chunk_size) * 0.001

            total_score = chunk_count_penalty + size_penalty + variance_penalty

            if total_score < best_score:
                best_score = total_score
                best_strategy = strategy_name

        return best_strategy or "size_1000", results.get(best_strategy, results[list(results.keys())[0]])

    def print_comparison_results(self, results: Dict[str, ChunkingResult]) -> None:
        """비교 결과 출력"""
        print("\n📊 청킹 전략 비교 결과")
        print("=" * 80)
        print(f"{'전략':<15} {'청크수':<8} {'평균크기':<10} {'최소크기':<10} {'최대크기':<10} {'분산':<10}")
        print("-" * 80)

        for strategy_name, result in results.items():
            if result.total_chunks > 0:
                variance = result.max_chunk_size - result.min_chunk_size
                print(f"{strategy_name:<15} {result.total_chunks:<8} "
                      f"{result.avg_chunk_size:<10.0f} {result.min_chunk_size:<10} "
                      f"{result.max_chunk_size:<10} {variance:<10}")


# 편의 함수들
def chunk_document(content: str, domain: str, strategy: str = "optimal",
                  chunk_size: Optional[int] = None) -> List[Document]:
    """문서 청킹 편의 함수"""
    chunker = MarkdownChunker()

    if strategy == "optimal":
        return chunker.chunk_with_optimal_strategy(content, domain)
    elif strategy == "header":
        return chunker.chunk_by_headers(content, domain)
    elif strategy == "size":
        return chunker.chunk_by_size(content, domain, chunk_size)
    elif strategy == "hybrid":
        return chunker.chunk_hybrid(content, domain)
    else:
        raise ValueError(f"지원하지 않는 전략: {strategy}")


def analyze_chunking_quality(documents: List[Document]) -> Dict[str, Any]:
    """청킹 품질 분석 편의 함수"""
    if not documents:
        return {"error": "빈 문서 리스트"}

    chunker = MarkdownChunker()
    strategy = ChunkingStrategy(documents[0].metadata.get("chunk_type", "size_based"))
    result = chunker.analyze_chunks(documents, strategy)

    return {
        "total_chunks": result.total_chunks,
        "avg_chunk_size": result.avg_chunk_size,
        "min_chunk_size": result.min_chunk_size,
        "max_chunk_size": result.max_chunk_size,
        "strategy": result.strategy.value,
        "chunk_distribution": result.metadata.get("chunk_distribution", {}),
        "quality_score": _calculate_quality_score(result)
    }


def _calculate_quality_score(result: ChunkingResult) -> float:
    """청킹 품질 점수 계산 (0-100, 높을수록 좋음)"""
    if result.total_chunks == 0:
        return 0.0

    # 기본 점수
    score = 100.0

    # 청크 수 패널티 (너무 많거나 적으면 감점)
    ideal_chunk_count = 15
    chunk_penalty = abs(result.total_chunks - ideal_chunk_count) * 2
    score -= min(chunk_penalty, 30)

    # 크기 일관성 점수 (편차가 작을수록 좋음)
    size_variance = result.max_chunk_size - result.min_chunk_size
    variance_penalty = min(size_variance / 100, 30)
    score -= variance_penalty

    # 적정 크기 점수 (800-1200이 이상적)
    avg_size = result.avg_chunk_size
    if 800 <= avg_size <= 1200:
        size_bonus = 10
    else:
        size_penalty = abs(avg_size - 1000) / 100
        size_bonus = -min(size_penalty, 20)
    score += size_bonus

    return max(0.0, min(100.0, score))


if __name__ == "__main__":
    # 청킹 모듈 테스트
    print("✂️ CoolStay 청킹 모듈 테스트")
    print("=" * 50)

    # 샘플 마크다운 내용
    sample_content = """
# 테스트 문서

## 섹션 1
이것은 첫 번째 섹션입니다. 여기에는 여러 줄의 텍스트가 포함되어 있습니다.

### 하위 섹션 1.1
더 세부적인 내용이 여기에 있습니다.

## 섹션 2
두 번째 섹션입니다. 이 섹션은 조금 더 긴 내용을 포함합니다.
여러 줄에 걸쳐 작성된 내용으로 청킹 테스트에 적합합니다.

### 하위 섹션 2.1
또 다른 하위 섹션입니다.

### 하위 섹션 2.2
마지막 하위 섹션입니다.
"""

    # 청킹 실험 실행
    experiment = ChunkingExperiment()
    results = experiment.compare_strategies(sample_content, "test_domain")

    if results:
        experiment.print_comparison_results(results)

        # 최적 전략 찾기
        best_strategy, best_result = experiment.find_optimal_strategy(sample_content, "test_domain")
        print(f"\n🎯 최적 전략: {best_strategy}")
        print(f"   - 청크 수: {best_result.total_chunks}")
        print(f"   - 평균 크기: {best_result.avg_chunk_size:.0f}자")
        print(f"   - 품질 점수: {_calculate_quality_score(best_result):.1f}/100")
    else:
        print("❌ 청킹 실험 실패")