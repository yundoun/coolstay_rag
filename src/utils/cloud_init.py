"""
Streamlit Cloud 배포를 위한 자동 초기화 유틸리티

이 모듈은 Streamlit Cloud 환경에서 벡터 DB가 없을 때 자동으로 초기화합니다.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def is_cloud_environment() -> bool:
    """클라우드 환경(Streamlit Cloud) 감지"""
    # Streamlit Cloud에서는 /mount/src/ 경로 사용
    return '/mount/src/' in str(Path(__file__).resolve())


def check_vectordb_exists(config) -> Dict[str, bool]:
    """벡터 DB 컬렉션 존재 여부 확인"""
    from src.vectorstore import ChromaManager

    try:
        chroma_manager = ChromaManager()
        results = {}

        for domain in config.get_domains():
            collection_name = config.get_collection_name(domain)
            collection_path = config.chroma_db_dir / domain
            results[domain] = collection_path.exists()

        return results
    except Exception as e:
        logger.error(f"벡터 DB 확인 실패: {e}")
        return {}


def initialize_vectordb_if_needed(config) -> Dict[str, Any]:
    """필요시 벡터 DB 자동 초기화"""
    result = {
        "initialized": False,
        "domains_processed": [],
        "errors": [],
        "message": ""
    }

    try:
        from src.vectorstore import ChromaManager
        from src.data import MarkdownLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # 벡터 DB 존재 여부 확인
        db_status = check_vectordb_exists(config)
        missing_domains = [domain for domain, exists in db_status.items() if not exists]

        if not missing_domains:
            result["message"] = "모든 벡터 DB가 이미 존재합니다."
            return result

        logger.info(f"벡터 DB 초기화 시작: {missing_domains}")

        # ChromaManager 초기화
        chroma_manager = ChromaManager()

        for domain in missing_domains:
            try:
                logger.info(f"🔄 {domain} 도메인 초기화 중...")

                # 1. MD 파일 로드
                file_path = config.get_domain_file_path(domain)
                if not file_path.exists():
                    error_msg = f"파일 없음: {file_path}"
                    logger.error(f"   ❌ {error_msg}")
                    result["errors"].append({domain: error_msg})
                    continue

                loader = MarkdownLoader(str(file_path))
                documents = loader.load()

                # 2. 텍스트 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.get_optimal_chunk_size(domain),
                    chunk_overlap=config.chunking_config.chunk_overlap,
                    separators=config.chunking_config.separators
                )

                split_docs = text_splitter.split_documents(documents)

                # 3. 메타데이터 추가
                for doc in split_docs:
                    doc.metadata.update({
                        'domain': domain,
                        'source': str(file_path.name),
                        'collection': config.get_collection_name(domain)
                    })

                # 4. 벡터스토어 생성
                collection_name = config.get_collection_name(domain)
                vectorstore = chroma_manager.create_vectorstore(
                    collection_name=collection_name,
                    documents=split_docs
                )

                if vectorstore:
                    logger.info(f"   ✅ {domain} 초기화 완료: {len(split_docs)}개 청크")
                    result["domains_processed"].append(domain)
                else:
                    error_msg = "벡터스토어 생성 실패"
                    logger.error(f"   ❌ {error_msg}")
                    result["errors"].append({domain: error_msg})

            except Exception as e:
                error_msg = f"초기화 오류: {str(e)}"
                logger.error(f"   ❌ {domain}: {error_msg}")
                result["errors"].append({domain: error_msg})

        result["initialized"] = len(result["domains_processed"]) > 0
        result["message"] = f"{len(result["domains_processed"])}개 도메인 초기화 완료"

        return result

    except Exception as e:
        result["errors"].append({"system": str(e)})
        result["message"] = f"초기화 실패: {str(e)}"
        logger.error(f"벡터 DB 초기화 실패: {e}")
        return result
