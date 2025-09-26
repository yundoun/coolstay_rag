#!/usr/bin/env python3
"""
벡터 데이터베이스 초기화 스크립트

이 스크립트는 CoolStay RAG 시스템의 벡터 데이터베이스를 초기화합니다.
data/ 폴더의 MD 문서들을 읽어 ChromaDB에 임베딩하여 저장합니다.

사용법:

1. 전체 초기화 (처음 실행 시):
python initialize_vectordb.py

2. 기존 DB 삭제 후 재생성:
python initialize_vectordb.py --reset

3. 특정 도메인만 초기화:
python initialize_vectordb.py --domain hr_policy

현재 상태:

- ChromaDB는 이미 초기화되어 있음 ✅
- 7개 도메인 컬렉션 모두 존재 ✅

언제 초기화가 필요한가?

1. 처음 설치 후
2. MD 문서 내용 변경 시
3. 임베딩 모델 변경 시
4. 청킹 전략 변경 시

벡터 DB 상태 확인:

ls -la chroma_db/
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config import CoolStayConfig
from src.vectorstore import ChromaManager
from src.data import MarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def initialize_domain_collection(
    domain: str,
    chroma_manager: ChromaManager,
    config: CoolStayConfig,
    reset: bool = False
) -> bool:
    """특정 도메인의 벡터스토어 초기화"""
    try:
        print(f"\n🔄 {domain} 도메인 초기화 중...")

        # 1. MD 파일 경로 확인
        file_path = config.get_domain_file_path(domain)
        if not file_path.exists():
            print(f"   ❌ 파일이 없습니다: {file_path}")
            return False

        print(f"   📄 파일 로드: {file_path}")

        # 2. 문서 로드
        loader = MarkdownLoader(str(file_path))
        documents = loader.load()
        print(f"   📑 로드된 문서: {len(documents)}개")

        # 3. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get_optimal_chunk_size(domain),
            chunk_overlap=config.chunking_config.chunk_overlap,
            separators=config.chunking_config.separators
        )

        split_docs = text_splitter.split_documents(documents)
        print(f"   ✂️  분할된 청크: {len(split_docs)}개")

        # 4. 메타데이터 추가
        for doc in split_docs:
            doc.metadata.update({
                'domain': domain,
                'source': str(file_path.name),
                'collection': config.get_collection_name(domain)
            })

        # 5. 벡터스토어 생성/업데이트
        if reset:
            # 기존 컬렉션 삭제
            collection_name = config.get_collection_name(domain)
            try:
                chroma_manager.client.delete_collection(collection_name)
                print(f"   🗑️  기존 컬렉션 삭제: {collection_name}")
            except:
                pass  # 컬렉션이 없는 경우 무시

        # 새 벡터스토어 생성
        vectorstore = chroma_manager.create_vectorstore(domain, split_docs)

        # 6. 검증
        collection = chroma_manager.client.get_collection(config.get_collection_name(domain))
        doc_count = collection.count()
        print(f"   ✅ 초기화 완료: {doc_count}개 벡터 저장됨")

        return True

    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='CoolStay RAG 벡터 DB 초기화')
    parser.add_argument(
        '--reset',
        action='store_true',
        help='기존 DB를 삭제하고 새로 생성'
    )
    parser.add_argument(
        '--domain',
        type=str,
        help='특정 도메인만 초기화 (예: hr_policy)'
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║     CoolStay RAG 시스템 - 벡터 데이터베이스 초기화            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 설정 로드
    config = CoolStayConfig()
    chroma_manager = ChromaManager()

    # DB 경로 확인
    db_path = config.chroma_db_dir
    print(f"📁 DB 경로: {db_path}")

    if args.reset and db_path.exists():
        print(f"🗑️  기존 DB 삭제 중...")
        shutil.rmtree(db_path)
        print(f"   ✅ 삭제 완료")

    # DB 디렉토리 생성
    db_path.mkdir(exist_ok=True)

    # 도메인 목록 결정
    if args.domain:
        if args.domain not in config.domain_list:
            print(f"❌ 알 수 없는 도메인: {args.domain}")
            print(f"   사용 가능한 도메인: {', '.join(config.domain_list)}")
            return 1
        domains = [args.domain]
    else:
        domains = config.domain_list

    print(f"\n🎯 초기화할 도메인: {', '.join(domains)}")

    # 각 도메인 초기화
    success_count = 0
    failed_domains = []

    for domain in domains:
        success = initialize_domain_collection(
            domain,
            chroma_manager,
            config,
            reset=args.reset
        )

        if success:
            success_count += 1
        else:
            failed_domains.append(domain)

    # 결과 출력
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      초기화 완료                              ║
╚══════════════════════════════════════════════════════════════╝
✅ 성공: {success_count}/{len(domains)} 도메인
""")

    if failed_domains:
        print(f"❌ 실패한 도메인: {', '.join(failed_domains)}")
        return 1

    # 최종 확인
    print("\n📊 최종 벡터스토어 상태:")
    for domain in domains:
        if domain not in failed_domains:
            try:
                collection = chroma_manager.client.get_collection(
                    config.get_collection_name(domain)
                )
                count = collection.count()
                print(f"   - {domain}: {count}개 벡터")
            except:
                print(f"   - {domain}: 확인 실패")

    print("\n✨ 벡터 데이터베이스 초기화가 완료되었습니다!")
    print("   이제 웹 애플리케이션을 실행할 수 있습니다: ./run_app.sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())