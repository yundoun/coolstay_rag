"""
CoolStay RAG 시스템 문서 로더 모듈

이 모듈은 마크다운 파일을 로딩하고 전처리하는 기능을 제공합니다.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from ..core.config import config, get_domain_config

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """마크다운 파일 로더 클래스"""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        마크다운 로더 초기화

        Args:
            data_dir: 데이터 디렉토리 경로. None인 경우 기본 설정 사용
        """
        self.data_dir = data_dir or config.data_dir
        self.loaded_documents: Dict[str, str] = {}
        self.document_stats: Dict[str, Dict[str, Any]] = {}

    def load_single_file(self, file_path: Path) -> Optional[str]:
        """단일 마크다운 파일 로딩"""
        try:
            if not file_path.exists():
                logger.warning(f"파일이 존재하지 않습니다: {file_path}")
                return None

            if not file_path.suffix.lower() in ['.md', '.markdown']:
                logger.warning(f"마크다운 파일이 아닙니다: {file_path}")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"빈 파일입니다: {file_path}")
                return None

            logger.info(f"✅ 파일 로딩 완료: {file_path.name} ({len(content):,}자)")
            return content

        except Exception as e:
            logger.error(f"파일 로딩 실패 {file_path}: {e}")
            return None

    def load_domain_file(self, domain: str) -> Optional[str]:
        """도메인별 마크다운 파일 로딩"""
        try:
            domain_config = get_domain_config(domain)
            file_path = self.data_dir / domain_config.file

            content = self.load_single_file(file_path)

            if content:
                self.loaded_documents[domain] = content

                # 문서 통계 생성
                self.document_stats[domain] = self._analyze_document_structure(
                    content, domain, domain_config.file
                )

            return content

        except ValueError as e:
            logger.error(f"알 수 없는 도메인: {domain}")
            return None
        except Exception as e:
            logger.error(f"도메인 파일 로딩 실패 {domain}: {e}")
            return None

    def load_all_domains(self) -> Dict[str, str]:
        """모든 도메인 파일 로딩"""
        logger.info("📚 모든 도메인 파일 로딩 시작...")

        loaded_count = 0
        total_chars = 0

        for domain in config.domain_list:
            content = self.load_domain_file(domain)
            if content:
                loaded_count += 1
                total_chars += len(content)

        logger.info(f"✅ 문서 로딩 완료: {loaded_count}/{len(config.domain_list)}개 도메인, {total_chars:,}자")

        return self.loaded_documents.copy()

    def _analyze_document_structure(self, content: str, domain: str, filename: str) -> Dict[str, Any]:
        """문서 구조 분석"""
        lines = content.split('\n')

        # 헤더 구조 분석
        headers = {
            'h1': [],
            'h2': [],
            'h3': [],
            'h4': []
        }

        for i, line in enumerate(lines):
            if line.startswith('# ') and not line.startswith('## '):
                headers['h1'].append((i, line.strip('# ').strip()))
            elif line.startswith('## ') and not line.startswith('### '):
                headers['h2'].append((i, line.strip('# ').strip()))
            elif line.startswith('### ') and not line.startswith('#### '):
                headers['h3'].append((i, line.strip('# ').strip()))
            elif line.startswith('#### '):
                headers['h4'].append((i, line.strip('# ').strip()))

        # 특수 구조 분석
        code_blocks = len(re.findall(r'```[\s\S]*?```', content))
        bullet_points = len(re.findall(r'^[-*+]\s', content, re.MULTILINE))
        numbered_lists = len(re.findall(r'^\d+\.\s', content, re.MULTILINE))
        tables = len(re.findall(r'^\|.*\|.*$', content, re.MULTILINE))
        links = len(re.findall(r'\[.*?\]\(.*?\)', content))

        return {
            'domain': domain,
            'filename': filename,
            'total_lines': len(lines),
            'total_chars': len(content),
            'total_words': len(content.split()),
            'headers': headers,
            'structure_elements': {
                'code_blocks': code_blocks,
                'bullet_points': bullet_points,
                'numbered_lists': numbered_lists,
                'tables': tables,
                'links': links
            }
        }

    def get_document_stats(self) -> Dict[str, Dict[str, Any]]:
        """로딩된 문서들의 통계 반환"""
        return self.document_stats.copy()

    def print_loading_summary(self) -> None:
        """로딩 결과 요약 출력"""
        if not self.loaded_documents:
            print("📭 로딩된 문서가 없습니다.")
            return

        print("\n📊 문서 로딩 요약")
        print("=" * 60)

        total_chars = 0
        total_words = 0
        total_lines = 0

        for domain, stats in self.document_stats.items():
            print(f"\n🏷️  {domain}")
            print(f"   📝 파일: {stats['filename']}")
            print(f"   📏 크기: {stats['total_chars']:,}자, {stats['total_words']:,}단어, {stats['total_lines']:,}줄")
            print(f"   📋 헤더: H1({len(stats['headers']['h1'])}), H2({len(stats['headers']['h2'])}), H3({len(stats['headers']['h3'])})")

            structure = stats['structure_elements']
            structure_info = []
            for key, value in structure.items():
                if value > 0:
                    structure_info.append(f"{key}({value})")

            if structure_info:
                print(f"   🔧 구조: {', '.join(structure_info)}")

            total_chars += stats['total_chars']
            total_words += stats['total_words']
            total_lines += stats['total_lines']

        print(f"\n📈 전체 통계")
        print(f"   - 도메인: {len(self.loaded_documents)}개")
        print(f"   - 총 크기: {total_chars:,}자, {total_words:,}단어, {total_lines:,}줄")
        print(f"   - 평균 크기: {total_chars // len(self.loaded_documents):,}자/도메인")

    def get_domain_content(self, domain: str) -> Optional[str]:
        """특정 도메인 내용 반환"""
        return self.loaded_documents.get(domain)

    def is_domain_loaded(self, domain: str) -> bool:
        """도메인 로딩 여부 확인"""
        return domain in self.loaded_documents

    def get_loaded_domains(self) -> List[str]:
        """로딩된 도메인 리스트 반환"""
        return list(self.loaded_documents.keys())

    def reload_domain(self, domain: str) -> bool:
        """특정 도메인 재로딩"""
        logger.info(f"🔄 도메인 재로딩: {domain}")

        # 기존 데이터 정리
        if domain in self.loaded_documents:
            del self.loaded_documents[domain]
        if domain in self.document_stats:
            del self.document_stats[domain]

        # 재로딩
        content = self.load_domain_file(domain)
        return content is not None

    def clear(self) -> None:
        """로딩된 모든 데이터 정리"""
        self.loaded_documents.clear()
        self.document_stats.clear()
        logger.info("🗑️ 로딩된 데이터 정리 완료")


class DocumentValidator:
    """문서 유효성 검증 클래스"""

    @staticmethod
    def validate_markdown_content(content: str) -> Dict[str, Any]:
        """마크다운 내용 유효성 검증"""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        lines = content.split('\n')

        # 1. 기본 유효성 검사
        if not content.strip():
            validation_result["valid"] = False
            validation_result["errors"].append("빈 문서입니다.")
            return validation_result

        if len(content) < 100:
            validation_result["warnings"].append("매우 짧은 문서입니다 (100자 미만).")

        # 2. 헤더 구조 검증
        has_h1 = any(line.startswith('# ') and not line.startswith('## ') for line in lines)
        if not has_h1:
            validation_result["warnings"].append("H1 헤더가 없습니다.")

        # 연속된 빈 줄 체크
        empty_line_count = 0
        for line in lines:
            if line.strip() == '':
                empty_line_count += 1
                if empty_line_count > 3:
                    validation_result["suggestions"].append("연속된 빈 줄이 많습니다.")
                    break
            else:
                empty_line_count = 0

        # 3. 특수 문자 및 인코딩 검증
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            validation_result["errors"].append("UTF-8 인코딩 문제가 있습니다.")
            validation_result["valid"] = False

        # 4. 마크다운 문법 기본 검증
        # 불완전한 코드 블록 체크
        code_block_starts = content.count('```')
        if code_block_starts % 2 != 0:
            validation_result["warnings"].append("불완전한 코드 블록이 있을 수 있습니다.")

        # 5. 내용 품질 검증
        word_count = len(content.split())
        if word_count < 50:
            validation_result["warnings"].append("내용이 매우 적습니다.")

        return validation_result

    @staticmethod
    def validate_domain_file(domain: str, content: str) -> Dict[str, Any]:
        """도메인별 파일 유효성 검증"""
        basic_validation = DocumentValidator.validate_markdown_content(content)

        # 도메인별 추가 검증
        domain_config = get_domain_config(domain)
        keywords = domain_config.keywords

        # 키워드 포함 여부 검사
        content_lower = content.lower()
        found_keywords = [keyword for keyword in keywords if keyword.lower() in content_lower]

        if len(found_keywords) < len(keywords) * 0.5:  # 50% 미만의 키워드만 포함
            basic_validation["warnings"].append(
                f"도메인 키워드가 충분하지 않습니다. 예상: {keywords}, 발견: {found_keywords}"
            )

        return basic_validation


# 편의 함수들
def load_all_documents() -> Dict[str, str]:
    """모든 도메인 문서 로딩 편의 함수"""
    loader = MarkdownLoader()
    return loader.load_all_domains()


def load_domain_document(domain: str) -> Optional[str]:
    """특정 도메인 문서 로딩 편의 함수"""
    loader = MarkdownLoader()
    return loader.load_domain_file(domain)


def validate_document(content: str) -> Dict[str, Any]:
    """문서 유효성 검증 편의 함수"""
    return DocumentValidator.validate_markdown_content(content)


def get_document_structure_analysis(content: str, domain: str = "unknown", filename: str = "unknown.md") -> Dict[str, Any]:
    """문서 구조 분석 편의 함수"""
    loader = MarkdownLoader()
    return loader._analyze_document_structure(content, domain, filename)


if __name__ == "__main__":
    # 문서 로더 테스트
    print("📚 CoolStay 문서 로더 테스트")
    print("=" * 50)

    loader = MarkdownLoader()

    print("🔍 모든 도메인 파일 로딩 중...")
    documents = loader.load_all_domains()

    if documents:
        print(f"\n✅ {len(documents)}개 도메인 로딩 성공!")
        loader.print_loading_summary()

        # 유효성 검증 테스트
        print(f"\n🔍 문서 유효성 검증...")
        for domain, content in list(documents.items())[:2]:  # 처음 2개만 검증
            validation = DocumentValidator.validate_domain_file(domain, content)
            print(f"\n   {domain}: {'✅ 유효' if validation['valid'] else '❌ 오류'}")

            if validation['warnings']:
                for warning in validation['warnings'][:2]:
                    print(f"      ⚠️ {warning}")

            if validation['errors']:
                for error in validation['errors']:
                    print(f"      ❌ {error}")
    else:
        print("❌ 로딩된 문서가 없습니다.")
        print("\n💡 확인사항:")
        print(f"   1. 데이터 디렉토리 존재: {config.data_dir}")
        print(f"   2. 마크다운 파일 존재 여부")
        print(f"   3. 파일 읽기 권한")