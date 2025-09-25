"""
CoolStay RAG 시스템 문서 전처리 모듈

이 모듈은 마크다운 문서의 전처리 및 정규화 기능을 제공합니다.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStats:
    """전처리 통계"""
    original_length: int
    processed_length: int
    lines_removed: int
    characters_cleaned: int
    empty_lines_removed: int
    code_blocks_processed: int
    links_processed: int
    headers_normalized: int


class MarkdownPreprocessor:
    """마크다운 문서 전처리 클래스"""

    def __init__(self):
        # 정리할 패턴들
        self.cleanup_patterns = {
            # 과도한 공백 및 줄바꿈
            'excessive_newlines': re.compile(r'\n{4,}'),
            'excessive_spaces': re.compile(r'[ \t]{3,}'),
            'trailing_whitespace': re.compile(r'[ \t]+$', re.MULTILINE),

            # 잘못된 마크다운 문법
            'invalid_headers': re.compile(r'^#{7,}', re.MULTILINE),
            'empty_headers': re.compile(r'^#+\s*$', re.MULTILINE),

            # 특수 문자 정리
            'weird_quotes': re.compile(r'[""''„‚‹›«»]'),
            'weird_dashes': re.compile(r'[–—―]'),
            'weird_apostrophes': re.compile(r'[''‚]'),

            # HTML 태그 (기본적인 것만)
            'html_comments': re.compile(r'<!--.*?-->', re.DOTALL),
            'html_tags': re.compile(r'<[^>]+>'),
        }

        # 정규화 규칙
        self.normalization_rules = {
            'quotes': (r'[""''„‚]', '"'),
            'apostrophes': (r'[''‚]', "'"),
            'dashes': (r'[–—―]', '-'),
            'ellipsis': (r'\.{3,}', '...'),
        }

    def clean_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """텍스트 기본 정리"""
        stats = {
            'characters_removed': 0,
            'lines_removed': 0,
            'patterns_cleaned': 0
        }

        original_length = len(text)
        original_lines = text.count('\n')

        # 1. HTML 주석 및 태그 제거
        text = self.cleanup_patterns['html_comments'].sub('', text)
        text = self.cleanup_patterns['html_tags'].sub('', text)

        # 2. 잘못된 헤더 수정
        text = self.cleanup_patterns['invalid_headers'].sub('###### ', text)
        text = self.cleanup_patterns['empty_headers'].sub('', text)

        # 3. 공백 정리
        text = self.cleanup_patterns['trailing_whitespace'].sub('', text)
        text = self.cleanup_patterns['excessive_spaces'].sub('  ', text)
        text = self.cleanup_patterns['excessive_newlines'].sub('\n\n\n', text)

        # 통계 계산
        stats['characters_removed'] = original_length - len(text)
        stats['lines_removed'] = original_lines - text.count('\n')

        return text, stats

    def normalize_unicode(self, text: str) -> str:
        """유니코드 문자 정규화"""
        for pattern, replacement in self.normalization_rules.values():
            text = re.sub(pattern, replacement, text)
        return text

    def fix_markdown_syntax(self, text: str) -> Tuple[str, int]:
        """마크다운 문법 수정"""
        fixes_count = 0

        # 헤더 공백 수정
        text = re.sub(r'^(#+)([^\s])', r'\1 \2', text, flags=re.MULTILINE)
        fixes_count += len(re.findall(r'^(#+)([^\s])', text, flags=re.MULTILINE))

        # 리스트 아이템 공백 수정
        text = re.sub(r'^([-*+])([^\s])', r'\1 \2', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+\.)([^\s])', r'\1 \2', text, flags=re.MULTILINE)

        # 링크 문법 수정
        text = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', text)

        # 코드 블록 언어 표시 정리
        text = re.sub(r'^```\s*$', '```', text, flags=re.MULTILINE)

        return text, fixes_count

    def remove_empty_sections(self, text: str) -> Tuple[str, int]:
        """빈 섹션 제거"""
        # 연속된 헤더 (내용이 없는 섹션) 제거
        empty_sections_pattern = r'^(#+\s+[^\n]*)\n+(?=^#+\s+|\Z)'
        matches = re.findall(empty_sections_pattern, text, flags=re.MULTILINE)
        text = re.sub(empty_sections_pattern, '', text, flags=re.MULTILINE)

        return text, len(matches)

    def enhance_code_blocks(self, text: str) -> Tuple[str, int]:
        """코드 블록 개선"""
        code_blocks_processed = 0

        # 코드 블록에 언어 지정이 없는 경우 처리
        def add_language_hint(match):
            nonlocal code_blocks_processed
            code_content = match.group(1)

            # 간단한 언어 감지
            if re.search(r'\b(import|export|const|let|var|function)\b', code_content):
                language = 'javascript'
            elif re.search(r'\b(def|import|class|if __name__)\b', code_content):
                language = 'python'
            elif re.search(r'\b(SELECT|FROM|WHERE|INSERT)\b', code_content.upper()):
                language = 'sql'
            elif re.search(r'<[^>]+>', code_content):
                language = 'html'
            else:
                language = ''

            code_blocks_processed += 1
            return f'```{language}\n{code_content}```'

        text = re.sub(r'```\n(.*?)```', add_language_hint, text, flags=re.DOTALL)

        return text, code_blocks_processed

    def standardize_headers(self, text: str) -> Tuple[str, int]:
        """헤더 표준화"""
        headers_normalized = 0

        # 헤더 레벨 표준화 (최대 4레벨까지만)
        lines = text.split('\n')
        normalized_lines = []

        for line in lines:
            if re.match(r'^#+', line):
                # 헤더 레벨 계산
                header_match = re.match(r'^(#+)\s*(.*)', line)
                if header_match:
                    level = min(len(header_match.group(1)), 4)  # 최대 4레벨
                    content = header_match.group(2).strip()

                    if content:  # 내용이 있는 헤더만 유지
                        normalized_line = '#' * level + ' ' + content
                        normalized_lines.append(normalized_line)
                        headers_normalized += 1
                    continue

            normalized_lines.append(line)

        return '\n'.join(normalized_lines), headers_normalized

    def process_links(self, text: str) -> Tuple[str, int]:
        """링크 처리 및 정리"""
        links_processed = 0

        # 상대경로를 절대경로로 변환하거나 제거
        def process_link(match):
            nonlocal links_processed
            link_text = match.group(1)
            url = match.group(2)

            # 내부 링크나 앵커는 제거
            if url.startswith('#') or url.startswith('./') or url.startswith('../'):
                links_processed += 1
                return link_text

            # HTTP(S) 링크는 유지
            if url.startswith(('http://', 'https://')):
                return match.group(0)

            # 나머지는 텍스트만 유지
            links_processed += 1
            return link_text

        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', process_link, text)

        return text, links_processed

    def preprocess_content(self, content: str, domain: str = "unknown") -> Tuple[str, PreprocessingStats]:
        """전체 전처리 프로세스"""
        original_length = len(content)

        # 통계 누적
        total_stats = PreprocessingStats(
            original_length=original_length,
            processed_length=0,
            lines_removed=0,
            characters_cleaned=0,
            empty_lines_removed=0,
            code_blocks_processed=0,
            links_processed=0,
            headers_normalized=0
        )

        try:
            # 1. 기본 텍스트 정리
            content, clean_stats = self.clean_text(content)
            total_stats.characters_cleaned = clean_stats['characters_removed']
            total_stats.lines_removed = clean_stats['lines_removed']

            # 2. 유니코드 정규화
            content = self.normalize_unicode(content)

            # 3. 마크다운 문법 수정
            content, syntax_fixes = self.fix_markdown_syntax(content)

            # 4. 빈 섹션 제거
            content, empty_sections = self.remove_empty_sections(content)
            total_stats.empty_lines_removed = empty_sections

            # 5. 코드 블록 개선
            content, code_blocks = self.enhance_code_blocks(content)
            total_stats.code_blocks_processed = code_blocks

            # 6. 헤더 표준화
            content, headers = self.standardize_headers(content)
            total_stats.headers_normalized = headers

            # 7. 링크 처리
            content, links = self.process_links(content)
            total_stats.links_processed = links

            # 8. 최종 정리
            content = content.strip()
            total_stats.processed_length = len(content)

            logger.info(f"✅ {domain} 전처리 완료: {original_length:,} → {len(content):,}자")

            return content, total_stats

        except Exception as e:
            logger.error(f"전처리 실패 {domain}: {e}")
            return content, total_stats

    def preprocess_document(self, document: Document) -> Document:
        """Document 객체 전처리"""
        domain = document.metadata.get('domain', 'unknown')

        processed_content, stats = self.preprocess_content(document.page_content, domain)

        # 전처리 통계를 메타데이터에 추가
        processed_metadata = {
            **document.metadata,
            'preprocessing': {
                'original_length': stats.original_length,
                'processed_length': stats.processed_length,
                'reduction_ratio': (stats.original_length - stats.processed_length) / stats.original_length if stats.original_length > 0 else 0,
                'characters_cleaned': stats.characters_cleaned,
                'lines_removed': stats.lines_removed,
                'empty_lines_removed': stats.empty_lines_removed,
                'code_blocks_processed': stats.code_blocks_processed,
                'links_processed': stats.links_processed,
                'headers_normalized': stats.headers_normalized,
            }
        }

        return Document(
            page_content=processed_content,
            metadata=processed_metadata
        )

    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """여러 Document 객체 일괄 전처리"""
        processed_documents = []

        for i, doc in enumerate(documents):
            try:
                processed_doc = self.preprocess_document(doc)
                processed_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"문서 {i} 전처리 실패: {e}")
                # 원본 문서 유지
                processed_documents.append(doc)

        logger.info(f"✅ {len(processed_documents)}/{len(documents)}개 문서 전처리 완료")
        return processed_documents

    def validate_processed_content(self, content: str) -> Dict[str, Any]:
        """전처리된 내용 검증"""
        validation = {
            'valid': True,
            'warnings': [],
            'issues': []
        }

        # 기본 검증
        if not content.strip():
            validation['valid'] = False
            validation['issues'].append('빈 내용')
            return validation

        # 최소 길이 검증
        if len(content) < 50:
            validation['warnings'].append('매우 짧은 내용 (50자 미만)')

        # 마크다운 문법 검증
        if not re.search(r'^#+\s+', content, re.MULTILINE):
            validation['warnings'].append('헤더가 없음')

        # 불완전한 코드 블록 검증
        code_block_count = content.count('```')
        if code_block_count % 2 != 0:
            validation['issues'].append('불완전한 코드 블록')
            validation['valid'] = False

        # 과도한 중복 패턴 검증
        lines = content.split('\n')
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.5:
            validation['warnings'].append('과도한 중복 내용')

        return validation

    def get_preprocessing_summary(self, stats_list: List[PreprocessingStats]) -> Dict[str, Any]:
        """전처리 결과 요약"""
        if not stats_list:
            return {}

        total_original = sum(s.original_length for s in stats_list)
        total_processed = sum(s.processed_length for s in stats_list)
        total_cleaned = sum(s.characters_cleaned for s in stats_list)

        return {
            'total_documents': len(stats_list),
            'total_original_length': total_original,
            'total_processed_length': total_processed,
            'total_reduction': total_original - total_processed,
            'avg_reduction_ratio': (total_original - total_processed) / total_original if total_original > 0 else 0,
            'total_characters_cleaned': total_cleaned,
            'total_lines_removed': sum(s.lines_removed for s in stats_list),
            'total_empty_sections_removed': sum(s.empty_lines_removed for s in stats_list),
            'total_code_blocks_processed': sum(s.code_blocks_processed for s in stats_list),
            'total_links_processed': sum(s.links_processed for s in stats_list),
            'total_headers_normalized': sum(s.headers_normalized for s in stats_list),
        }


# 편의 함수들
def preprocess_text(content: str, domain: str = "unknown") -> str:
    """텍스트 전처리 편의 함수"""
    preprocessor = MarkdownPreprocessor()
    processed_content, _ = preprocessor.preprocess_content(content, domain)
    return processed_content


def preprocess_document(document: Document) -> Document:
    """Document 전처리 편의 함수"""
    preprocessor = MarkdownPreprocessor()
    return preprocessor.preprocess_document(document)


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Documents 일괄 전처리 편의 함수"""
    preprocessor = MarkdownPreprocessor()
    return preprocessor.preprocess_documents(documents)


def validate_content(content: str) -> Dict[str, Any]:
    """내용 검증 편의 함수"""
    preprocessor = MarkdownPreprocessor()
    return preprocessor.validate_processed_content(content)


if __name__ == "__main__":
    # 전처리 모듈 테스트
    print("🔧 CoolStay 전처리 모듈 테스트")
    print("=" * 50)

    # 테스트 마크다운 내용
    test_content = """
#  테스트 문서

##섹션 1
이것은    첫 번째   섹션입니다.
여기에는 "이상한" 따옴표와 –대시– 그리고 'apostrophe'가 있습니다.



### 하위 섹션 1.1
```
console.log("Hello World");
```

##

## 섹션 2
두 번째 섹션입니다.

<div>HTML 태그</div>

<!-- HTML 주석 -->

[링크](./local-file.md)

[외부링크](https://example.com)
"""

    preprocessor = MarkdownPreprocessor()

    print("📝 원본 내용:")
    print(f"   길이: {len(test_content)}자")
    print(f"   줄 수: {test_content.count(chr(10))}줄")

    # 전처리 실행
    processed_content, stats = preprocessor.preprocess_content(test_content, "test")

    print(f"\n✨ 전처리 결과:")
    print(f"   길이: {stats.processed_length}자 (원본 대비 {stats.processed_length/stats.original_length*100:.1f}%)")
    print(f"   정리된 문자: {stats.characters_cleaned}개")
    print(f"   제거된 줄: {stats.lines_removed}개")
    print(f"   처리된 코드블록: {stats.code_blocks_processed}개")
    print(f"   처리된 링크: {stats.links_processed}개")
    print(f"   정규화된 헤더: {stats.headers_normalized}개")

    # 검증
    validation = preprocessor.validate_processed_content(processed_content)
    print(f"\n🔍 검증 결과: {'✅ 유효' if validation['valid'] else '❌ 문제'}")

    if validation['warnings']:
        print("   ⚠️ 경고:")
        for warning in validation['warnings']:
            print(f"      - {warning}")

    if validation['issues']:
        print("   ❌ 문제:")
        for issue in validation['issues']:
            print(f"      - {issue}")

    print(f"\n📄 전처리된 내용 미리보기:")
    print("-" * 30)
    print(processed_content[:500] + "..." if len(processed_content) > 500 else processed_content)