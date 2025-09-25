"""
CoolStay RAG data processing module

Provides document loading, preprocessing, and vector storage functionality.

Core Components:
- MarkdownLoader: Multi-format document loading
- MarkdownPreprocessor: Text cleaning and preprocessing
- MarkdownChunker: Intelligent text chunking
- ChromaManager: ChromaDB vector store management
"""

from .loader import MarkdownLoader, DocumentValidator
from .preprocessor import MarkdownPreprocessor, PreprocessingStats
from .chunker import MarkdownChunker, ChunkingStrategy, ChunkingResult

# ChromaManager는 vectorstore 모듈에 있음
try:
    from ..vectorstore.chroma_manager import ChromaManager
except ImportError:
    ChromaManager = None

# Retriever는 vectorstore 모듈에 있음
try:
    from ..vectorstore.retriever import ChromaRetriever
except ImportError:
    ChromaRetriever = None

__all__ = [
    # Document Loader
    "MarkdownLoader",
    "DocumentValidator",

    # Document Preprocessor
    "MarkdownPreprocessor",
    "PreprocessingStats",

    # Document Chunker
    "MarkdownChunker",
    "ChunkingStrategy",
    "ChunkingResult",

    # ChromaDB (if available)
    "ChromaManager",
    "ChromaRetriever",
]