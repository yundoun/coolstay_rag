"""
CoolStay RAG data processing module

Provides document loading, preprocessing, and vector storage functionality.

Core Components:
- DocumentLoader: Multi-format document loading
- DocumentPreprocessor: Text cleaning and preprocessing
- DocumentChunker: Intelligent text chunking
- ChromaManager: ChromaDB vector store management
"""

from .loader import (
    DocumentLoader,
    DocumentMetadata,
    SupportedFormat,
    create_document_loader,
    load_documents_from_directory
)

from .preprocessor import (
    DocumentPreprocessor,
    PreprocessingConfig,
    TextCleaner,
    create_preprocessor,
    preprocess_documents_simple
)

from .chunker import (
    DocumentChunker,
    ChunkingStrategy,
    ChunkMetadata,
    create_chunker,
    chunk_documents_simple
)

from .chroma_manager import (
    ChromaManager,
    CollectionInfo,
    create_chroma_manager,
    get_collection_stats
)

from .retriever import (
    ChromaRetriever,
    RetrievalConfig,
    RetrievalResult,
    create_retriever,
    search_simple
)

__all__ = [
    # Document Loader
    "DocumentLoader",
    "DocumentMetadata",
    "SupportedFormat",
    "create_document_loader",
    "load_documents_from_directory",

    # Document Preprocessor
    "DocumentPreprocessor",
    "PreprocessingConfig",
    "TextCleaner",
    "create_preprocessor",
    "preprocess_documents_simple",

    # Document Chunker
    "DocumentChunker",
    "ChunkingStrategy",
    "ChunkMetadata",
    "create_chunker",
    "chunk_documents_simple",

    # Chroma Manager
    "ChromaManager",
    "CollectionInfo",
    "create_chroma_manager",
    "get_collection_stats",

    # Retriever
    "ChromaRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "create_retriever",
    "search_simple"
]