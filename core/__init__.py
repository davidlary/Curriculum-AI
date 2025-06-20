"""
OpenBooks Core Module

Enhanced parallel processing system for acquiring, processing, and managing
open textbooks from various sources including OpenStax and other open educational resources.

Features:
- 20-core parallel processing by default
- Advanced text extraction with mathematical notation preservation
- Real-time search indexing across all content formats
- Comprehensive content analysis and cataloging

The module follows patterns established in the Pedigree project with significant
enhancements for high-performance textbook processing.
"""

__version__ = "0.2.0"
__author__ = "David Lary"
__email__ = "davidlary@me.com"

# Core functionality imports
from .config import OpenBooksConfig
from .book_discoverer import BookDiscoverer
from .repository_manager import RepositoryManager
from .content_processor import ContentProcessor
from .parallel_processor import ParallelProcessor, ProcessingTask, ProcessingResult
from .text_extractor import TextExtractor, ExtractedContent
from .search_indexer import SearchIndexer, SearchResult, IndexStats

__all__ = [
    "OpenBooksConfig",
    "BookDiscoverer", 
    "RepositoryManager",
    "ContentProcessor",
    "ParallelProcessor",
    "ProcessingTask",
    "ProcessingResult",
    "TextExtractor",
    "ExtractedContent",
    "SearchIndexer",
    "SearchResult",
    "IndexStats"
]