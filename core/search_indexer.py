"""
Advanced search indexing system for textbook content.

This module creates comprehensive searchable indexes across all textbook formats,
enabling fast full-text search, mathematical formula search, and semantic search.

Goal: Provide lightning-fast search capabilities across entire textbook collections.
"""

import logging
import json
import sqlite3
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
import time
from collections import defaultdict, Counter
import math

from .config import OpenBooksConfig
from .text_extractor import ExtractedContent

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with relevance scoring."""
    book_id: str
    book_title: str
    chapter_number: str
    chapter_title: str
    content_snippet: str
    relevance_score: float
    match_type: str  # 'text', 'formula', 'title', 'author'
    source_path: str
    page_number: Optional[int] = None


@dataclass
class IndexStats:
    """Statistics about the search index."""
    total_books: int
    total_chapters: int
    total_words: int
    total_formulas: int
    unique_terms: int
    index_size_mb: float
    last_updated: float


class SearchIndexer:
    """
    High-performance search indexing system for textbook collections.
    
    Features:
    - Full-text search with TF-IDF scoring
    - Mathematical formula indexing
    - Chapter and section structure preservation
    - Fast autocomplete suggestions
    - Relevance-based result ranking
    """
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize search indexer with configuration."""
        self.config = config
        self.index_dir = Path(config.search_index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Database files
        self.db_path = self.index_dir / "search_index.db"
        self.stopwords = self._load_stopwords()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"SearchIndexer initialized with index at {self.index_dir}")
    
    def index_content(self, content: ExtractedContent) -> bool:
        """
        Index extracted content for search.
        
        Args:
            content: ExtractedContent object to index
            
        Returns:
            True if indexing successful, False otherwise
        """
        try:
            logger.info(f"Indexing content: {content.title}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert book record
                book_id = content.content_hash
                cursor.execute('''
                    INSERT OR REPLACE INTO books 
                    (id, title, authors, source_path, format_type, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    book_id,
                    content.title,
                    json.dumps(content.authors),
                    content.source_path,
                    content.format_type,
                    content.content_hash,
                    time.time()
                ))
                
                # Index chapters
                for chapter in content.chapters:
                    self._index_chapter(cursor, book_id, chapter)
                
                # Index mathematical formulas
                for formula in content.mathematical_notation:
                    self._index_formula(cursor, book_id, formula)
                
                # Build search terms index
                self._build_terms_index(cursor, book_id, content)
                
                conn.commit()
                logger.info(f"Successfully indexed {content.title}")
                return True
                
        except Exception as e:
            logger.error(f"Error indexing content {content.title}: {e}")
            return False
    
    def search(self, 
               query: str, 
               max_results: int = 50,
               search_type: str = 'all') -> List[SearchResult]:
        """
        Search the index for matching content.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_type: Type of search ('all', 'text', 'formula', 'title')
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        query = query.strip().lower()
        if not query:
            return []
        
        logger.debug(f"Searching for: '{query}' (type: {search_type})")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                results = []
                
                if search_type in ['all', 'text']:
                    results.extend(self._search_text(conn, query, max_results))
                
                if search_type in ['all', 'formula']:
                    results.extend(self._search_formulas(conn, query, max_results))
                
                if search_type in ['all', 'title']:
                    results.extend(self._search_titles(conn, query, max_results))
                
                # Remove duplicates and sort by relevance
                unique_results = self._deduplicate_results(results)
                sorted_results = sorted(unique_results, 
                                      key=lambda x: x.relevance_score, 
                                      reverse=True)
                
                return sorted_results[:max_results]
                
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []
    
    def get_suggestions(self, partial_query: str, max_suggestions: int = 10) -> List[str]:
        """
        Get autocomplete suggestions for a partial query.
        
        Args:
            partial_query: Partial search query
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested query completions
        """
        partial_query = partial_query.strip().lower()
        if len(partial_query) < 2:
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find terms that start with the partial query
                cursor.execute('''
                    SELECT term, frequency FROM terms 
                    WHERE term LIKE ? 
                    ORDER BY frequency DESC 
                    LIMIT ?
                ''', (f"{partial_query}%", max_suggestions))
                
                suggestions = [row[0] for row in cursor.fetchall()]
                return suggestions
                
        except Exception as e:
            logger.error(f"Error getting suggestions for '{partial_query}': {e}")
            return []
    
    def get_index_stats(self) -> IndexStats:
        """Get statistics about the search index."""
        try:
            # Initialize database if it doesn't exist
            if not self.db_path.exists():
                self._init_database()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count books
                cursor.execute('SELECT COUNT(*) FROM books')
                total_books = cursor.fetchone()[0]
                
                # Count chapters
                cursor.execute('SELECT COUNT(*) FROM chapters')
                total_chapters = cursor.fetchone()[0]
                
                # Count unique terms
                cursor.execute('SELECT COUNT(*) FROM terms')
                unique_terms = cursor.fetchone()[0]
                
                # Count total words
                cursor.execute('SELECT SUM(frequency) FROM terms')
                total_words = cursor.fetchone()[0] or 0
                
                # Count formulas
                cursor.execute('SELECT COUNT(*) FROM formulas')
                total_formulas = cursor.fetchone()[0]
                
                # Get index size
                index_size_mb = self.db_path.stat().st_size / (1024 * 1024)
                
                # Get last updated time
                cursor.execute('SELECT MAX(indexed_at) FROM books')
                last_updated = cursor.fetchone()[0] or 0
                
                return IndexStats(
                    total_books=total_books,
                    total_chapters=total_chapters,
                    total_words=total_words,
                    total_formulas=total_formulas,
                    unique_terms=unique_terms,
                    index_size_mb=index_size_mb,
                    last_updated=last_updated
                )
                
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return IndexStats(0, 0, 0, 0, 0, 0.0, 0.0)
    
    def rebuild_index(self) -> bool:
        """Rebuild the entire search index from scratch."""
        try:
            logger.info("Rebuilding search index from scratch")
            
            # Remove existing database
            if self.db_path.exists():
                self.db_path.unlink()
            
            # Reinitialize database
            self._init_database()
            
            logger.info("Search index rebuilt successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False
    
    def _init_database(self) -> None:
        """Initialize SQLite database for search index."""
        try:
            # Ensure directory exists with proper permissions
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            # Create database file if it doesn't exist
            if not self.db_path.exists():
                self.db_path.touch()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Books table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS books (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors TEXT,
                        source_path TEXT NOT NULL,
                        format_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        indexed_at REAL NOT NULL
                    )
                ''')
                
                # Chapters table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chapters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        book_id TEXT NOT NULL,
                        chapter_number TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        page_number INTEGER,
                        FOREIGN KEY (book_id) REFERENCES books (id)
                    )
                ''')
                
                # Terms table for full-text search
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS terms (
                        term TEXT NOT NULL,
                        book_id TEXT NOT NULL,
                        chapter_id INTEGER NOT NULL,
                        frequency INTEGER NOT NULL,
                        tf_idf REAL NOT NULL,
                        PRIMARY KEY (term, book_id, chapter_id),
                        FOREIGN KEY (book_id) REFERENCES books (id),
                        FOREIGN KEY (chapter_id) REFERENCES chapters (id)
                    )
                ''')
                
                # Formulas table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS formulas (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        book_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        formula_type TEXT,
                        context TEXT,
                        FOREIGN KEY (book_id) REFERENCES books (id)
                    )
                ''')
                
                # Create indexes for performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_terms_term ON terms (term)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_terms_book ON terms (book_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_chapters_book ON chapters (book_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_formulas_book ON formulas (book_id)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _index_chapter(self, cursor: sqlite3.Cursor, book_id: str, chapter: Dict[str, Any]) -> None:
        """Index a single chapter."""
        # Insert chapter record
        cursor.execute('''
            INSERT INTO chapters 
            (book_id, chapter_number, title, content, page_number)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            book_id,
            chapter.get('number', ''),
            chapter.get('title', ''),
            chapter.get('content', ''),
            chapter.get('page_start')
        ))
        
        chapter_id = cursor.lastrowid
        
        # Extract and index terms from chapter content
        content = chapter.get('content', '')
        terms = self._extract_terms(content)
        
        # Calculate term frequencies
        term_freq = Counter(terms)
        total_terms = len(terms)
        
        for term, freq in term_freq.items():
            if term not in self.stopwords and len(term) > 2:
                tf = freq / total_terms if total_terms > 0 else 0
                
                # Insert term (IDF will be calculated later)
                cursor.execute('''
                    INSERT OR REPLACE INTO terms 
                    (term, book_id, chapter_id, frequency, tf_idf)
                    VALUES (?, ?, ?, ?, ?)
                ''', (term, book_id, chapter_id, freq, tf))
    
    def _index_formula(self, cursor: sqlite3.Cursor, book_id: str, formula: Dict[str, str]) -> None:
        """Index a mathematical formula."""
        cursor.execute('''
            INSERT INTO formulas 
            (book_id, content, formula_type, context)
            VALUES (?, ?, ?, ?)
        ''', (
            book_id,
            formula.get('content', ''),
            formula.get('type', 'formula'),
            formula.get('context', '')
        ))
    
    def _build_terms_index(self, cursor: sqlite3.Cursor, book_id: str, content: ExtractedContent) -> None:
        """Build and update TF-IDF scores for terms."""
        # Get total number of documents
        cursor.execute('SELECT COUNT(DISTINCT book_id) FROM terms')
        total_docs = cursor.fetchone()[0]
        
        if total_docs == 0:
            return
        
        # Update TF-IDF scores for all terms in this book
        cursor.execute('SELECT DISTINCT term FROM terms WHERE book_id = ?', (book_id,))
        book_terms = [row[0] for row in cursor.fetchall()]
        
        for term in book_terms:
            # Calculate IDF
            cursor.execute('SELECT COUNT(DISTINCT book_id) FROM terms WHERE term = ?', (term,))
            docs_with_term = cursor.fetchone()[0]
            
            if docs_with_term > 0:
                idf = math.log(total_docs / docs_with_term)
                
                # Update TF-IDF scores
                cursor.execute('''
                    UPDATE terms 
                    SET tf_idf = (frequency * ?) 
                    WHERE term = ? AND book_id = ?
                ''', (idf, term, book_id))
    
    def _search_text(self, conn: sqlite3.Connection, query: str, max_results: int) -> List[SearchResult]:
        """Search text content using TF-IDF scoring."""
        cursor = conn.cursor()
        query_terms = self._extract_terms(query)
        
        if not query_terms:
            return []
        
        results = []
        
        # Search for each term
        for term in query_terms:
            cursor.execute('''
                SELECT b.id, b.title, c.chapter_number, c.title, c.content, 
                       t.tf_idf, b.source_path, c.page_number
                FROM terms t
                JOIN books b ON t.book_id = b.id
                JOIN chapters c ON t.chapter_id = c.id
                WHERE t.term LIKE ?
                ORDER BY t.tf_idf DESC
                LIMIT ?
            ''', (f"%{term}%", max_results))
            
            for row in cursor.fetchall():
                book_id, book_title, chapter_num, chapter_title, content, tf_idf, source_path, page_num = row
                
                # Create snippet around the match
                snippet = self._create_snippet(content, term)
                
                result = SearchResult(
                    book_id=book_id,
                    book_title=book_title,
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    content_snippet=snippet,
                    relevance_score=tf_idf,
                    match_type='text',
                    source_path=source_path,
                    page_number=page_num
                )
                results.append(result)
        
        return results
    
    def _search_formulas(self, conn: sqlite3.Connection, query: str, max_results: int) -> List[SearchResult]:
        """Search mathematical formulas."""
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT b.id, b.title, f.content, f.context, b.source_path
            FROM formulas f
            JOIN books b ON f.book_id = b.id
            WHERE f.content LIKE ? OR f.context LIKE ?
            LIMIT ?
        ''', (f"%{query}%", f"%{query}%", max_results))
        
        results = []
        for row in cursor.fetchall():
            book_id, book_title, formula_content, context, source_path = row
            
            result = SearchResult(
                book_id=book_id,
                book_title=book_title,
                chapter_number="",
                chapter_title="Mathematical Formula",
                content_snippet=f"Formula: {formula_content[:200]}...",
                relevance_score=1.0,  # Simple relevance for formulas
                match_type='formula',
                source_path=source_path
            )
            results.append(result)
        
        return results
    
    def _search_titles(self, conn: sqlite3.Connection, query: str, max_results: int) -> List[SearchResult]:
        """Search book and chapter titles."""
        cursor = conn.cursor()
        
        # Search book titles
        cursor.execute('''
            SELECT id, title, source_path
            FROM books
            WHERE title LIKE ?
            LIMIT ?
        ''', (f"%{query}%", max_results))
        
        results = []
        for row in cursor.fetchall():
            book_id, title, source_path = row
            
            result = SearchResult(
                book_id=book_id,
                book_title=title,
                chapter_number="",
                chapter_title="Book Title Match",
                content_snippet=f"Book: {title}",
                relevance_score=2.0,  # Higher relevance for title matches
                match_type='title',
                source_path=source_path
            )
            results.append(result)
        
        # Search chapter titles
        cursor.execute('''
            SELECT b.id, b.title, c.chapter_number, c.title, b.source_path, c.page_number
            FROM chapters c
            JOIN books b ON c.book_id = b.id
            WHERE c.title LIKE ?
            LIMIT ?
        ''', (f"%{query}%", max_results))
        
        for row in cursor.fetchall():
            book_id, book_title, chapter_num, chapter_title, source_path, page_num = row
            
            result = SearchResult(
                book_id=book_id,
                book_title=book_title,
                chapter_number=chapter_num,
                chapter_title=chapter_title,
                content_snippet=f"Chapter: {chapter_title}",
                relevance_score=1.5,  # High relevance for chapter title matches
                match_type='title',
                source_path=source_path,
                page_number=page_num
            )
            results.append(result)
        
        return results
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract searchable terms from text."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter out short words and stopwords
        terms = [word for word in words 
                if len(word) > 2 and word not in self.stopwords]
        
        return terms
    
    def _create_snippet(self, content: str, term: str, snippet_length: int = 200) -> str:
        """Create a content snippet around a search term."""
        term_pos = content.lower().find(term.lower())
        if term_pos == -1:
            return content[:snippet_length] + "..."
        
        # Get context around the term
        start = max(0, term_pos - snippet_length // 2)
        end = min(len(content), term_pos + len(term) + snippet_length // 2)
        
        snippet = content[start:end]
        
        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results."""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create a key based on book_id and chapter_number
            key = (result.book_id, result.chapter_number)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords."""
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'how', 'other',
            'many', 'some', 'time', 'very', 'when', 'much', 'can', 'would',
            'there', 'could', 'see', 'him', 'two', 'more', 'go', 'no', 'way',
            'may', 'say', 'come', 'his', 'been', 'now', 'find', 'long', 'down',
            'day', 'did', 'get', 'come', 'made', 'new', 'also', 'any', 'after',
            'back', 'other', 'well', 'where', 'just', 'first', 'over', 'think',
            'than', 'only', 'work', 'life', 'into', 'year', 'state', 'never',
            'become', 'between', 'high', 'really', 'something', 'most', 'another',
            'much', 'family', 'own', 'out', 'leave', 'put', 'old', 'while',
            'mean', 'keep', 'student', 'why', 'let', 'great', 'same', 'big',
            'group', 'begin', 'seem', 'country', 'help', 'talk', 'turn', 'ask',
            'show', 'try', 'during', 'without', 'again', 'place', 'right',
            'move', 'too', 'here', 'off', 'need', 'give', 'different', 'away',
            'follow', 'around', 'three', 'small', 'set', 'every', 'large',
            'must', 'before', 'change', 'does', 'part'
        }
        return stopwords