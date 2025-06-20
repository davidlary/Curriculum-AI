"""
Unit tests for core.search_indexer module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sqlite3
from pathlib import Path
import shutil

from core.search_indexer import SearchIndexer, SearchResult, IndexStats
from core.config import OpenBooksConfig
from core.text_extractor import ExtractedContent


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test SearchResult creation"""
        result = SearchResult(
            book_id="book123",
            book_title="Test Book",
            chapter_number="1",
            chapter_title="Introduction",
            content_snippet="This is a test snippet...",
            relevance_score=0.75,
            match_type="text",
            source_path="/books/test.pdf",
            page_number=10
        )
        
        self.assertEqual(result.book_id, "book123")
        self.assertEqual(result.book_title, "Test Book")
        self.assertEqual(result.chapter_number, "1")
        self.assertEqual(result.relevance_score, 0.75)
        self.assertEqual(result.match_type, "text")
        self.assertEqual(result.page_number, 10)


class TestIndexStats(unittest.TestCase):
    """Test cases for IndexStats dataclass"""

    def test_index_stats_creation(self):
        """Test IndexStats creation"""
        stats = IndexStats(
            total_books=10,
            total_chapters=50,
            total_words=10000,
            total_formulas=25,
            unique_terms=2500,
            index_size_mb=5.5,
            last_updated=1234567890.0
        )
        
        self.assertEqual(stats.total_books, 10)
        self.assertEqual(stats.total_chapters, 50)
        self.assertEqual(stats.total_words, 10000)
        self.assertEqual(stats.total_formulas, 25)
        self.assertEqual(stats.unique_terms, 2500)
        self.assertEqual(stats.index_size_mb, 5.5)


class TestSearchIndexer(unittest.TestCase):
    """Test cases for SearchIndexer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.config.search_index_dir = os.path.join(self.temp_dir, 'search_index')
        self.indexer = SearchIndexer(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_indexer_initialization(self):
        """Test SearchIndexer initialization"""
        self.assertIsInstance(self.indexer.config, OpenBooksConfig)
        self.assertTrue(self.indexer.index_dir.exists())
        self.assertTrue(self.indexer.db_path.exists())
        self.assertIsInstance(self.indexer.stopwords, set)
        self.assertGreater(len(self.indexer.stopwords), 0)

    def test_database_initialization(self):
        """Test database initialization"""
        # Check that tables exist
        with sqlite3.connect(self.indexer.db_path) as conn:
            cursor = conn.cursor()
            
            # Check books table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='books'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check chapters table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chapters'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check terms table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='terms'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check formulas table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='formulas'")
            self.assertIsNotNone(cursor.fetchone())

    def test_index_content(self):
        """Test indexing extracted content"""
        # Create test content
        content = ExtractedContent(
            title="Test Physics Book",
            authors=["Dr. Test"],
            source_path="/books/physics.pdf",
            format_type="PDF",
            content_hash="testhash123",
            chapters=[
                {
                    "number": "1",
                    "title": "Introduction to Physics",
                    "content": "Physics is the science of matter and energy.",
                    "page_start": 1
                },
                {
                    "number": "2", 
                    "title": "Mechanics",
                    "content": "Mechanics deals with motion and forces.",
                    "page_start": 10
                }
            ],
            mathematical_notation=[
                {
                    "content": "F = ma",
                    "type": "equation",
                    "context": "Newton's second law"
                }
            ]
        )
        
        success = self.indexer.index_content(content)
        
        self.assertTrue(success)
        
        # Verify content was indexed
        with sqlite3.connect(self.indexer.db_path) as conn:
            cursor = conn.cursor()
            
            # Check book was inserted
            cursor.execute("SELECT COUNT(*) FROM books WHERE title = ?", (content.title,))
            self.assertEqual(cursor.fetchone()[0], 1)
            
            # Check chapters were inserted
            cursor.execute("SELECT COUNT(*) FROM chapters WHERE book_id = ?", (content.content_hash,))
            self.assertEqual(cursor.fetchone()[0], 2)
            
            # Check formulas were inserted
            cursor.execute("SELECT COUNT(*) FROM formulas WHERE book_id = ?", (content.content_hash,))
            self.assertEqual(cursor.fetchone()[0], 1)

    def test_search_text(self):
        """Test text search functionality"""
        # Index test content first
        content = ExtractedContent(
            title="Physics Textbook",
            authors=["Author"],
            source_path="/books/physics.pdf",
            format_type="PDF",
            content_hash="physics123",
            chapters=[
                {
                    "number": "1",
                    "title": "Force and Motion",
                    "content": "Force is a vector quantity that causes acceleration in objects.",
                    "page_start": 1
                }
            ],
            mathematical_notation=[]
        )
        
        self.indexer.index_content(content)
        
        # Search for "force"
        results = self.indexer.search("force", max_results=10, search_type="text")
        
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result.book_title, "Physics Textbook")
        self.assertEqual(result.match_type, "text")
        self.assertIn("force", result.content_snippet.lower())

    def test_search_formulas(self):
        """Test formula search functionality"""
        # Index content with formulas
        content = ExtractedContent(
            title="Math Book",
            authors=["Author"],
            source_path="/books/math.pdf", 
            format_type="PDF",
            content_hash="math123",
            chapters=[],
            mathematical_notation=[
                {
                    "content": "E = mc²",
                    "type": "equation",
                    "context": "Einstein's mass-energy equivalence"
                },
                {
                    "content": "F = ma",
                    "type": "equation", 
                    "context": "Newton's second law"
                }
            ]
        )
        
        self.indexer.index_content(content)
        
        # Search for formulas
        results = self.indexer.search("E = mc", max_results=10, search_type="formula")
        
        self.assertGreater(len(results), 0)
        result = results[0]
        self.assertEqual(result.match_type, "formula")
        self.assertIn("E = mc", result.content_snippet)

    def test_search_titles(self):
        """Test title search functionality"""
        # Index content
        content = ExtractedContent(
            title="Quantum Mechanics",
            authors=["Author"],
            source_path="/books/quantum.pdf",
            format_type="PDF", 
            content_hash="quantum123",
            chapters=[
                {
                    "number": "1",
                    "title": "Wave-Particle Duality",
                    "content": "Light exhibits both wave and particle properties.",
                    "page_start": 1
                }
            ],
            mathematical_notation=[]
        )
        
        self.indexer.index_content(content)
        
        # Search book titles
        results = self.indexer.search("quantum", max_results=10, search_type="title")
        
        self.assertGreater(len(results), 0)
        
        # Should find both book title and chapter title matches
        book_title_match = next((r for r in results if r.chapter_title == "Book Title Match"), None)
        self.assertIsNotNone(book_title_match)
        self.assertEqual(book_title_match.book_title, "Quantum Mechanics")

    def test_search_empty_query(self):
        """Test search with empty query"""
        results = self.indexer.search("", max_results=10)
        
        self.assertEqual(len(results), 0)

    def test_search_no_results(self):
        """Test search with query that has no matches"""
        results = self.indexer.search("nonexistentterm12345", max_results=10)
        
        self.assertEqual(len(results), 0)

    def test_get_suggestions(self):
        """Test autocomplete suggestions"""
        # Index content with terms
        content = ExtractedContent(
            title="Physics Book",
            authors=["Author"],
            source_path="/books/physics.pdf",
            format_type="PDF",
            content_hash="physics456", 
            chapters=[
                {
                    "number": "1",
                    "title": "Physics Fundamentals",
                    "content": "Physics studies physical phenomena. Physicists analyze physical properties.",
                    "page_start": 1
                }
            ],
            mathematical_notation=[]
        )
        
        self.indexer.index_content(content)
        
        # Get suggestions for "phy"
        suggestions = self.indexer.get_suggestions("phy", max_suggestions=5)
        
        self.assertGreater(len(suggestions), 0)
        # Should suggest words starting with "phy"
        for suggestion in suggestions:
            self.assertTrue(suggestion.startswith("phy"))

    def test_get_suggestions_short_query(self):
        """Test suggestions with too short query"""
        suggestions = self.indexer.get_suggestions("a", max_suggestions=5)
        
        self.assertEqual(len(suggestions), 0)

    def test_get_index_stats(self):
        """Test getting index statistics"""
        # Index some content
        content = ExtractedContent(
            title="Test Book",
            authors=["Author"],
            source_path="/books/test.pdf",
            format_type="PDF",
            content_hash="test789",
            chapters=[
                {
                    "number": "1",
                    "title": "Chapter One",
                    "content": "This is test content for statistics.",
                    "page_start": 1
                }
            ],
            mathematical_notation=[
                {
                    "content": "x = y + z",
                    "type": "equation",
                    "context": "Simple addition"
                }
            ]
        )
        
        self.indexer.index_content(content)
        
        stats = self.indexer.get_index_stats()
        
        self.assertIsInstance(stats, IndexStats)
        self.assertEqual(stats.total_books, 1)
        self.assertEqual(stats.total_chapters, 1)
        self.assertEqual(stats.total_formulas, 1)
        self.assertGreater(stats.unique_terms, 0)
        self.assertGreater(stats.index_size_mb, 0)

    def test_rebuild_index(self):
        """Test rebuilding index"""
        # Index some content first
        content = ExtractedContent(
            title="Original Book",
            authors=["Author"],
            source_path="/books/original.pdf",
            format_type="PDF",
            content_hash="original123",
            chapters=[{"number": "1", "title": "Chapter", "content": "Content", "page_start": 1}],
            mathematical_notation=[]
        )
        
        self.indexer.index_content(content)
        
        # Verify content exists
        stats_before = self.indexer.get_index_stats()
        self.assertEqual(stats_before.total_books, 1)
        
        # Rebuild index
        success = self.indexer.rebuild_index()
        
        self.assertTrue(success)
        
        # Verify index is empty after rebuild
        stats_after = self.indexer.get_index_stats()
        self.assertEqual(stats_after.total_books, 0)

    def test_extract_terms(self):
        """Test term extraction from text"""
        text = "This is a test sentence with various words and punctuation!"
        
        terms = self.indexer._extract_terms(text)
        
        self.assertIsInstance(terms, list)
        # Should filter out stopwords and short words
        self.assertNotIn("is", terms)  # stopword
        self.assertNotIn("a", terms)   # short word
        self.assertIn("test", terms)   # valid term
        self.assertIn("sentence", terms)  # valid term

    def test_create_snippet(self):
        """Test content snippet creation"""
        content = "This is a long piece of text that contains the search term physics in the middle of the sentence and continues with more text."
        term = "physics"
        
        snippet = self.indexer._create_snippet(content, term, snippet_length=50)
        
        self.assertIn(term, snippet)
        self.assertLessEqual(len(snippet), 60)  # Should be around snippet_length + ellipsis

    def test_create_snippet_term_not_found(self):
        """Test snippet creation when term is not found"""
        content = "This is some text without the search term."
        term = "nonexistent"
        
        snippet = self.indexer._create_snippet(content, term, snippet_length=20)
        
        self.assertEqual(len(snippet), 23)  # 20 chars + "..."
        self.assertTrue(snippet.endswith("..."))

    def test_deduplicate_results(self):
        """Test result deduplication"""
        results = [
            SearchResult("book1", "Book 1", "1", "Chapter 1", "snippet", 1.0, "text", "/path1"),
            SearchResult("book1", "Book 1", "1", "Chapter 1", "different snippet", 0.8, "text", "/path1"),
            SearchResult("book1", "Book 1", "2", "Chapter 2", "snippet", 0.9, "text", "/path1"),
        ]
        
        unique_results = self.indexer._deduplicate_results(results)
        
        # Should have 2 unique results (book1,chapter1) and (book1,chapter2)
        self.assertEqual(len(unique_results), 2)
        
        # Check that we kept the right results
        chapter_numbers = [r.chapter_number for r in unique_results]
        self.assertIn("1", chapter_numbers)
        self.assertIn("2", chapter_numbers)

    def test_load_stopwords(self):
        """Test stopwords loading"""
        stopwords = self.indexer._load_stopwords()
        
        self.assertIsInstance(stopwords, set)
        self.assertGreater(len(stopwords), 0)
        
        # Check for common stopwords
        self.assertIn("the", stopwords)
        self.assertIn("and", stopwords)
        self.assertIn("is", stopwords)

    def test_index_chapter(self):
        """Test chapter indexing"""
        with sqlite3.connect(self.indexer.db_path) as conn:
            cursor = conn.cursor()
            
            # First insert a book
            book_id = "testbook123"
            cursor.execute('''
                INSERT INTO books (id, title, authors, source_path, format_type, content_hash, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (book_id, "Test Book", "[]", "/test", "PDF", "hash", 12345))
            
            # Test chapter
            chapter = {
                "number": "1",
                "title": "Test Chapter",
                "content": "This is test content with terms for indexing.",
                "page_start": 10
            }
            
            self.indexer._index_chapter(cursor, book_id, chapter)
            conn.commit()
            
            # Verify chapter was inserted
            cursor.execute("SELECT COUNT(*) FROM chapters WHERE book_id = ?", (book_id,))
            self.assertEqual(cursor.fetchone()[0], 1)
            
            # Verify terms were indexed
            cursor.execute("SELECT COUNT(*) FROM terms WHERE book_id = ?", (book_id,))
            term_count = cursor.fetchone()[0]
            self.assertGreater(term_count, 0)

    def test_index_formula(self):
        """Test formula indexing"""
        with sqlite3.connect(self.indexer.db_path) as conn:
            cursor = conn.cursor()
            
            book_id = "testbook456"
            formula = {
                "content": "E = mc²",
                "type": "equation",
                "context": "Mass-energy equivalence"
            }
            
            self.indexer._index_formula(cursor, book_id, formula)
            conn.commit()
            
            # Verify formula was inserted
            cursor.execute("SELECT COUNT(*) FROM formulas WHERE book_id = ?", (book_id,))
            self.assertEqual(cursor.fetchone()[0], 1)
            
            # Verify formula content
            cursor.execute("SELECT content FROM formulas WHERE book_id = ?", (book_id,))
            stored_content = cursor.fetchone()[0]
            self.assertEqual(stored_content, "E = mc²")


if __name__ == '__main__':
    unittest.main()