"""
Unit tests for core.text_extractor module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
import json
from pathlib import Path
import shutil

from core.text_extractor import TextExtractor, ExtractedContent, Chapter
from core.config import OpenBooksConfig


class TestExtractedContent(unittest.TestCase):
    """Test cases for ExtractedContent dataclass"""

    def test_extracted_content_creation(self):
        """Test ExtractedContent creation"""
        content = ExtractedContent(
            source_path="/books/test.pdf",
            format_type="pdf",
            title="Test Book",
            authors=["Dr. Test"],
            chapters=[],
            raw_text="Test content",
            mathematical_notation=[],
            images=[],
            metadata={},
            extraction_stats={},
            content_hash="abc123"
        )
        
        self.assertEqual(content.source_path, "/books/test.pdf")
        self.assertEqual(content.format_type, "pdf")
        self.assertEqual(content.title, "Test Book")
        self.assertEqual(content.authors, ["Dr. Test"])
        self.assertEqual(content.raw_text, "Test content")
        self.assertEqual(content.content_hash, "abc123")


class TestChapter(unittest.TestCase):
    """Test cases for Chapter dataclass"""

    def test_chapter_creation(self):
        """Test Chapter creation"""
        chapter = Chapter(
            number="1",
            title="Introduction",
            content="Chapter content",
            subsections=[],
            formulas=[],
            figures=[],
            exercises=[]
        )
        
        self.assertEqual(chapter.number, "1")
        self.assertEqual(chapter.title, "Introduction")
        self.assertEqual(chapter.content, "Chapter content")
        self.assertEqual(len(chapter.subsections), 0)


class TestTextExtractor(unittest.TestCase):
    """Test cases for TextExtractor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.extractor = TextExtractor(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_extractor_initialization(self):
        """Test TextExtractor initialization"""
        self.assertIsInstance(self.extractor.config, OpenBooksConfig)
        self.assertEqual(len(self.extractor.extraction_cache), 0)
        self.assertTrue(self.extractor.xml_available)

    def test_detect_format_pdf(self):
        """Test format detection for PDF files"""
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        format_type = self.extractor._detect_format(pdf_path)
        self.assertEqual(format_type, "pdf")

    def test_detect_format_epub(self):
        """Test format detection for EPUB files"""
        epub_path = Path(self.temp_dir) / "test.epub"
        epub_path.touch()
        
        format_type = self.extractor._detect_format(epub_path)
        self.assertEqual(format_type, "epub")

    def test_detect_format_xml(self):
        """Test format detection for XML files"""
        xml_path = Path(self.temp_dir) / "test.xml"
        xml_path.touch()
        
        format_type = self.extractor._detect_format(xml_path)
        self.assertEqual(format_type, "xml")

    def test_detect_format_cnxml_directory(self):
        """Test format detection for CNXML directories"""
        cnxml_dir = Path(self.temp_dir) / "cnxml-book"
        cnxml_dir.mkdir()
        (cnxml_dir / "collections").mkdir()
        
        format_type = self.extractor._detect_format(cnxml_dir)
        self.assertEqual(format_type, "cnxml")

    def test_detect_format_git_repository(self):
        """Test format detection for git repositories"""
        repo_dir = Path(self.temp_dir) / "git-repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        (repo_dir / "media").mkdir()
        
        format_type = self.extractor._detect_format(repo_dir)
        self.assertEqual(format_type, "cnxml")

    def test_detect_format_unknown(self):
        """Test format detection for unknown files"""
        unknown_path = Path(self.temp_dir) / "test.unknown"
        unknown_path.touch()
        
        format_type = self.extractor._detect_format(unknown_path)
        self.assertEqual(format_type, "unknown")

    def test_extract_content_caching(self):
        """Test content extraction caching"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        # Mock the extraction method
        with patch.object(self.extractor, '_extract_markdown') as mock_extract:
            mock_content = ExtractedContent(
                source_path=str(test_file),
                format_type="markdown",
                title="Test",
                authors=[],
                chapters=[],
                raw_text="Test content",
                mathematical_notation=[],
                images=[],
                metadata={},
                extraction_stats={},
                content_hash="test123"
            )
            mock_extract.return_value = mock_content
            
            # First call should use extractor
            content1 = self.extractor.extract_content(str(test_file))
            self.assertEqual(mock_extract.call_count, 1)
            
            # Second call should use cache
            content2 = self.extractor.extract_content(str(test_file))
            self.assertEqual(mock_extract.call_count, 1)  # Should not increase
            
            self.assertEqual(content1, content2)

    def test_extract_content_unsupported_format(self):
        """Test extraction with unsupported format"""
        with patch.object(self.extractor, '_detect_format', return_value='unsupported'):
            result = self.extractor.extract_content("/fake/path")
            self.assertIsNone(result)

    def test_extract_formulas(self):
        """Test formula extraction from text"""
        text = """
        This is text with formulas:
        $E = mc^2$ and $$F = ma$$
        Also: \\begin{equation}x = y + z\\end{equation}
        """
        
        formulas = self.extractor._extract_formulas(text)
        
        self.assertGreater(len(formulas), 0)
        self.assertIn("E = mc^2", formulas)
        self.assertIn("F = ma", formulas)

    def test_extract_math_from_text(self):
        """Test mathematical notation extraction with context"""
        text = "The famous equation $E = mc^2$ shows mass-energy equivalence."
        
        math_elements = self.extractor._extract_math_from_text(text)
        
        self.assertGreater(len(math_elements), 0)
        element = math_elements[0]
        self.assertIn('content', element)
        self.assertIn('type', element)
        self.assertIn('context', element)
        self.assertEqual(element['type'], 'formula')

    def test_extract_latex_math(self):
        """Test LaTeX mathematical expression extraction"""
        latex_content = """
        \\begin{equation}
        x = y + z
        \\end{equation}
        
        \\begin{align}
        a &= b \\\\
        c &= d
        \\end{align}
        
        $$E = mc^2$$
        """
        
        formulas = self.extractor._extract_latex_math(latex_content)
        
        self.assertGreater(len(formulas), 0)
        # Should find equations from different environments

    def test_clean_latex_text(self):
        """Test LaTeX text cleaning"""
        latex_content = """
        \\chapter{Introduction}
        This is \\textbf{bold} text with \\emph{emphasis}.
        \\begin{itemize}
        \\item First item
        \\item Second item
        \\end{itemize}
        """
        
        cleaned = self.extractor._clean_latex_text(latex_content)
        
        self.assertNotIn("\\chapter", cleaned)
        self.assertNotIn("\\textbf", cleaned)
        self.assertNotIn("\\begin", cleaned)
        self.assertIn("Introduction", cleaned)
        self.assertIn("bold", cleaned)

    def test_extract_chapter_title(self):
        """Test chapter title extraction"""
        text = """
        Chapter 1: Introduction to Physics
        
        This chapter covers basic concepts.
        """
        
        title = self.extractor._extract_chapter_title(text)
        self.assertEqual(title, "Introduction to Physics")

    def test_extract_chapter_title_no_match(self):
        """Test chapter title extraction with no match"""
        text = "This is just regular text without chapter headings."
        
        title = self.extractor._extract_chapter_title(text)
        self.assertEqual(title, "Untitled Chapter")

    def test_get_formula_context(self):
        """Test formula context extraction"""
        text = "This is text before the formula E = mc^2 and this is text after."
        formula = "E = mc^2"
        
        context = self.extractor._get_formula_context(text, formula)
        
        self.assertIn("before", context)
        self.assertIn("after", context)
        self.assertIn(formula, context)

    def test_get_formula_context_not_found(self):
        """Test formula context when formula not found"""
        text = "This text doesn't contain the formula."
        formula = "E = mc^2"
        
        context = self.extractor._get_formula_context(text, formula)
        self.assertEqual(context, "")

    def test_get_cache_key(self):
        """Test cache key generation"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("content")
        
        key1 = self.extractor._get_cache_key(test_file)
        key2 = self.extractor._get_cache_key(test_file)
        
        self.assertEqual(key1, key2)  # Should be consistent
        self.assertIsInstance(key1, str)
        self.assertEqual(len(key1), 32)  # MD5 hash length

    def test_get_cache_key_nonexistent_file(self):
        """Test cache key generation for non-existent file"""
        nonexistent = Path("/nonexistent/file.txt")
        
        key = self.extractor._get_cache_key(nonexistent)
        
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 32)

    @patch('core.text_extractor.fitz')
    def test_extract_pdf_available(self, mock_fitz):
        """Test PDF extraction when PyMuPDF is available"""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.page_count = 2
        mock_doc.metadata = {
            'title': 'Test PDF',
            'author': 'Test Author'
        }
        
        # Mock pages
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Chapter 1: Introduction\nThis is the first chapter."
        mock_page1.get_images.return_value = []
        
        mock_page2 = Mock()
        mock_page2.get_text.return_value = "More content in chapter 1."
        mock_page2.get_images.return_value = []
        
        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz.open.return_value = mock_doc
        
        # Set PDF available
        self.extractor.pdf_available = True
        
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        content = self.extractor._extract_pdf(pdf_path)
        
        self.assertIsNotNone(content)
        self.assertEqual(content.format_type, "pdf")
        self.assertEqual(content.title, "Test PDF")
        self.assertIn("Introduction", content.raw_text)

    def test_extract_pdf_not_available(self):
        """Test PDF extraction when PyMuPDF is not available"""
        self.extractor.pdf_available = False
        
        pdf_path = Path(self.temp_dir) / "test.pdf"
        pdf_path.touch()
        
        content = self.extractor._extract_pdf(pdf_path)
        
        self.assertIsNone(content)

    def test_extract_xml(self):
        """Test XML content extraction"""
        xml_content = """<?xml version="1.0"?>
        <book>
            <title>Test Book</title>
            <chapter>
                <title>Chapter 1</title>
                <content>This is chapter content.</content>
            </chapter>
        </book>"""
        
        xml_path = Path(self.temp_dir) / "test.xml"
        xml_path.write_text(xml_content)
        
        content = self.extractor._extract_xml(xml_path)
        
        self.assertIsNotNone(content)
        self.assertEqual(content.format_type, "xml")
        self.assertEqual(content.title, "test")
        self.assertIn("Test Book", content.raw_text)
        self.assertIn("Chapter 1", content.raw_text)

    def test_extract_markdown(self):
        """Test Markdown content extraction"""
        md_content = """# Test Book

## Chapter 1: Introduction

This is *italic* and **bold** text.

- List item 1
- List item 2

```python
print("Hello World")
```
"""
        
        md_path = Path(self.temp_dir) / "test.md"
        md_path.write_text(md_content)
        
        content = self.extractor._extract_markdown(md_path)
        
        self.assertIsNotNone(content)
        self.assertEqual(content.format_type, "markdown")
        self.assertEqual(content.title, "test")
        self.assertIn("Test Book", content.raw_text)
        self.assertIn("Introduction", content.raw_text)

    def test_extract_latex(self):
        """Test LaTeX content extraction"""
        latex_content = """\\documentclass{article}
\\title{Test Document}
\\author{Test Author}

\\begin{document}
\\maketitle

\\section{Introduction}
This is the introduction.

\\begin{equation}
E = mc^2
\\end{equation}

\\end{document}"""
        
        latex_path = Path(self.temp_dir) / "test.tex"
        latex_path.write_text(latex_content)
        
        content = self.extractor._extract_latex(latex_path)
        
        self.assertIsNotNone(content)
        self.assertEqual(content.format_type, "latex")
        self.assertIn("Introduction", content.raw_text)

    def test_extract_html_with_beautifulsoup(self):
        """Test HTML extraction with BeautifulSoup available"""
        html_content = """<!DOCTYPE html>
<html>
<head><title>Test HTML</title></head>
<body>
    <h1>Test Book</h1>
    <p>This is a paragraph with <strong>bold</strong> text.</p>
    <script>alert('script');</script>
</body>
</html>"""
        
        html_path = Path(self.temp_dir) / "test.html"
        html_path.write_text(html_content)
        
        with patch('core.text_extractor.BeautifulSoup') as mock_bs:
            mock_soup = Mock()
            mock_soup.get_text.return_value = "Test Book This is a paragraph with bold text."
            mock_bs.return_value = mock_soup
            
            content = self.extractor._extract_html(html_path)
            
            self.assertIsNotNone(content)
            self.assertEqual(content.format_type, "html")
            self.assertIn("Test Book", content.raw_text)

    def test_extract_html_without_beautifulsoup(self):
        """Test HTML extraction without BeautifulSoup"""
        html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>"
        
        html_path = Path(self.temp_dir) / "test.html"
        html_path.write_text(html_content)
        
        with patch('core.text_extractor.BeautifulSoup', side_effect=ImportError):
            content = self.extractor._extract_html(html_path)
            
            self.assertIsNotNone(content)
            self.assertEqual(content.format_type, "html")
            # Should use regex fallback to remove HTML tags
            self.assertNotIn("<html>", content.raw_text)
            self.assertIn("Test", content.raw_text)

    def test_save_extracted_content(self):
        """Test saving extracted content to file"""
        content = ExtractedContent(
            source_path="/test/path.pdf",
            format_type="pdf",
            title="Test Book",
            authors=["Author"],
            chapters=[],
            raw_text="Test content",
            mathematical_notation=[],
            images=[],
            metadata={},
            extraction_stats={},
            content_hash="test123"
        )
        
        output_dir = Path(self.temp_dir) / "output"
        
        output_file = self.extractor.save_extracted_content(content, output_dir)
        
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.name, "test123.json")
        
        # Verify content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['title'], "Test Book")
        self.assertEqual(saved_data['format_type'], "pdf")
        self.assertIn('extracted_at', saved_data)

    def test_extract_cnxml_directory_no_collection(self):
        """Test CNXML extraction with no collection file"""
        cnxml_dir = Path(self.temp_dir) / "cnxml-book"
        cnxml_dir.mkdir()
        (cnxml_dir / "collections").mkdir()
        
        content = self.extractor._extract_cnxml_directory(cnxml_dir)
        
        self.assertIsNone(content)

    @patch('core.text_extractor.ET.parse')
    def test_extract_cnxml_directory_with_collection(self, mock_parse):
        """Test CNXML extraction with collection file"""
        # Create directory structure
        cnxml_dir = Path(self.temp_dir) / "cnxml-book"
        cnxml_dir.mkdir()
        collections_dir = cnxml_dir / "collections"
        collections_dir.mkdir()
        collection_file = collections_dir / "collection.xml"
        collection_file.touch()
        
        # Mock XML parsing
        mock_root = Mock()
        mock_title = Mock()
        mock_title.text = "Test Collection"
        mock_root.find.return_value = mock_title
        mock_root.findall.return_value = []  # No authors or chapters
        
        mock_tree = Mock()
        mock_tree.getroot.return_value = mock_root
        mock_parse.return_value = mock_tree
        
        content = self.extractor._extract_cnxml_directory(cnxml_dir)
        
        self.assertIsNotNone(content)
        self.assertEqual(content.format_type, "cnxml")
        self.assertEqual(content.title, "Test Collection")

    def test_extract_cnxml_module_not_found(self):
        """Test CNXML module extraction when file not found"""
        cnxml_dir = Path(self.temp_dir) / "cnxml-book"
        cnxml_dir.mkdir()
        
        result = self.extractor._extract_cnxml_module(cnxml_dir, "nonexistent")
        
        self.assertIsNone(result)

    @patch('core.text_extractor.ET.parse')
    def test_extract_cnxml_module_success(self, mock_parse):
        """Test successful CNXML module extraction"""
        cnxml_dir = Path(self.temp_dir) / "cnxml-book"
        cnxml_dir.mkdir()
        module_file = cnxml_dir / "module1.cnxml"
        module_file.touch()
        
        # Mock XML parsing
        mock_root = Mock()
        mock_tree = Mock()
        mock_tree.getroot.return_value = mock_root
        mock_parse.return_value = mock_tree
        
        with patch('core.text_extractor.ET.tostring', return_value="Module content"):
            content = self.extractor._extract_cnxml_module(cnxml_dir, "module1")
            
            self.assertEqual(content, "Module content")


if __name__ == '__main__':
    unittest.main()