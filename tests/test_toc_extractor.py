"""
Unit tests for TOC Extractor module.

Goal: Test table of contents extraction functionality across different file formats
and ensure proper handling of various content structures.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.toc_extractor import TOCExtractor, TOCEntry, BookTOC


class TestTOCEntry:
    """Test TOCEntry data class functionality."""
    
    def test_toc_entry_creation(self):
        """Goal: Test basic TOC entry creation and attributes."""
        entry = TOCEntry(
            title="Chapter 1: Introduction",
            level=1,
            page_number=10,
            section_number="1.1",
            entry_id="entry_001"
        )
        
        assert entry.title == "Chapter 1: Introduction"
        assert entry.level == 1
        assert entry.page_number == 10
        assert entry.section_number == "1.1"
        assert entry.entry_id == "entry_001"
        assert entry.children == []
    
    def test_toc_entry_to_dict(self):
        """Goal: Test conversion of TOC entry to dictionary."""
        entry = TOCEntry(
            title="Test Chapter",
            level=2,
            entry_id="test_001"
        )
        
        result = entry.to_dict()
        
        assert result['title'] == "Test Chapter"
        assert result['level'] == 2
        assert result['entry_id'] == "test_001"
        assert 'children' in result
        assert isinstance(result['children'], list)
    
    def test_toc_entry_with_children(self):
        """Goal: Test TOC entry with child entries."""
        parent = TOCEntry(title="Chapter 1", level=1, entry_id="ch1")
        child1 = TOCEntry(title="Section 1.1", level=2, entry_id="s1_1", parent_id="ch1")
        child2 = TOCEntry(title="Section 1.2", level=2, entry_id="s1_2", parent_id="ch1")
        
        parent.children = [child1, child2]
        
        assert len(parent.children) == 2
        assert parent.children[0].parent_id == "ch1"
        assert parent.children[1].parent_id == "ch1"


class TestBookTOC:
    """Test BookTOC data class functionality."""
    
    def test_book_toc_creation(self):
        """Goal: Test basic BookTOC creation and attributes."""
        entries = [
            TOCEntry(title="Chapter 1", level=1, entry_id="ch1"),
            TOCEntry(title="Chapter 2", level=1, entry_id="ch2")
        ]
        
        toc = BookTOC(
            book_title="Test Book",
            language="english",
            discipline="Physics",
            level="University",
            file_path="/path/to/book.pdf",
            entries=entries,
            extraction_method="PDF extraction",
            total_entries=2
        )
        
        assert toc.book_title == "Test Book"
        assert toc.language == "english"
        assert toc.discipline == "Physics"
        assert toc.level == "University"
        assert len(toc.entries) == 2
        assert toc.total_entries == 2
    
    def test_book_toc_to_dict(self):
        """Goal: Test conversion of BookTOC to dictionary."""
        toc = BookTOC(
            book_title="Test Book",
            language="english",
            discipline="Physics",
            level="University",
            file_path="/path/to/book.pdf"
        )
        
        result = toc.to_dict()
        
        assert result['book_title'] == "Test Book"
        assert result['language'] == "english"
        assert result['discipline'] == "Physics"
        assert 'entries' in result
        assert isinstance(result['entries'], list)


class TestTOCExtractor:
    """Test TOCExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TOCExtractor()
    
    def test_extractor_initialization(self):
        """Goal: Test TOC extractor initialization."""
        assert hasattr(self.extractor, 'supported_formats')
        assert '.pdf' in self.extractor.supported_formats
        assert '.xml' in self.extractor.supported_formats
        assert '.md' in self.extractor.supported_formats
    
    def test_extract_toc_nonexistent_file(self):
        """Goal: Test handling of non-existent files."""
        fake_path = Path("/nonexistent/file.pdf")
        result = self.extractor.extract_toc(fake_path, "english", "Physics", "University")
        
        assert result is None
    
    def test_extract_toc_unsupported_format(self):
        """Goal: Test handling of unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            
        try:
            result = self.extractor.extract_toc(temp_path, "english", "Physics", "University")
            assert result is None
        finally:
            temp_path.unlink()
    
    def test_extract_markdown_toc(self):
        """Goal: Test extraction from Markdown files."""
        markdown_content = """# Chapter 1: Introduction
        
## Section 1.1: Overview

### Subsection 1.1.1: Details

## Section 1.2: Conclusion

# Chapter 2: Advanced Topics

## Section 2.1: Theory
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(markdown_content)
            temp_path = Path(temp_file.name)
        
        try:
            toc = self.extractor.extract_toc(temp_path, "english", "Physics", "University")
            
            assert toc is not None
            assert len(toc.entries) == 6  # All headings extracted
            assert toc.entries[0].title == "Chapter 1: Introduction"
            assert toc.entries[0].level == 1
            assert toc.entries[1].title == "Section 1.1: Overview"
            assert toc.entries[1].level == 2
            
        finally:
            temp_path.unlink()
    
    def test_extract_xml_toc(self):
        """Goal: Test extraction from XML files."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <title>Main Document Title</title>
    <chapter>
        <title>Chapter 1</title>
        <section>
            <title>Section 1.1</title>
        </section>
    </chapter>
    <chapter>
        <title>Chapter 2</title>
    </chapter>
</document>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(xml_content)
            temp_path = Path(temp_file.name)
        
        try:
            toc = self.extractor.extract_toc(temp_path, "english", "Physics", "University")
            
            assert toc is not None
            assert len(toc.entries) > 0
            
            # Check that titles are extracted
            titles = [entry.title for entry in toc.entries]
            assert "Main Document Title" in titles
            assert "Chapter 1" in titles
            assert "Chapter 2" in titles
            
        finally:
            temp_path.unlink()
    
    def test_extract_latex_toc(self):
        """Goal: Test extraction from LaTeX files."""
        latex_content = r"""
\documentclass{book}
\begin{document}

\chapter{Introduction to Physics}
\section{Classical Mechanics}
\subsection{Newton's Laws}
\subsubsection{First Law}

\section{Thermodynamics}

\chapter{Modern Physics}
\section{Quantum Mechanics}

\end{document}
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(latex_content)
            temp_path = Path(temp_file.name)
        
        try:
            toc = self.extractor.extract_toc(temp_path, "english", "Physics", "University")
            
            assert toc is not None
            assert len(toc.entries) > 0
            
            # Check that LaTeX sections are extracted
            titles = [entry.title for entry in toc.entries]
            assert "Introduction to Physics" in titles
            assert "Classical Mechanics" in titles
            assert "Newton's Laws" in titles
            
        finally:
            temp_path.unlink()
    
    def test_get_xml_element_level(self):
        """Goal: Test XML element level determination."""
        assert self.extractor._get_xml_element_level('title', '') == 1
        assert self.extractor._get_xml_element_level('h1', '') == 1
        assert self.extractor._get_xml_element_level('h2', '') == 2
        assert self.extractor._get_xml_element_level('section', '') == 2
        assert self.extractor._get_xml_element_level('subsection', '') == 3
    
    def test_get_rst_heading_level(self):
        """Goal: Test reStructuredText heading level mapping."""
        assert self.extractor._get_rst_heading_level('=') == 1
        assert self.extractor._get_rst_heading_level('-') == 2
        assert self.extractor._get_rst_heading_level('~') == 3
        assert self.extractor._get_rst_heading_level('^') == 4
        assert self.extractor._get_rst_heading_level('"') == 5
        assert self.extractor._get_rst_heading_level('?') == 3  # Default
    
    @patch('core.toc_extractor.PyPDF2')
    def test_extract_pdf_with_outline(self, mock_pypdf2):
        """Goal: Test PDF extraction with outline/bookmarks."""
        # Mock PDF reader with outline
        mock_outline_item = MagicMock()
        mock_outline_item.title = "Chapter 1"
        
        mock_reader = MagicMock()
        mock_reader.outline = [mock_outline_item]
        mock_reader.pages = []
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Test the _extract_pdf_toc method directly
            entries = self.extractor._extract_pdf_toc(temp_path)
            
            assert len(entries) > 0
            assert entries[0].title == "Chapter 1"
            
        finally:
            temp_path.unlink()
    
    def test_extract_rst_toc(self):
        """Goal: Test extraction from reStructuredText files."""
        rst_content = """
Introduction
============

This is the introduction.

Chapter 1
---------

This is chapter 1.

Section 1.1
~~~~~~~~~~~

This is a subsection.

Chapter 2
---------

This is chapter 2.
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rst', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(rst_content)
            temp_path = Path(temp_file.name)
        
        try:
            toc = self.extractor.extract_toc(temp_path, "english", "Physics", "University")
            
            assert toc is not None
            assert len(toc.entries) > 0
            
            # Check that RST headings are extracted with correct levels
            titles_and_levels = [(entry.title, entry.level) for entry in toc.entries]
            
            assert ("Introduction", 1) in titles_and_levels
            assert ("Chapter 1", 2) in titles_and_levels
            assert ("Section 1.1", 3) in titles_and_levels
            assert ("Chapter 2", 2) in titles_and_levels
            
        finally:
            temp_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__])