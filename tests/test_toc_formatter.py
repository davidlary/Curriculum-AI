"""
Unit tests for TOC Formatter module.

Goal: Test table of contents formatting and file output functionality
for TSV and JSON formats with proper data integrity.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path

from core.toc_formatter import TOCFormatter
from core.toc_extractor import BookTOC, TOCEntry


class TestTOCFormatter:
    """Test TOCFormatter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = TOCFormatter()
        
        # Create sample TOC for testing
        self.sample_entries = [
            TOCEntry(
                title="Chapter 1: Introduction",
                level=1,
                page_number=1,
                section_number="1",
                entry_id="ch1"
            ),
            TOCEntry(
                title="Section 1.1: Overview",
                level=2,
                page_number=5,
                section_number="1.1",
                entry_id="s1_1",
                parent_id="ch1"
            ),
            TOCEntry(
                title="Chapter 2: Methods",
                level=1,
                page_number=10,
                section_number="2",
                entry_id="ch2"
            )
        ]
        
        self.sample_toc = BookTOC(
            book_title="Test Physics Book",
            language="english",
            discipline="Physics",
            level="University",
            file_path="/path/to/test_book.pdf",
            entries=self.sample_entries,
            extraction_method="Test extraction",
            total_entries=3
        )
    
    def test_formatter_initialization(self):
        """Goal: Test TOC formatter initialization."""
        assert isinstance(self.formatter, TOCFormatter)
    
    def test_sanitize_filename(self):
        """Goal: Test filename sanitization for safe file system usage."""
        test_cases = [
            ("Normal Filename", "Normal Filename"),
            ("File<>Name", "File__Name"),
            ("File:Name", "File_Name"),
            ("File\"Name", "File_Name"),
            ("File/Name", "File_Name"),
            ("File\\Name", "File_Name"),
            ("File|Name", "File_Name"),
            ("File?Name", "File_Name"),
            ("File*Name", "File_Name"),
            ("", "untitled"),
            ("   ", "untitled"),
            ("A" * 150, "A" * 100),  # Long filename truncation
        ]
        
        for input_name, expected in test_cases:
            result = self.formatter._sanitize_filename(input_name)
            assert result == expected
    
    def test_save_toc_both_formats(self):
        """Goal: Test saving TOC in both TSV and JSON formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir)
            
            assert 'tsv' in saved_files
            assert 'json' in saved_files
            assert saved_files['tsv'].exists()
            assert saved_files['json'].exists()
            assert saved_files['tsv'].suffix == '.tsv'
            assert saved_files['json'].suffix == '.json'
    
    def test_save_toc_specific_format(self):
        """Goal: Test saving TOC in specific format only."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Test TSV only
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir, formats=['tsv'])
            assert 'tsv' in saved_files
            assert 'json' not in saved_files
            
            # Test JSON only
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir, formats=['json'])
            assert 'json' in saved_files
            assert 'tsv' not in saved_files
    
    def test_save_tsv_content(self):
        """Goal: Test TSV file content and structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir, formats=['tsv'])
            tsv_path = saved_files['tsv']
            
            # Read and verify TSV content
            with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t')
                rows = list(reader)
            
            # Check headers
            expected_headers = ['entry_id', 'title', 'level', 'section_number', 'page_number', 'parent_id']
            assert rows[0] == expected_headers
            
            # Check data rows
            assert len(rows) == 4  # Header + 3 entries
            
            # Check first entry
            assert rows[1][0] == "ch1"  # entry_id
            assert rows[1][1] == "Chapter 1: Introduction"  # title
            assert rows[1][2] == "1"  # level
            assert rows[1][3] == "1"  # section_number
            assert rows[1][4] == "1"  # page_number
            assert rows[1][5] == ""   # parent_id (empty)
    
    def test_save_json_content(self):
        """Goal: Test JSON file content and structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir, formats=['json'])
            json_path = saved_files['json']
            
            # Read and verify JSON content
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            # Check top-level structure
            assert data['book_title'] == "Test Physics Book"
            assert data['language'] == "english"
            assert data['discipline'] == "Physics"
            assert data['level'] == "University"
            assert data['total_entries'] == 3
            assert 'entries' in data
            assert 'format_info' in data
            
            # Check entries
            entries = data['entries']
            assert len(entries) == 3
            
            # Check first entry
            first_entry = entries[0]
            assert first_entry['title'] == "Chapter 1: Introduction"
            assert first_entry['level'] == 1
            assert first_entry['entry_id'] == "ch1"
            assert first_entry['page_number'] == 1
    
    def test_load_toc_json(self):
        """Goal: Test loading TOC from JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Save original TOC
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir, formats=['json'])
            json_path = saved_files['json']
            
            # Load TOC back
            loaded_toc = self.formatter.load_toc_json(json_path)
            
            # Verify loaded data matches original
            assert loaded_toc.book_title == self.sample_toc.book_title
            assert loaded_toc.language == self.sample_toc.language
            assert loaded_toc.discipline == self.sample_toc.discipline
            assert loaded_toc.level == self.sample_toc.level
            assert loaded_toc.total_entries == self.sample_toc.total_entries
            assert len(loaded_toc.entries) == len(self.sample_toc.entries)
            
            # Check first entry details
            original_entry = self.sample_toc.entries[0]
            loaded_entry = loaded_toc.entries[0]
            assert loaded_entry.title == original_entry.title
            assert loaded_entry.level == original_entry.level
            assert loaded_entry.entry_id == original_entry.entry_id
    
    def test_dict_to_toc_entry(self):
        """Goal: Test conversion from dictionary to TOCEntry object."""
        entry_dict = {
            'title': 'Test Chapter',
            'level': 2,
            'page_number': 15,
            'section_number': '2.1',
            'parent_id': 'ch2',
            'entry_id': 'test_entry',
            'children': []
        }
        
        entry = self.formatter._dict_to_toc_entry(entry_dict)
        
        assert entry.title == 'Test Chapter'
        assert entry.level == 2
        assert entry.page_number == 15
        assert entry.section_number == '2.1'
        assert entry.parent_id == 'ch2'
        assert entry.entry_id == 'test_entry'
        assert len(entry.children) == 0
    
    def test_dict_to_toc_entry_with_children(self):
        """Goal: Test conversion with nested children."""
        entry_dict = {
            'title': 'Parent Chapter',
            'level': 1,
            'entry_id': 'parent',
            'children': [
                {
                    'title': 'Child Section',
                    'level': 2,
                    'entry_id': 'child',
                    'parent_id': 'parent',
                    'children': []
                }
            ]
        }
        
        entry = self.formatter._dict_to_toc_entry(entry_dict)
        
        assert entry.title == 'Parent Chapter'
        assert len(entry.children) == 1
        assert entry.children[0].title == 'Child Section'
        assert entry.children[0].parent_id == 'parent'
    
    def test_create_combined_report(self):
        """Goal: Test creation of combined report from multiple TOCs."""
        # Create second sample TOC
        toc2 = BookTOC(
            book_title="Another Book",
            language="spanish",
            discipline="Mathematics",
            level="HighSchool",
            file_path="/path/to/book2.pdf",
            entries=[TOCEntry("Capítulo 1", 1, entry_id="cap1")],
            extraction_method="Test extraction",
            total_entries=1
        )
        
        tocs = [self.sample_toc, toc2]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            
            result_path = self.formatter.create_combined_report(tocs, report_path)
            assert result_path == report_path
            assert report_path.exists()
            
            # Read and verify report content
            with open(report_path, 'r', encoding='utf-8') as report_file:
                report = json.load(report_file)
            
            # Check summary statistics
            summary = report['summary']
            assert summary['total_books'] == 2
            assert summary['total_entries'] == 4  # 3 + 1
            assert set(summary['languages']) == {'english', 'spanish'}
            assert set(summary['disciplines']) == {'Physics', 'Mathematics'}
            assert set(summary['levels']) == {'University', 'HighSchool'}
            
            # Check individual book entries
            books = report['books']
            assert len(books) == 2
            
            # Books should be sorted by discipline, level, title
            assert books[0]['discipline'] in ['Mathematics', 'Physics']
            assert books[1]['discipline'] in ['Mathematics', 'Physics']
    
    def test_save_toc_creates_directories(self):
        """Goal: Test that save_toc creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use nested directory that doesn't exist
            output_dir = Path(temp_dir) / "nested" / "directory"
            assert not output_dir.exists()
            
            saved_files = self.formatter.save_toc(self.sample_toc, output_dir)
            
            # Directory should be created
            assert output_dir.exists()
            assert output_dir.is_dir()
            
            # Files should be saved
            assert saved_files['tsv'].exists()
            assert saved_files['json'].exists()
    
    def test_write_entry_tsv_recursive(self):
        """Goal: Test recursive writing of entries with children to TSV."""
        # Create entry with children
        parent = TOCEntry(
            title="Parent Chapter",
            level=1,
            entry_id="parent"
        )
        child1 = TOCEntry(
            title="Child Section 1",
            level=2,
            entry_id="child1",
            parent_id="parent"
        )
        child2 = TOCEntry(
            title="Child Section 2", 
            level=2,
            entry_id="child2",
            parent_id="parent"
        )
        parent.children = [child1, child2]
        
        # Create TOC with nested structure
        nested_toc = BookTOC(
            book_title="Nested Book",
            language="english",
            discipline="Physics",
            level="University",
            file_path="/path/to/nested.pdf",
            entries=[parent],
            total_entries=3
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            saved_files = self.formatter.save_toc(nested_toc, output_dir, formats=['tsv'])
            tsv_path = saved_files['tsv']
            
            # Read TSV content
            with open(tsv_path, 'r', newline='', encoding='utf-8') as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t')
                rows = list(reader)
            
            # Should have header + parent + 2 children = 4 rows
            assert len(rows) == 4
            
            # Check parent entry
            assert rows[1][0] == "parent"
            assert rows[1][1] == "Parent Chapter"
            
            # Check child entries
            assert rows[2][0] == "child1"
            assert rows[2][1] == "Child Section 1" 
            assert rows[2][5] == "parent"  # parent_id
            
            assert rows[3][0] == "child2"
            assert rows[3][1] == "Child Section 2"
            assert rows[3][5] == "parent"  # parent_id


if __name__ == '__main__':
    pytest.main([__file__])