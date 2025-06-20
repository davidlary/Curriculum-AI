"""
Unit tests for core.content_processor module
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
import json
from pathlib import Path
import shutil
from datetime import datetime

from core.content_processor import ContentProcessor
from core.config import OpenBooksConfig


class TestContentProcessor(unittest.TestCase):
    """Test cases for ContentProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.config.metadata_path = os.path.join(self.temp_dir, 'metadata')
        self.processor = ContentProcessor(self.config)
        
        # Create test collection data
        self.test_collection = {
            'pdfs': [
                {
                    'name': 'College Physics.pdf',
                    'path': '/books/physics/college-physics.pdf',
                    'size_mb': 50.5
                },
                {
                    'name': 'Biology Textbook.pdf', 
                    'path': '/books/biology/biology.pdf',
                    'size_mb': 75.2
                }
            ],
            'git_repos': [
                {
                    'name': 'osbooks-astronomy',
                    'path': '/books/astronomy/osbooks-astronomy'
                },
                {
                    'name': 'cnxbook-physics',
                    'path': '/books/physics/cnxbook-physics'
                }
            ],
            'other_formats': [
                {
                    'name': 'chemistry-textbook',
                    'path': '/books/chemistry/chemistry-textbook'
                }
            ],
            'total_size_mb': 125.7
        }

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_processor_initialization(self):
        """Test ContentProcessor initialization"""
        self.assertIsInstance(self.processor.config, OpenBooksConfig)
        self.assertTrue(os.path.exists(self.processor.metadata_path))

    def test_extract_subject_from_filename(self):
        """Test subject extraction from filenames"""
        test_cases = [
            ('College Physics.pdf', 'Physics'),
            ('Biology Textbook.pdf', 'Biology'),
            ('Organic Chemistry.pdf', 'Chemistry'),
            ('Calculus Bundle.pdf', 'Mathematics'),
            ('Introduction to Psychology.pdf', 'Psychology'),
            ('Unknown Subject.pdf', 'Other')
        ]
        
        for filename, expected_subject in test_cases:
            subject = self.processor._extract_subject_from_filename(filename)
            self.assertEqual(subject, expected_subject)

    def test_classify_repository_type(self):
        """Test repository type classification"""
        test_cases = [
            ('cnxbook-physics', 'CNX Book'),
            ('osbooks-astronomy', 'OpenStax Book'),
            ('derived-from-osbooks-chemistry', 'OpenStax Derived'),
            ('university-physics-volume-1', 'University Physics'),
            ('random-repository', 'Other Repository')
        ]
        
        for repo_name, expected_type in test_cases:
            repo_type = self.processor._classify_repository_type(repo_name)
            self.assertEqual(repo_type, expected_type)

    def test_classify_directory_type(self):
        """Test directory type classification"""
        test_cases = [
            ('philschatz-physics-book', 'GitBook Format'),
            ('textversions-cache', 'Processed Content'),
            ('unknown-format', 'Unknown Format')
        ]
        
        for dir_name, expected_type in test_cases:
            dir_type = self.processor._classify_directory_type(dir_name)
            self.assertEqual(dir_type, expected_type)

    def test_generate_catalog_markdown(self):
        """Test Markdown catalog generation"""
        with patch.object(self.config, 'get_known_repositories', return_value=[]):
            markdown = self.processor.generate_catalog_markdown(self.test_collection)
            
            self.assertIsInstance(markdown, str)
            self.assertIn('# OpenBooks Collection Catalog', markdown)
            self.assertIn('College Physics.pdf', markdown)
            self.assertIn('osbooks-astronomy', markdown)
            self.assertIn('Total PDFs**: 2', markdown)
            self.assertIn('Git Repositories**: 2', markdown)

    def test_generate_json_catalog(self):
        """Test JSON catalog generation"""
        with patch.object(self.config, 'get_known_repositories', return_value=[]):
            catalog = self.processor.generate_json_catalog(self.test_collection)
            
            self.assertIsInstance(catalog, dict)
            self.assertIn('metadata', catalog)
            self.assertIn('collections', catalog)
            self.assertIn('subjects', catalog)
            
            # Check metadata
            self.assertEqual(catalog['metadata']['total_books'], 5)
            self.assertEqual(catalog['metadata']['total_size_mb'], 125.7)
            
            # Check collections
            self.assertEqual(len(catalog['collections']['pdfs']), 2)
            self.assertEqual(len(catalog['collections']['git_repositories']), 2)
            self.assertEqual(len(catalog['collections']['other_formats']), 1)

    def test_calculate_file_hash(self):
        """Test file hash calculation"""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('Hello World')
        
        file_hash = self.processor._calculate_file_hash(test_file)
        
        self.assertIsInstance(file_hash, str)
        self.assertEqual(len(file_hash), 16)  # Short hash

    def test_calculate_file_hash_error(self):
        """Test file hash calculation with non-existent file"""
        nonexistent_file = '/path/that/does/not/exist.txt'
        
        file_hash = self.processor._calculate_file_hash(nonexistent_file)
        self.assertEqual(file_hash, "unknown")

    def test_get_last_modified(self):
        """Test getting last modified timestamp"""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        timestamp = self.processor._get_last_modified(test_file)
        
        self.assertIsInstance(timestamp, str)
        # Should be a valid ISO format timestamp
        datetime.fromisoformat(timestamp)

    def test_get_last_modified_error(self):
        """Test getting last modified timestamp with error"""
        nonexistent_file = '/path/that/does/not/exist.txt'
        
        timestamp = self.processor._get_last_modified(nonexistent_file)
        
        self.assertIsInstance(timestamp, str)
        # Should still return a valid timestamp (current time)
        datetime.fromisoformat(timestamp)

    def test_save_catalog(self):
        """Test saving catalog to files"""
        with patch.object(self.config, 'get_known_repositories', return_value=[]):
            self.processor.save_catalog(self.test_collection)
            
            # Check that files were created
            markdown_path = self.processor.metadata_path / "CATALOG.md"
            json_path = self.processor.metadata_path / "catalog.json"
            
            self.assertTrue(markdown_path.exists())
            self.assertTrue(json_path.exists())
            
            # Check content
            with open(markdown_path, 'r') as f:
                markdown_content = f.read()
                self.assertIn('# OpenBooks Collection Catalog', markdown_content)
            
            with open(json_path, 'r') as f:
                json_content = json.load(f)
                self.assertIn('metadata', json_content)

    def test_extract_book_metadata_pdf(self):
        """Test extracting metadata from PDF file"""
        # Create test PDF file
        test_pdf = Path(self.temp_dir) / 'test.pdf'
        test_pdf.write_bytes(b'%PDF-1.4\nfake pdf content')
        
        metadata = self.processor.extract_book_metadata(str(test_pdf))
        
        self.assertEqual(metadata['name'], 'test')
        self.assertEqual(metadata['type'], 'PDF')
        self.assertEqual(metadata['format'], 'PDF')
        self.assertGreater(metadata['size_mb'], 0)

    def test_extract_book_metadata_git_repo(self):
        """Test extracting metadata from Git repository"""
        # Create test Git repository
        test_repo = Path(self.temp_dir) / 'osbooks-astronomy'
        test_repo.mkdir()
        (test_repo / '.git').mkdir()
        (test_repo / 'content.txt').write_text('repository content')
        
        metadata = self.processor.extract_book_metadata(str(test_repo))
        
        self.assertEqual(metadata['name'], 'osbooks-astronomy')
        self.assertEqual(metadata['type'], 'OpenStax Book')
        self.assertEqual(metadata['format'], 'Git Repository')
        self.assertGreater(metadata['size_mb'], 0)
        self.assertIn('file_count', metadata)

    def test_extract_book_metadata_directory(self):
        """Test extracting metadata from general directory"""
        # Create test directory
        test_dir = Path(self.temp_dir) / 'test-directory'
        test_dir.mkdir()
        (test_dir / 'file1.txt').write_text('content 1')
        (test_dir / 'file2.txt').write_text('content 2')
        
        metadata = self.processor.extract_book_metadata(str(test_dir))
        
        self.assertEqual(metadata['name'], 'test-directory')
        self.assertEqual(metadata['format'], 'Directory')
        self.assertGreater(metadata['size_mb'], 0)
        self.assertEqual(metadata['file_count'], 2)

    def test_extract_pdf_metadata_with_pymupdf(self):
        """Test PDF metadata extraction with PyMuPDF"""
        test_pdf = Path(self.temp_dir) / 'test.pdf'
        test_pdf.write_bytes(b'%PDF-1.4\nfake pdf content')
        
        # Mock PyMuPDF
        mock_doc = Mock()
        mock_doc.page_count = 100
        mock_doc.metadata = {
            'title': 'Test PDF Title',
            'author': 'Test Author',
            'creator': 'Test Creator',
            'subject': 'Test Subject'
        }
        
        with patch('core.content_processor.fitz') as mock_fitz:
            mock_fitz.open.return_value.__enter__.return_value = mock_doc
            
            metadata = self.processor._extract_pdf_metadata(test_pdf)
            
            self.assertEqual(metadata['pages'], 100)
            self.assertEqual(metadata['title'], 'Test PDF Title')
            self.assertEqual(metadata['author'], 'Test Author')

    def test_extract_pdf_metadata_without_pymupdf(self):
        """Test PDF metadata extraction without PyMuPDF"""
        test_pdf = Path(self.temp_dir) / 'test.pdf'
        test_pdf.write_bytes(b'%PDF-1.4\nfake pdf content')
        
        with patch('core.content_processor.fitz', side_effect=ImportError):
            metadata = self.processor._extract_pdf_metadata(test_pdf)
            
            # Should still return basic metadata
            self.assertEqual(metadata['type'], 'PDF')
            self.assertEqual(metadata['format'], 'PDF')
            self.assertNotIn('pages', metadata)

    def test_extract_git_metadata(self):
        """Test Git repository metadata extraction"""
        test_repo = Path(self.temp_dir) / 'osbooks-physics'
        test_repo.mkdir()
        
        # Create some test files
        (test_repo / 'file1.txt').write_text('content 1' * 100)
        (test_repo / 'file2.txt').write_text('content 2' * 200)
        
        metadata = self.processor._extract_git_metadata(test_repo)
        
        self.assertEqual(metadata['type'], 'OpenStax Book')
        self.assertEqual(metadata['format'], 'Git Repository')
        self.assertEqual(metadata['subject'], 'Physics')
        self.assertGreater(metadata['size_mb'], 0)
        self.assertEqual(metadata['file_count'], 2)

    def test_extract_directory_metadata(self):
        """Test general directory metadata extraction"""
        test_dir = Path(self.temp_dir) / 'chemistry-textbook'
        test_dir.mkdir()
        
        # Create some test files
        (test_dir / 'chapter1.txt').write_text('chapter 1 content')
        (test_dir / 'chapter2.txt').write_text('chapter 2 content')
        
        metadata = self.processor._extract_directory_metadata(test_dir)
        
        self.assertEqual(metadata['format'], 'Directory')
        self.assertEqual(metadata['subject'], 'Chemistry')
        self.assertGreater(metadata['size_mb'], 0)
        self.assertEqual(metadata['file_count'], 2)

    def test_generate_catalog_with_known_repositories(self):
        """Test catalog generation with known repositories"""
        known_repos = [
            {
                'name': 'College Physics',
                'repo': 'osbooks-college-physics',
                'org': 'openstax',
                'subject': 'Physics'
            },
            {
                'name': 'Introduction to Biology',
                'repo': 'osbooks-biology',
                'org': 'openstax',
                'subject': 'Biology'
            }
        ]
        
        with patch.object(self.config, 'get_known_repositories', return_value=known_repos):
            markdown = self.processor.generate_catalog_markdown(self.test_collection)
            
            self.assertIn('Known OpenStax Repositories', markdown)
            self.assertIn('College Physics', markdown)
            self.assertIn('osbooks-college-physics', markdown)
            self.assertIn('Biology', markdown)

    def test_json_catalog_subjects_grouping(self):
        """Test that JSON catalog properly groups items by subject"""
        with patch.object(self.config, 'get_known_repositories', return_value=[]):
            catalog = self.processor.generate_json_catalog(self.test_collection)
            
            subjects = catalog['subjects']
            
            # Should have Physics, Biology, and Chemistry subjects
            self.assertIn('Physics', subjects)
            self.assertIn('Biology', subjects)
            
            # Physics should have multiple items
            physics_items = subjects['Physics']
            self.assertGreater(len(physics_items), 0)

    def test_markdown_catalog_structure(self):
        """Test Markdown catalog structure and content"""
        with patch.object(self.config, 'get_known_repositories', return_value=[]):
            markdown = self.processor.generate_catalog_markdown(self.test_collection)
            
            # Check for required sections
            required_sections = [
                '# OpenBooks Collection Catalog',
                '## Collection Summary',
                '## PDF Books',
                '## Git Repositories',
                '## Other Formats',
                '## Collection Guidelines',
                '## Usage Instructions'
            ]
            
            for section in required_sections:
                self.assertIn(section, markdown)
            
            # Check for table headers
            self.assertIn('| Book Title | Size (MB) | Subject | Status |', markdown)
            self.assertIn('| Repository Name | Type | Subject | Status |', markdown)


if __name__ == '__main__':
    unittest.main()