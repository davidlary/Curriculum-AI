"""
Content processing module for textbook analysis and catalog generation.

This module processes textbook content from various formats and generates
comprehensive catalogs and metadata.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from .config import OpenBooksConfig

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Processes textbook content and generates catalogs."""
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize with configuration."""
        self.config = config
        self.metadata_path = Path(config.metadata_path)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
    
    def generate_catalog_markdown(self, existing_collection: Dict[str, Any]) -> str:
        """
        Generate comprehensive catalog in Markdown format.
        
        Args:
            existing_collection: Dictionary of existing books
            
        Returns:
            Markdown content for catalog
        """
        logger.info("Generating catalog markdown")
        
        # Header
        catalog = []
        catalog.append("# OpenBooks Collection Catalog")
        catalog.append("")
        catalog.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        catalog.append("")
        catalog.append("This catalog provides a comprehensive overview of the OpenBooks collection,")
        catalog.append("including PDFs, Git repositories, and other textbook formats.")
        catalog.append("")
        
        # Summary statistics
        catalog.append("## Collection Summary")
        catalog.append("")
        catalog.append(f"- **Total PDFs**: {len(existing_collection['pdfs'])}")
        catalog.append(f"- **Git Repositories**: {len(existing_collection['git_repos'])}")
        catalog.append(f"- **Other Formats**: {len(existing_collection['other_formats'])}")
        catalog.append(f"- **Total Size**: {existing_collection['total_size_mb']:.1f} MB")
        catalog.append("")
        
        # PDF Books
        if existing_collection['pdfs']:
            catalog.append("## PDF Books")
            catalog.append("")
            catalog.append("| Book Title | Size (MB) | Subject | Status |")
            catalog.append("|------------|-----------|---------|--------|")
            
            for pdf in sorted(existing_collection['pdfs'], key=lambda x: x['name']):
                subject = self._extract_subject_from_filename(pdf['name'])
                status = "✅ Available"
                catalog.append(f"| {pdf['name']} | {pdf['size_mb']} | {subject} | {status} |")
            
            catalog.append("")
        
        # Git Repositories
        if existing_collection['git_repos']:
            catalog.append("## Git Repositories")
            catalog.append("")
            catalog.append("These are machine-readable textbook repositories, typically from OpenStax CNX.")
            catalog.append("")
            catalog.append("| Repository Name | Type | Subject | Status |")
            catalog.append("|-----------------|------|---------|--------|")
            
            for repo in sorted(existing_collection['git_repos'], key=lambda x: x['name']):
                repo_type = self._classify_repository_type(repo['name'])
                subject = self._extract_subject_from_filename(repo['name'])
                status = "✅ Cloned"
                catalog.append(f"| {repo['name']} | {repo_type} | {subject} | {status} |")
            
            catalog.append("")
        
        # Other Formats
        if existing_collection['other_formats']:
            catalog.append("## Other Formats")
            catalog.append("")
            catalog.append("| Directory Name | Type | Subject | Status |")
            catalog.append("|----------------|------|---------|--------|")
            
            for item in sorted(existing_collection['other_formats'], key=lambda x: x['name']):
                item_type = self._classify_directory_type(item['name'])
                subject = self._extract_subject_from_filename(item['name'])
                status = "📁 Directory"
                catalog.append(f"| {item['name']} | {item_type} | {subject} | {status} |")
            
            catalog.append("")
        
        # Collection Guidelines
        catalog.append("## Collection Guidelines")
        catalog.append("")
        catalog.append("### Source Priority")
        catalog.append("1. **Existing Collection**: Check for already acquired books")
        catalog.append("2. **Git Repositories**: Prefer machine-readable OpenStax CNX repositories")
        catalog.append("3. **Structured Formats**: EPUB, XML, LaTeX sources")
        catalog.append("4. **PDF Downloads**: Last resort for content acquisition")
        catalog.append("")
        
        catalog.append("### Quality Standards")
        catalog.append("- Minimum book size: 1 MB")
        catalog.append("- Maximum book size: 500 MB")
        catalog.append("- Prefer open licenses (Creative Commons)")
        catalog.append("- Maintain original formatting and mathematical notation")
        catalog.append("")
        
        # Known OpenStax Repositories
        catalog.append("## Known OpenStax Repositories")
        catalog.append("")
        catalog.append("This section lists known OpenStax textbook repositories available for cloning.")
        catalog.append("")
        
        known_repos = self.config.get_known_repositories()
        if known_repos:
            # Group by subject
            subjects = {}
            for repo in known_repos:
                subject = repo.get('subject', 'Other')
                if subject not in subjects:
                    subjects[subject] = []
                subjects[subject].append(repo)
            
            for subject in sorted(subjects.keys()):
                catalog.append(f"### {subject}")
                catalog.append("")
                catalog.append("| Book Title | Repository | Organization |")
                catalog.append("|------------|------------|--------------|")
                
                for repo in sorted(subjects[subject], key=lambda x: x['name']):
                    repo_link = f"[{repo['repo']}](https://github.com/{repo['org']}/{repo['repo']})"
                    catalog.append(f"| {repo['name']} | {repo_link} | {repo['org']} |")
                
                catalog.append("")
        
        # Usage Instructions
        catalog.append("## Usage Instructions")
        catalog.append("")
        catalog.append("### Acquiring New Books")
        catalog.append("```bash")
        catalog.append("# Get all available books")
        catalog.append("python GetOpenBooks.py")
        catalog.append("")
        catalog.append("# Focus on specific subjects")
        catalog.append("python GetOpenBooks.py --subjects Physics,Biology")
        catalog.append("")
        catalog.append("# Prefer git repositories")
        catalog.append("python GetOpenBooks.py --format git")
        catalog.append("")
        catalog.append("# Dry run to see what would be acquired")
        catalog.append("python GetOpenBooks.py --dry-run")
        catalog.append("```")
        catalog.append("")
        
        catalog.append("### Updating Existing Collection")
        catalog.append("```bash")
        catalog.append("# Update all git repositories")
        catalog.append("python GetOpenBooks.py --update-existing")
        catalog.append("")
        catalog.append("# Update with verbose output")
        catalog.append("python GetOpenBooks.py --update-existing --verbose")
        catalog.append("```")
        catalog.append("")
        
        # Footer
        catalog.append("---")
        catalog.append("")
        catalog.append("**Note**: This catalog is automatically generated by the OpenBooks system.")
        catalog.append("For the most up-to-date information, run the catalog generation again.")
        catalog.append("")
        catalog.append(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(catalog)
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        """Extract subject from filename using patterns."""
        filename_lower = filename.lower()
        
        subject_patterns = {
            'Physics': ['physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics'],
            'Biology': ['biology', 'anatomy', 'physiology', 'microbiology', 'life'],
            'Chemistry': ['chemistry', 'organic', 'biochemistry'],
            'Mathematics': ['math', 'calculus', 'algebra', 'trigonometry', 'precalculus'],
            'Statistics': ['statistics', 'statistical', 'probability'],
            'Economics': ['economics', 'microeconomics', 'macroeconomics'],
            'Psychology': ['psychology', 'psychological'],
            'Sociology': ['sociology', 'social'],
            'Computer Science': ['computer', 'programming', 'software']
        }
        
        for subject, patterns in subject_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return subject
        
        return 'Other'
    
    def _classify_repository_type(self, repo_name: str) -> str:
        """Classify repository type based on name patterns."""
        name_lower = repo_name.lower()
        
        if name_lower.startswith('cnxbook-'):
            return 'CNX Book'
        elif name_lower.startswith('osbooks-'):
            return 'OpenStax Book'
        elif name_lower.startswith('derived-from-osbooks-'):
            return 'OpenStax Derived'
        elif 'university-physics' in name_lower:
            return 'University Physics'
        else:
            return 'Other Repository'
    
    def _classify_directory_type(self, dir_name: str) -> str:
        """Classify directory type based on name and contents."""
        name_lower = dir_name.lower()
        
        if 'philschatz' in name_lower:
            return 'GitBook Format'
        elif any(x in name_lower for x in ['textversions', 'cache']):
            return 'Processed Content'
        else:
            return 'Unknown Format'
    
    def generate_json_catalog(self, existing_collection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate catalog in JSON format for programmatic access.
        
        Args:
            existing_collection: Dictionary of existing books
            
        Returns:
            JSON-serializable catalog dictionary
        """
        logger.info("Generating JSON catalog")
        
        catalog = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'OpenBooks v0.1.0',
                'total_books': (len(existing_collection['pdfs']) + 
                              len(existing_collection['git_repos']) + 
                              len(existing_collection['other_formats'])),
                'total_size_mb': existing_collection['total_size_mb']
            },
            'collections': {
                'pdfs': [],
                'git_repositories': [],
                'other_formats': []
            },
            'subjects': {},
            'known_repositories': self.config.get_known_repositories()
        }
        
        # Process PDFs
        for pdf in existing_collection['pdfs']:
            pdf_info = {
                'name': pdf['name'],
                'path': pdf['path'],
                'size_mb': pdf['size_mb'],
                'subject': self._extract_subject_from_filename(pdf['name']),
                'type': 'PDF',
                'hash': self._calculate_file_hash(pdf['path'])
            }
            catalog['collections']['pdfs'].append(pdf_info)
        
        # Process Git repositories
        for repo in existing_collection['git_repos']:
            repo_info = {
                'name': repo['name'],
                'path': repo['path'],
                'subject': self._extract_subject_from_filename(repo['name']),
                'type': self._classify_repository_type(repo['name']),
                'last_modified': self._get_last_modified(repo['path'])
            }
            catalog['collections']['git_repositories'].append(repo_info)
        
        # Process other formats
        for item in existing_collection['other_formats']:
            item_info = {
                'name': item['name'],
                'path': item['path'],
                'subject': self._extract_subject_from_filename(item['name']),
                'type': self._classify_directory_type(item['name']),
                'last_modified': self._get_last_modified(item['path'])
            }
            catalog['collections']['other_formats'].append(item_info)
        
        # Group by subjects
        all_items = (catalog['collections']['pdfs'] + 
                    catalog['collections']['git_repositories'] + 
                    catalog['collections']['other_formats'])
        
        for item in all_items:
            subject = item['subject']
            if subject not in catalog['subjects']:
                catalog['subjects'][subject] = []
            catalog['subjects'][subject].append({
                'name': item['name'],
                'type': item['type'],
                'path': item['path']
            })
        
        return catalog
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()[:16]  # Short hash
        except Exception as e:
            logger.warning(f"Error calculating hash for {file_path}: {e}")
            return "unknown"
    
    def _get_last_modified(self, path: str) -> str:
        """Get last modified timestamp for a path."""
        try:
            stat = os.stat(path)
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception as e:
            logger.warning(f"Error getting last modified for {path}: {e}")
            return datetime.now().isoformat()
    
    def save_catalog(self, existing_collection: Dict[str, Any]) -> None:
        """
        Save catalog in both Markdown and JSON formats.
        
        Args:
            existing_collection: Dictionary of existing books
        """
        logger.info("Saving catalog files")
        
        # Generate and save Markdown catalog
        markdown_content = self.generate_catalog_markdown(existing_collection)
        markdown_path = self.metadata_path / "CATALOG.md"
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Markdown catalog saved: {markdown_path}")
        
        # Generate and save JSON catalog
        json_catalog = self.generate_json_catalog(existing_collection)
        json_path = self.metadata_path / "catalog.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_catalog, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON catalog saved: {json_path}")
    
    def extract_book_metadata(self, book_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a book (PDF or repository).
        
        Args:
            book_path: Path to book file or directory
            
        Returns:
            Dictionary with book metadata
        """
        path = Path(book_path)
        
        metadata = {
            'name': path.stem if path.is_file() else path.name,
            'path': str(path),
            'type': 'unknown',
            'size_mb': 0,
            'subject': 'Other',
            'last_modified': self._get_last_modified(str(path)),
            'format': 'unknown'
        }
        
        if path.is_file() and path.suffix.lower() == '.pdf':
            metadata.update(self._extract_pdf_metadata(path))
        elif path.is_dir() and (path / '.git').exists():
            metadata.update(self._extract_git_metadata(path))
        elif path.is_dir():
            metadata.update(self._extract_directory_metadata(path))
        
        return metadata
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        metadata = {
            'type': 'PDF',
            'format': 'PDF',
            'size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'subject': self._extract_subject_from_filename(pdf_path.stem)
        }
        
        # Try to extract PDF metadata if PyMuPDF is available
        try:
            import fitz  # PyMuPDF
            
            with fitz.open(str(pdf_path)) as doc:
                metadata.update({
                    'pages': doc.page_count,
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'subject_meta': doc.metadata.get('subject', '')
                })
        except ImportError:
            logger.debug("PyMuPDF not available, skipping PDF metadata extraction")
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return metadata
    
    def _extract_git_metadata(self, repo_path: Path) -> Dict[str, Any]:
        """Extract metadata from Git repository."""
        metadata = {
            'type': self._classify_repository_type(repo_path.name),
            'format': 'Git Repository',
            'subject': self._extract_subject_from_filename(repo_path.name)
        }
        
        # Calculate directory size
        total_size = 0
        file_count = 0
        try:
            for item in repo_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            metadata.update({
                'size_mb': total_size / (1024 * 1024),
                'file_count': file_count
            })
        except Exception as e:
            logger.warning(f"Error calculating repository size: {e}")
        
        return metadata
    
    def _extract_directory_metadata(self, dir_path: Path) -> Dict[str, Any]:
        """Extract metadata from general directory."""
        metadata = {
            'type': self._classify_directory_type(dir_path.name),
            'format': 'Directory',
            'subject': self._extract_subject_from_filename(dir_path.name)
        }
        
        # Calculate directory size and file count
        total_size = 0
        file_count = 0
        try:
            for item in dir_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            metadata.update({
                'size_mb': total_size / (1024 * 1024),
                'file_count': file_count
            })
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        
        return metadata