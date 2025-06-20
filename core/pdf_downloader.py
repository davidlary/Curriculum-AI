"""
PDF downloader module for downloading OpenStax and other open textbook PDFs.

This module handles downloading PDF files from URLs and organizing them
using the same Language/Discipline/Level hierarchy as Git repositories.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import time

from .config import OpenBooksConfig

logger = logging.getLogger(__name__)


class PDFDownloader:
    """Downloads and manages PDF textbook files."""
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize with configuration."""
        self.config = config
        self.books_path = Path(config.books_path)
        self.books_path.mkdir(parents=True, exist_ok=True)
        
        # Create session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenBooks/1.0 (Educational Research; davidlary@me.com)',
            'Accept': 'application/pdf,*/*'
        })
    
    def download_pdf(self, pdf_info: Dict[str, Any], dry_run: bool = False) -> bool:
        """
        Download a single PDF file.
        
        Args:
            pdf_info: Dictionary containing PDF information (url, name, subject, etc.)
            dry_run: Preview mode without downloading
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            url = pdf_info.get('url')
            if not url:
                logger.error("No URL provided for PDF download")
                return False
            
            # Determine target path
            target_path = self._get_pdf_target_path(pdf_info)
            
            if dry_run:
                logger.info(f"[DRY RUN] Would download PDF: {pdf_info.get('name')} -> {target_path}")
                return True
            
            # Check if file already exists
            if target_path.exists():
                logger.info(f"PDF already exists: {target_path}")
                return True
            
            # Create directory structure
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading PDF: {pdf_info.get('name')} from {url}")
            
            # Download with streaming
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Verify it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.endswith('.pdf'):
                logger.warning(f"Content may not be PDF: {content_type}")
            
            # Download to temporary file first
            temp_path = target_path.with_suffix('.tmp')
            
            try:
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Validate downloaded file
                if self._validate_pdf_file(temp_path):
                    # Move to final location
                    temp_path.rename(target_path)
                    logger.info(f"Successfully downloaded PDF: {target_path}")
                    return True
                else:
                    logger.error(f"Downloaded file failed validation: {temp_path}")
                    temp_path.unlink(missing_ok=True)
                    return False
                    
            except Exception as e:
                logger.error(f"Error writing PDF file: {e}")
                temp_path.unlink(missing_ok=True)
                return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF: {e}")
            return False
    
    def download_pdfs_batch(self, pdf_books: List[Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
        """
        Download multiple PDF files.
        
        Args:
            pdf_books: List of PDF book information dictionaries
            dry_run: Preview mode without downloading
            
        Returns:
            Dictionary with download results
        """
        results = {
            'total': len(pdf_books),
            'successful': 0,
            'failed': 0,
            'already_exists': 0,
            'errors': []
        }
        
        for i, pdf_info in enumerate(pdf_books, 1):
            try:
                logger.info(f"Processing PDF {i}/{len(pdf_books)}: {pdf_info.get('name')}")
                
                if not dry_run:
                    # Add delay between downloads to be respectful
                    if i > 1:
                        time.sleep(self.config.request_delay_seconds)
                
                # Check if already exists
                target_path = self._get_pdf_target_path(pdf_info)
                if target_path.exists() and not dry_run:
                    logger.info(f"PDF already exists: {target_path}")
                    results['already_exists'] += 1
                    continue
                
                # Download
                success = self.download_pdf(pdf_info, dry_run)
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_info.get('name')}: {e}")
                results['failed'] += 1
                results['errors'].append(str(e))
        
        return results
    
    def _get_pdf_target_path(self, pdf_info: Dict[str, Any]) -> Path:
        """
        Determine target path for PDF using Language/Discipline/Level hierarchy.
        
        Args:
            pdf_info: Dictionary containing PDF information
            
        Returns:
            Path object for the target PDF file
        """
        # Extract information
        name = pdf_info.get('name', 'Unknown').strip()
        subject = pdf_info.get('subject', 'Other')
        level = pdf_info.get('level', 'University')
        
        # Detect level if not provided
        if level is None or level == 'University':
            from .repository_manager import RepositoryManager
            # Create a temporary repository manager to use its level detection
            repo_manager = RepositoryManager(self.config)
            level = repo_manager._determine_educational_level(pdf_info)
        
        # Clean up filename
        filename = self._sanitize_filename(name)
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        # Use same language detection as repository manager
        language = self._detect_language(pdf_info)
        
        # Map subject to discipline (same as repository manager)
        discipline = self._map_subject_to_discipline(subject)
        
        # Create path: Books/Language/Discipline/Level/filename.pdf
        target_path = self.books_path / language / discipline / level / filename
        
        return target_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for filesystem."""
        # Remove or replace problematic characters
        sanitized = filename.replace('/', '_').replace('\\', '_')
        sanitized = sanitized.replace(':', '_').replace('*', '_')
        sanitized = sanitized.replace('?', '_').replace('"', '_')
        sanitized = sanitized.replace('<', '_').replace('>', '_')
        sanitized = sanitized.replace('|', '_')
        
        # Remove extra spaces and truncate if too long
        sanitized = ' '.join(sanitized.split())
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        
        return sanitized
    
    def _detect_language(self, pdf_info: Dict[str, Any]) -> str:
        """Detect language from PDF information."""
        # For now, default to English
        # Could be enhanced to detect from title/description
        name = pdf_info.get('name', '').lower()
        
        # Simple language detection
        if any(word in name for word in ['spanish', 'español', 'espanol']):
            return 'spanish'
        elif any(word in name for word in ['french', 'français', 'francais']):
            return 'french'
        elif any(word in name for word in ['german', 'deutsch']):
            return 'german'
        elif any(word in name for word in ['polish', 'polski']):
            return 'polish'
        
        return 'english'  # Default
    
    def _map_subject_to_discipline(self, subject: str) -> str:
        """Map subject to discipline category."""
        if not subject:
            return 'Other'
        
        # Subject to discipline mapping
        discipline_mapping = {
            'Physics': 'Physics',
            'Biology': 'Biology', 
            'Chemistry': 'Chemistry',
            'Mathematics': 'Mathematics',
            'Math': 'Mathematics',
            'Statistics': 'Statistics',
            'Economics': 'Economics',
            'Psychology': 'Psychology',
            'Sociology': 'Sociology',
            'History': 'History',
            'Business': 'Business',
            'Computer Science': 'Computer Science',
            'Engineering': 'Engineering',
            'English': 'English',
            'Study Skills': 'Academic Skills'
        }
        
        # Try exact match first
        if subject in discipline_mapping:
            return discipline_mapping[subject]
        
        # Try partial matches
        subject_lower = subject.lower()
        for key, discipline in discipline_mapping.items():
            if key.lower() in subject_lower:
                return discipline
        
        # Default to capitalized subject
        return subject.title() if subject else 'Other'
    
    def _validate_pdf_file(self, file_path: Path) -> bool:
        """
        Validate that downloaded file is a valid PDF.
        
        Args:
            file_path: Path to the downloaded file
            
        Returns:
            True if file appears to be a valid PDF
        """
        try:
            # Check file size
            if file_path.stat().st_size < 1024:  # Less than 1KB
                logger.warning(f"PDF file too small: {file_path}")
                return False
            
            # Check PDF magic bytes
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    logger.warning(f"File does not appear to be PDF: {file_path}")
                    return False
            
            logger.debug(f"PDF validation successful: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF file {file_path}: {e}")
            return False
    
    def get_existing_pdfs(self) -> List[Dict[str, Any]]:
        """
        Get list of existing PDF files in the books directory.
        
        Returns:
            List of dictionaries with PDF file information
        """
        existing_pdfs = []
        
        try:
            for pdf_file in self.books_path.rglob("*.pdf"):
                size_mb = pdf_file.stat().st_size / (1024 * 1024)
                
                # Extract information from path structure
                path_parts = pdf_file.relative_to(self.books_path).parts
                
                pdf_info = {
                    'name': pdf_file.stem,
                    'path': str(pdf_file),
                    'size_mb': round(size_mb, 2),
                    'language': path_parts[0] if len(path_parts) > 0 else 'unknown',
                    'discipline': path_parts[1] if len(path_parts) > 1 else 'unknown', 
                    'level': path_parts[2] if len(path_parts) > 2 else 'unknown'
                }
                
                existing_pdfs.append(pdf_info)
                
        except Exception as e:
            logger.error(f"Error scanning existing PDFs: {e}")
        
        return existing_pdfs