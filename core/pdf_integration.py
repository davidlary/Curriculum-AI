"""
Integration helpers for PDF processing with existing OpenBooks system.

This module provides utilities to integrate Claude-based PDF processing
with GetOpenBooks.py and curricula.py workflows.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import time
from .pdf_processor import PDFProcessor, PDFProcessingResult

logger = logging.getLogger(__name__)

class PDFContentManager:
    """
    Manager for integrating PDF content with existing OpenBooks workflows.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PDF content manager."""
        self.config = config or {}
        self.pdf_processor = None
        self._initialize_processor()
    
    def _initialize_processor(self):
        """Initialize PDF processor if API key is available."""
        try:
            self.pdf_processor = PDFProcessor(self.config)
            logger.info("PDF processor initialized successfully")
        except ValueError as e:
            logger.warning(f"PDF processor not available: {e}")
            self.pdf_processor = None
    
    def is_pdf_processing_available(self) -> bool:
        """Check if PDF processing is available."""
        return self.pdf_processor is not None
    
    def get_pdf_text(self, pdf_path: Union[str, Path]) -> Optional[str]:
        """
        Get text content from PDF, using cache if available.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content or None if processing failed
        """
        if not self.pdf_processor:
            logger.warning("PDF processor not available - skipping PDF text extraction")
            return None
        
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            result = self.pdf_processor.process_pdf(pdf_path)
            
            if result.success:
                logger.info(f"Successfully extracted text from {pdf_path.name}: "
                           f"{len(result.full_text)} characters")
                return result.full_text
            else:
                logger.error(f"Failed to extract text from {pdf_path.name}: {result.error_message}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def get_pdf_metadata(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Get metadata about PDF processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Metadata dictionary or None if not available
        """
        if not self.pdf_processor:
            return None
        
        try:
            pdf_path = Path(pdf_path)
            file_hash = self.pdf_processor.get_pdf_hash(pdf_path)
            cache_path = self.pdf_processor.get_cache_path(pdf_path, file_hash)
            
            cached_result = self.pdf_processor.load_from_cache(cache_path)
            if cached_result:
                return {
                    'total_pages': cached_result.total_pages,
                    'processed_pages': cached_result.processed_pages,
                    'processing_date': cached_result.processed_date,
                    'text_length': len(cached_result.full_text),
                    'chunks_count': len(cached_result.chunks_metadata),
                    'success': cached_result.success
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting PDF metadata for {pdf_path}: {e}")
            return None
    
    def find_pdfs_for_book(self, book_path: Path) -> List[Path]:
        """
        Find PDF files associated with a book directory.
        
        Args:
            book_path: Path to book directory
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        if book_path.exists() and book_path.is_dir():
            # Look for PDFs in the book directory and subdirectories
            pdf_files.extend(book_path.glob("*.pdf"))
            pdf_files.extend(book_path.glob("**/*.pdf"))
        
        return pdf_files
    
    def get_book_content(self, book_path: Path) -> Dict[str, Any]:
        """
        Get comprehensive content for a book including PDF text.
        
        Args:
            book_path: Path to book directory
            
        Returns:
            Dictionary containing all available content
        """
        content = {
            'book_path': str(book_path),
            'book_name': book_path.name,
            'pdfs': [],
            'git_content': {},
            'combined_text': ""
        }
        
        # Find and process PDFs
        pdf_files = self.find_pdfs_for_book(book_path)
        
        for pdf_file in pdf_files:
            pdf_content = {
                'file_path': str(pdf_file),
                'file_name': pdf_file.name,
                'text_content': None,
                'metadata': None
            }
            
            # Get PDF text
            text_content = self.get_pdf_text(pdf_file)
            if text_content:
                pdf_content['text_content'] = text_content
                content['combined_text'] += f"\n\n=== PDF: {pdf_file.name} ===\n\n{text_content}"
            
            # Get PDF metadata
            metadata = self.get_pdf_metadata(pdf_file)
            if metadata:
                pdf_content['metadata'] = metadata
            
            content['pdfs'].append(pdf_content)
        
        # Get Git repository content (existing logic would go here)
        # This would integrate with existing content extraction from .cnxml files
        
        return content
    
    def batch_process_books(self, books_directory: Path, 
                          languages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Batch process PDF content for multiple books.
        
        Args:
            books_directory: Path to Books directory
            languages: Optional list of languages to process
            
        Returns:
            Processing results summary
        """
        if not self.pdf_processor:
            logger.warning("PDF processor not available - skipping batch processing")
            return {'processed': 0, 'failed': 0, 'skipped': 1}
        
        results = {'processed': 0, 'failed': 0, 'skipped': 0}
        
        # Find all book directories
        book_paths = []
        
        if languages:
            for language in languages:
                lang_path = books_directory / language
                if lang_path.exists():
                    # Find all book directories in this language
                    for subject_path in lang_path.iterdir():
                        if subject_path.is_dir():
                            for level_path in subject_path.iterdir():
                                if level_path.is_dir():
                                    for book_path in level_path.iterdir():
                                        if book_path.is_dir():
                                            book_paths.append(book_path)
        else:
            # Process all languages
            for book_path in books_directory.rglob("osbooks-*"):
                if book_path.is_dir():
                    book_paths.append(book_path)
        
        logger.info(f"Found {len(book_paths)} book directories to process")
        
        # Process each book
        for book_path in book_paths:
            try:
                pdf_files = self.find_pdfs_for_book(book_path)
                
                if not pdf_files:
                    results['skipped'] += 1
                    continue
                
                # Process PDFs for this book
                success_count = 0
                for pdf_file in pdf_files:
                    text_content = self.get_pdf_text(pdf_file)
                    if text_content:
                        success_count += 1
                
                if success_count > 0:
                    results['processed'] += 1
                    logger.info(f"Processed {success_count} PDFs for {book_path.name}")
                else:
                    results['failed'] += 1
                    logger.warning(f"Failed to process PDFs for {book_path.name}")
                    
            except Exception as e:
                results['failed'] += 1
                logger.error(f"Error processing book {book_path.name}: {e}")
        
        return results


# Utility functions for integration with existing code

def get_text_content_for_book(book_path: Union[str, Path], 
                             config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Convenience function to get text content for a book.
    
    Usage in existing code:
        from core.pdf_integration import get_text_content_for_book
        
        book_text = get_text_content_for_book("Books/english/Physics/University/osbooks-physics")
        if book_text:
            # Use the extracted text for analysis, curriculum generation, etc.
            pass
    """
    manager = PDFContentManager(config)
    book_path = Path(book_path)
    
    content = manager.get_book_content(book_path)
    return content.get('combined_text', '') or None


def check_pdf_processing_status(books_directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Check the status of PDF processing across the collection.
    
    Returns:
        Status report with counts and details
    """
    books_directory = Path(books_directory)
    status = {
        'total_pdfs': 0,
        'processed_pdfs': 0,
        'unprocessed_pdfs': 0,
        'failed_pdfs': 0,
        'cache_files': 0,
        'details': []
    }
    
    manager = PDFContentManager()
    if not manager.is_pdf_processing_available():
        status['error'] = "PDF processing not available (missing API key)"
        return status
    
    # Find all PDFs
    for pdf_file in books_directory.rglob("*.pdf"):
        status['total_pdfs'] += 1
        
        # Check if processed
        metadata = manager.get_pdf_metadata(pdf_file)
        if metadata:
            status['cache_files'] += 1
            if metadata.get('success', False):
                status['processed_pdfs'] += 1
            else:
                status['failed_pdfs'] += 1
        else:
            status['unprocessed_pdfs'] += 1
        
        # Add to details
        relative_path = pdf_file.relative_to(books_directory)
        status['details'].append({
            'file': str(relative_path),
            'processed': metadata is not None,
            'success': metadata.get('success', False) if metadata else False,
            'pages': metadata.get('total_pages', 0) if metadata else 0
        })
    
    return status


def integrate_pdf_text_with_search_index(search_indexer, books_directory: Path):
    """
    Integration function for search indexing with PDF content.
    
    This can be called from the existing search indexer to include PDF text.
    """
    manager = PDFContentManager()
    if not manager.is_pdf_processing_available():
        logger.info("PDF processing not available - skipping PDF content indexing")
        return
    
    # Find all book directories
    for book_path in books_directory.rglob("osbooks-*"):
        if book_path.is_dir():
            try:
                # Get comprehensive book content including PDFs
                content = manager.get_book_content(book_path)
                
                if content['combined_text']:
                    # Add to search index
                    search_indexer.add_document({
                        'id': str(book_path.relative_to(books_directory)),
                        'title': content['book_name'],
                        'content': content['combined_text'],
                        'type': 'book_with_pdf',
                        'path': str(book_path),
                        'pdf_count': len(content['pdfs'])
                    })
                    
                    logger.info(f"Added PDF content to search index: {content['book_name']}")
                    
            except Exception as e:
                logger.error(f"Error indexing PDF content for {book_path.name}: {e}")


# Example integration snippets for existing code

def example_curriculum_integration():
    """
    Example of how to integrate PDF content into curriculum generation.
    """
    # This would be added to curricula.py
    
    from core.pdf_integration import get_text_content_for_book
    
    def generate_curriculum_with_pdf_content(book_path: str):
        """Modified curriculum generation with PDF content."""
        
        # Get both Git repository content (existing logic) and PDF content
        git_content = extract_existing_git_content(book_path)  # Existing function
        pdf_content = get_text_content_for_book(book_path)
        
        # Combine content for analysis
        combined_content = ""
        if git_content:
            combined_content += git_content
        if pdf_content:
            combined_content += f"\n\n=== PDF Content ===\n\n{pdf_content}"
        
        if combined_content:
            # Use combined content for LLM-based curriculum generation
            curriculum = generate_curriculum_from_text(combined_content)
            return curriculum
        else:
            logger.warning(f"No content available for {book_path}")
            return None


def example_search_integration():
    """
    Example of how to integrate PDF content into search functionality.
    """
    # This would be added to search indexing code
    
    from core.pdf_integration import PDFContentManager
    
    def build_search_index_with_pdfs(books_directory: Path):
        """Enhanced search index building with PDF content."""
        
        manager = PDFContentManager()
        search_documents = []
        
        for book_path in books_directory.rglob("osbooks-*"):
            if book_path.is_dir():
                # Get comprehensive content
                content = manager.get_book_content(book_path)
                
                # Create search document
                document = {
                    'id': str(book_path),
                    'title': content['book_name'],
                    'content': content['combined_text'],
                    'has_pdf': len(content['pdfs']) > 0,
                    'pdf_pages': sum(pdf.get('metadata', {}).get('total_pages', 0) 
                                   for pdf in content['pdfs'])
                }
                
                search_documents.append(document)
        
        return search_documents