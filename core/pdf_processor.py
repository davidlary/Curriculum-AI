"""
PDF processing module with Claude API integration and intelligent caching.

This module handles PDF text extraction using Claude's native PDF capabilities,
with intelligent chunking for large documents and comprehensive caching.
"""

import anthropic
import base64
import json
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime
import PyPDF2
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PDFProcessingResult:
    """Result of PDF processing with metadata."""
    file_path: str
    file_hash: str
    total_pages: int
    processed_pages: int
    full_text: str
    chunks_metadata: List[Dict[str, Any]]
    processing_time: float
    processed_date: str
    claude_model: str
    success: bool
    error_message: Optional[str] = None

class PDFProcessor:
    """
    Advanced PDF processor using Claude API with intelligent chunking and caching.
    """
    
    def __init__(self, config=None):
        """Initialize PDF processor with configuration."""
        self.config = config or {}
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Configuration settings
        self.max_pages_per_chunk = self.config.get('max_pages_per_chunk', 20)
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.model = self.config.get('claude_model', 'claude-3-5-sonnet-20241022')
        self.cache_dir = Path(self.config.get('cache_dir', './cache/pdf_processed'))
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)  # seconds between API calls
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'processed_count': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'total_pages_processed': 0,
            'processing_time': 0.0
        }
    
    def get_pdf_hash(self, pdf_path: Path) -> str:
        """Generate hash of PDF file for cache key."""
        with open(pdf_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def get_cache_path(self, pdf_path: Path, file_hash: str) -> Path:
        """Get cache file path for processed PDF."""
        pdf_name = pdf_path.stem
        cache_filename = f"{pdf_name}_{file_hash[:12]}.json"
        return self.cache_dir / cache_filename
    
    def load_from_cache(self, cache_path: Path) -> Optional[PDFProcessingResult]:
        """Load processed PDF result from cache."""
        try:
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return PDFProcessingResult(**data)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def save_to_cache(self, result: PDFProcessingResult, cache_path: Path):
        """Save processed PDF result to cache."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processed PDF to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {e}")
    
    def split_pdf_into_chunks(self, pdf_path: Path) -> List[Tuple[bytes, Dict[str, Any]]]:
        """
        Split PDF into manageable chunks with metadata.
        
        Returns:
            List of (pdf_bytes, metadata) tuples
        """
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"Splitting PDF {pdf_path.name} ({total_pages} pages) into chunks")
                
                # Create chunks of pages
                for start_page in range(0, total_pages, self.max_pages_per_chunk):
                    end_page = min(start_page + self.max_pages_per_chunk, total_pages)
                    
                    # Create new PDF with chunk pages
                    pdf_writer = PyPDF2.PdfWriter()
                    for page_num in range(start_page, end_page):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                    
                    # Write chunk to bytes
                    chunk_bytes = bytes()
                    import io
                    chunk_buffer = io.BytesIO()
                    pdf_writer.write(chunk_buffer)
                    chunk_bytes = chunk_buffer.getvalue()
                    chunk_buffer.close()
                    
                    # Create metadata
                    metadata = {
                        'chunk_index': len(chunks),
                        'start_page': start_page + 1,  # 1-indexed for human readability
                        'end_page': end_page,
                        'page_count': end_page - start_page,
                        'chunk_size_bytes': len(chunk_bytes)
                    }
                    
                    chunks.append((chunk_bytes, metadata))
                    logger.debug(f"Created chunk {len(chunks)}: pages {start_page+1}-{end_page}")
                
                logger.info(f"Split PDF into {len(chunks)} chunks")
                return chunks
                
        except Exception as e:
            logger.error(f"Failed to split PDF {pdf_path}: {e}")
            raise
    
    def process_pdf_chunk(self, chunk_bytes: bytes, metadata: Dict[str, Any]) -> str:
        """Process a single PDF chunk using Claude API."""
        try:
            chunk_base64 = base64.b64encode(chunk_bytes).decode('utf-8')
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": chunk_base64
                        }
                    }, {
                        "type": "text", 
                        "text": """Extract all text content from this PDF chunk without summarizing. 
                        Preserve formatting, headings, and structure as much as possible. 
                        Include any tables, figures captions, and mathematical content.
                        If there are equations, preserve them in a readable format."""
                    }]
                }]
            )
            
            self.stats['api_calls'] += 1
            
            # Extract text from response
            text_content = response.content[0].text if response.content else ""
            
            logger.debug(f"Processed chunk {metadata['chunk_index']}: "
                        f"pages {metadata['start_page']}-{metadata['end_page']}, "
                        f"{len(text_content)} characters extracted")
            
            return text_content
            
        except Exception as e:
            logger.error(f"Failed to process PDF chunk {metadata['chunk_index']}: {e}")
            raise
    
    def process_pdf(self, pdf_path: Path, force_reprocess: bool = False) -> PDFProcessingResult:
        """
        Process a PDF file with intelligent chunking and caching.
        
        Args:
            pdf_path: Path to PDF file
            force_reprocess: If True, ignore cache and reprocess
            
        Returns:
            PDFProcessingResult with extracted text and metadata
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Generate file hash for caching
        file_hash = self.get_pdf_hash(pdf_path)
        cache_path = self.get_cache_path(pdf_path, file_hash)
        
        # Check cache first (unless forced reprocessing)
        if not force_reprocess:
            cached_result = self.load_from_cache(cache_path)
            if cached_result:
                logger.info(f"Found cached result for {pdf_path.name}")
                self.stats['cache_hits'] += 1
                return cached_result
        
        try:
            # Split PDF into chunks
            chunks = self.split_pdf_into_chunks(pdf_path)
            
            if not chunks:
                raise ValueError("No chunks created from PDF")
            
            # Process each chunk
            full_text = ""
            chunks_metadata = []
            processed_pages = 0
            
            logger.info(f"Processing {len(chunks)} chunks for {pdf_path.name}")
            
            for chunk_bytes, chunk_metadata in chunks:
                try:
                    chunk_text = self.process_pdf_chunk(chunk_bytes, chunk_metadata)
                    
                    # Add chunk separator and text
                    if full_text:
                        full_text += f"\n\n--- PAGE {chunk_metadata['start_page']} ---\n\n"
                    full_text += chunk_text
                    
                    # Update metadata
                    chunk_metadata['text_length'] = len(chunk_text)
                    chunk_metadata['success'] = True
                    chunks_metadata.append(chunk_metadata)
                    
                    processed_pages += chunk_metadata['page_count']
                    
                    logger.info(f"Completed chunk {chunk_metadata['chunk_index'] + 1}/{len(chunks)} "
                              f"for {pdf_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_metadata['chunk_index']}: {e}")
                    chunk_metadata['success'] = False
                    chunk_metadata['error'] = str(e)
                    chunks_metadata.append(chunk_metadata)
            
            # Calculate total pages from original PDF
            with open(pdf_path, 'rb') as f:
                total_pages = len(PyPDF2.PdfReader(f).pages)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = PDFProcessingResult(
                file_path=str(pdf_path),
                file_hash=file_hash,
                total_pages=total_pages,
                processed_pages=processed_pages,
                full_text=full_text,
                chunks_metadata=chunks_metadata,
                processing_time=processing_time,
                processed_date=datetime.now().isoformat(),
                claude_model=self.model,
                success=True
            )
            
            # Save to cache
            self.save_to_cache(result, cache_path)
            
            # Update statistics
            self.stats['processed_count'] += 1
            self.stats['total_pages_processed'] += processed_pages
            self.stats['processing_time'] += processing_time
            
            logger.info(f"Successfully processed {pdf_path.name}: "
                       f"{processed_pages}/{total_pages} pages, "
                       f"{len(full_text)} characters, "
                       f"{processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = PDFProcessingResult(
                file_path=str(pdf_path),
                file_hash=file_hash,
                total_pages=0,
                processed_pages=0,
                full_text="",
                chunks_metadata=[],
                processing_time=processing_time,
                processed_date=datetime.now().isoformat(),
                claude_model=self.model,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Failed to process PDF {pdf_path.name}: {e}")
            return error_result
    
    def process_directory(self, directory_path: Path, pattern: str = "*.pdf") -> List[PDFProcessingResult]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Directory containing PDFs
            pattern: File pattern to match (default: "*.pdf")
            
        Returns:
            List of processing results
        """
        directory_path = Path(directory_path)
        pdf_files = list(directory_path.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path} matching {pattern}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process in {directory_path}")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                # Continue with other files
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['processed_count']),
            'avg_processing_time': self.stats['processing_time'] / max(1, self.stats['processed_count']),
            'avg_pages_per_pdf': self.stats['total_pages_processed'] / max(1, self.stats['processed_count'])
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear processed PDF cache.
        
        Args:
            older_than_days: Only clear cache files older than this many days
        """
        if older_than_days:
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            cache_files = [f for f in self.cache_dir.glob("*.json") 
                          if f.stat().st_mtime < cutoff_time]
        else:
            cache_files = list(self.cache_dir.glob("*.json"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file.name}")
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file.name}: {e}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")


def find_pdfs_in_books_directory(books_path: Path) -> List[Path]:
    """Find all PDF files in the Books directory structure."""
    pdf_files = []
    for pdf_file in books_path.rglob("*.pdf"):
        pdf_files.append(pdf_file)
    return pdf_files


def integrate_with_existing_system(config_path: Optional[str] = None):
    """
    Integration function for existing GetOpenBooks.py and curricula.py
    
    This function can be called from existing code to process PDFs
    """
    # Load configuration if provided
    config = {}
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('pdf_processing', {})
    
    # Initialize processor
    processor = PDFProcessor(config)
    
    # Find and process PDFs
    books_path = Path("./Books")
    if books_path.exists():
        pdf_files = find_pdfs_in_books_directory(books_path)
        logger.info(f"Found {len(pdf_files)} PDF files in Books directory")
        
        # Process all PDFs
        results = []
        for pdf_file in pdf_files:
            result = processor.process_pdf(pdf_file)
            results.append(result)
        
        # Print statistics
        stats = processor.get_statistics()
        logger.info(f"PDF Processing Complete: {stats}")
        
        return results
    else:
        logger.warning("Books directory not found")
        return []


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Process a single PDF
    processor = PDFProcessor()
    
    # Or process entire Books directory
    results = integrate_with_existing_system()
    
    for result in results:
        if result.success:
            print(f"✅ {Path(result.file_path).name}: {result.processed_pages} pages processed")
        else:
            print(f"❌ {Path(result.file_path).name}: {result.error_message}")