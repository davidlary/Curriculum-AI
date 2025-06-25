"""
Table of Contents Extractor - Core functionality for extracting TOC from books.

Goal: Extract structured table of contents from various book formats (PDF, XML, Markdown, etc.)
and provide standardized output format for further processing.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """
    Goal: Represent a single table of contents entry with hierarchical structure.
    
    Provides standardized representation of headings/subheadings with level information
    and source metadata for tracking and processing.
    """
    title: str
    level: int  # 1 = chapter, 2 = section, 3 = subsection, etc.
    page_number: Optional[int] = None
    section_number: Optional[str] = None
    parent_id: Optional[str] = None
    entry_id: str = ""
    children: List['TOCEntry'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'level': self.level,
            'page_number': self.page_number,
            'section_number': self.section_number,
            'parent_id': self.parent_id,
            'entry_id': self.entry_id,
            'children': [child.to_dict() for child in self.children]
        }


@dataclass
class BookTOC:
    """
    Goal: Represent complete table of contents for a book with metadata.
    
    Contains all TOC entries plus book metadata for identification and processing.
    """
    book_title: str
    language: str
    discipline: str
    level: str
    file_path: str
    entries: List[TOCEntry] = field(default_factory=list)
    extraction_method: str = ""
    total_entries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'book_title': self.book_title,
            'language': self.language,
            'discipline': self.discipline,
            'level': self.level,
            'file_path': self.file_path,
            'extraction_method': self.extraction_method,
            'total_entries': self.total_entries,
            'entries': [entry.to_dict() for entry in self.entries]
        }


class TOCExtractor:
    """
    Goal: Extract table of contents from various book formats.
    
    Provides unified interface for extracting structured TOC data from PDFs,
    XML files, Markdown, and other educational content formats.
    """
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.xml', '.cnxml', '.md', '.rst', '.tex', '.pkl', '.json', '.cache'}
        logger.info("TOCExtractor initialized with support for formats: %s", self.supported_formats)
    
    def extract_toc(self, file_path: Path, language: str, discipline: str, level: str) -> Optional[BookTOC]:
        """
        Goal: Extract table of contents from a book file.
        
        Args:
            file_path: Path to the book file
            language: Language of the book content
            discipline: Academic discipline
            level: Educational level
            
        Returns:
            BookTOC object with extracted entries or None if extraction fails
        """
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported format: {file_path.suffix}")
            return None
        
        book_title = file_path.stem
        logger.info(f"Extracting TOC from {book_title} ({file_path.suffix})")
        
        try:
            # Handle cached PDF content first (preferred over raw PDFs)
            if file_path.name.endswith('.pdf.cache'):
                entries = self._extract_cached_pdf_toc(file_path)
                method = "Cached PDF extraction"
            elif file_path.name.endswith('.pdf.pkl'):
                entries = self._extract_pickled_pdf_toc(file_path)
                method = "Pickled PDF extraction"
            elif file_path.name.endswith('.pdf.json'):
                entries = self._extract_json_pdf_toc(file_path)
                method = "JSON PDF extraction"
            elif file_path.suffix.lower() == '.pdf':
                # Check for cached version first
                cached_entries = self._try_cached_pdf_versions(file_path)
                if cached_entries:
                    entries = cached_entries
                    method = "Cached PDF extraction"
                else:
                    entries = self._extract_pdf_toc(file_path)
                    method = "PDF extraction"
            elif file_path.suffix.lower() in ['.xml', '.cnxml']:
                entries = self._extract_xml_toc(file_path)
                method = "XML extraction"
            elif file_path.suffix.lower() == '.md':
                entries = self._extract_markdown_toc(file_path)
                method = "Markdown extraction"
            elif file_path.suffix.lower() == '.rst':
                entries = self._extract_rst_toc(file_path)
                method = "reStructuredText extraction"
            elif file_path.suffix.lower() == '.tex':
                entries = self._extract_latex_toc(file_path)
                method = "LaTeX extraction"
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
            
            toc = BookTOC(
                book_title=book_title,
                language=language,
                discipline=discipline,
                level=level,
                file_path=str(file_path),
                entries=entries,
                extraction_method=method,
                total_entries=len(entries)
            )
            
            logger.info(f"Extracted {len(entries)} TOC entries from {book_title}")
            return toc
            
        except Exception as e:
            logger.error(f"Error extracting TOC from {file_path}: {e}")
            return None
    
    def _extract_pdf_toc(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract table of contents from PDF files using data-driven approach.
        
        Fully data-driven method that reads actual PDF content and uses multiple
        extraction strategies to identify real chapter/section structure.
        """
        entries = []
        
        try:
            # Try multiple PDF libraries for better compatibility
            entries = self._extract_pdf_with_pypdf2(file_path)
            
            if not entries:
                entries = self._extract_pdf_with_pdfplumber(file_path)
            
            if not entries:
                entries = self._extract_pdf_with_pymupdf(file_path)
                
            if not entries:
                # Last resort: basic text pattern matching
                entries = self._extract_pdf_basic_patterns(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting PDF TOC from {file_path}: {e}")
            
        return entries
    
    def _extract_pdf_with_pypdf2(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract PDF TOC using PyPDF2 (most reliable for bookmarks).
        """
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Strategy 1: Use PDF bookmarks/outline (most reliable)
                if hasattr(pdf_reader, 'outline') and pdf_reader.outline:
                    entries = self._process_pdf_outline(pdf_reader.outline)
                    if entries:
                        logger.info(f"Extracted {len(entries)} entries from PDF outline")
                        return entries
                
                # Strategy 2: Data-driven text analysis
                entries = self._extract_pdf_content_analysis(pdf_reader)
                if entries:
                    logger.info(f"Extracted {len(entries)} entries from PDF content analysis")
                    return entries
                    
        except ImportError:
            logger.debug("PyPDF2 not available")
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed: {e}")
            
        return []
    
    def _extract_pdf_with_pdfplumber(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract PDF TOC using pdfplumber for better text extraction.
        """
        try:
            import pdfplumber
            
            entries = []
            entry_id = 1
            
            with pdfplumber.open(file_path) as pdf:
                # Focus on first 20 pages for TOC
                for page_num, page in enumerate(pdf.pages[:20]):
                    text = page.extract_text()
                    if text:
                        page_entries = self._analyze_page_for_toc_entries(text, page_num + 1, entry_id)
                        entries.extend(page_entries)
                        entry_id += len(page_entries)
                        
                        # Stop if we've found substantial TOC content
                        if len(entries) > 20:
                            break
            
            return entries
            
        except ImportError:
            logger.debug("pdfplumber not available")
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed: {e}")
            
        return []
    
    def _extract_pdf_with_pymupdf(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract PDF TOC using PyMuPDF (fitz) for comprehensive extraction.
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            entries = []
            entry_id = 1
            
            # Strategy 1: Use built-in TOC
            toc = doc.get_toc()
            if toc:
                for item in toc:
                    level, title, page_num = item
                    entries.append(TOCEntry(
                        title=title.strip(),
                        level=level,
                        page_number=page_num,
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
                
                doc.close()
                return entries
            
            # Strategy 2: Text analysis on early pages
            for page_num in range(min(20, doc.page_count)):
                page = doc[page_num]
                text = page.get_text()
                page_entries = self._analyze_page_for_toc_entries(text, page_num + 1, entry_id)
                entries.extend(page_entries)
                entry_id += len(page_entries)
            
            doc.close()
            return entries
            
        except ImportError:
            logger.debug("PyMuPDF not available")
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed: {e}")
            
        return []
    
    def _extract_pdf_content_analysis(self, pdf_reader) -> List[TOCEntry]:
        """
        Goal: Data-driven content analysis of PDF text for TOC extraction.
        """
        entries = []
        entry_id = 1
        
        # Analyze first 20 pages for TOC patterns
        for page_num, page in enumerate(pdf_reader.pages[:20]):
            try:
                text = page.extract_text()
                if text:
                    page_entries = self._analyze_page_for_toc_entries(text, page_num + 1, entry_id)
                    entries.extend(page_entries)
                    entry_id += len(page_entries)
                    
                    # If we found many entries on one page, likely a TOC page
                    if len(page_entries) > 5:
                        logger.info(f"Found TOC page at page {page_num + 1}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Error processing page {page_num}: {e}")
                continue
        
        return entries
    
    def _analyze_page_for_toc_entries(self, text: str, page_num: int, start_entry_id: int) -> List[TOCEntry]:
        """
        Goal: Analyze page text for potential TOC entries using data-driven patterns.
        """
        entries = []
        entry_id = start_entry_id
        lines = text.split('\n')
        
        # Data-driven patterns based on common textbook structures
        toc_patterns = [
            # Chapter patterns
            r'^(?:Chapter\s+)?(\d+)[\.\:\s]+(.+?)(?:\s+\d+\s*)?$',
            r'^(Chapter\s+\d+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            r'^(\d+\.\s+.+?)(?:\s+\d+\s*)?$',
            
            # Section patterns  
            r'^(\d+\.\d+\s+.+?)(?:\s+\d+\s*)?$',
            r'^(\d+\.\d+\.\d+\s+.+?)(?:\s+\d+\s*)?$',
            
            # Part/Unit patterns
            r'^(Part\s+[IVX]+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            r'^(Unit\s+\d+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            
            # Appendix patterns
            r'^(Appendix\s+[A-Z][\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            
            # Generic heading patterns (ALL CAPS, Title Case)
            r'^([A-Z][A-Z\s&\-]{3,50})(?:\s+\d+\s*)?$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8})(?:\s+\d+\s*)?$'
        ]
        
        for line in lines:
            line = line.strip()
            
            # Skip very short or very long lines
            if len(line) < 5 or len(line) > 150:
                continue
                
            # Skip lines that are likely not TOC entries
            if any(skip_word in line.lower() for skip_word in 
                   ['page', 'continued', 'see also', 'figure', 'table', 'index']):
                continue
            
            for pattern_idx, pattern in enumerate(toc_patterns):
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Extract title and determine level
                    if len(match.groups()) == 2:
                        title = f"{match.group(1)} {match.group(2)}".strip()
                    else:
                        title = match.group(1).strip()
                    
                    # Determine hierarchical level based on pattern
                    level = self._determine_toc_level(title, pattern_idx)
                    
                    # Extract page number if present
                    page_match = re.search(r'\b(\d+)\s*$', line)
                    page_number = int(page_match.group(1)) if page_match else None
                    
                    entries.append(TOCEntry(
                        title=title,
                        level=level,
                        page_number=page_number,
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
                    break
        
        return entries
    
    def _determine_toc_level(self, title: str, pattern_idx: int) -> int:
        """
        Goal: Determine hierarchical level based on title content and pattern.
        """
        # Level determination based on pattern type and content analysis
        if pattern_idx <= 2:  # Chapter patterns
            return 1
        elif pattern_idx <= 4:  # Section patterns
            if title.count('.') >= 2:
                return 3  # Subsection
            else:
                return 2  # Section
        elif pattern_idx <= 7:  # Part/Unit/Appendix patterns
            return 1
        else:  # Generic patterns
            # Analyze content for level hints
            if any(word in title.lower() for word in ['chapter', 'part', 'unit']):
                return 1
            elif re.match(r'^\d+\.\d+', title):
                return 2
            elif re.match(r'^\d+\.\d+\.\d+', title):
                return 3
            else:
                return 2  # Default to section level
    
    def _extract_pdf_basic_patterns(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Basic pattern matching fallback for PDF text extraction.
        """
        try:
            # Try to extract text using basic methods
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                entries = self._extract_pdf_text_patterns(pdf_reader)
                
        except Exception:
            # Ultimate fallback - create minimal entry
            entries = [TOCEntry(
                title=f"Content from {file_path.stem}",
                level=1,
                entry_id="entry_1"
            )]
            
        return entries
    
    def _try_cached_pdf_versions(self, pdf_path: Path) -> Optional[List[TOCEntry]]:
        """
        Goal: Try to find and use cached Python-readable versions of PDF content.
        
        Looks for cached versions in common formats: .cache, .pkl, .json
        """
        base_path = pdf_path.with_suffix('')
        
        # Try different cached formats
        cache_extensions = ['.pdf.cache', '.pdf.pkl', '.pdf.json']
        
        for ext in cache_extensions:
            cache_path = Path(str(base_path) + ext)
            if cache_path.exists():
                logger.info(f"Found cached version: {cache_path}")
                
                if ext == '.pdf.cache':
                    return self._extract_cached_pdf_toc(cache_path)
                elif ext == '.pdf.pkl':
                    return self._extract_pickled_pdf_toc(cache_path)
                elif ext == '.pdf.json':
                    return self._extract_json_pdf_toc(cache_path)
        
        # Check in same directory with different naming
        cache_dir = pdf_path.parent
        pdf_name = pdf_path.stem
        
        for cache_file in cache_dir.glob(f"{pdf_name}*"):
            if any(cache_file.name.endswith(ext) for ext in ['.cache', '.pkl', '.json']):
                logger.info(f"Found related cache file: {cache_file}")
                
                if cache_file.suffix == '.cache' or cache_file.name.endswith('.pdf.cache'):
                    return self._extract_cached_pdf_toc(cache_file)
                elif cache_file.suffix == '.pkl' or cache_file.name.endswith('.pdf.pkl'):
                    return self._extract_pickled_pdf_toc(cache_file)
                elif cache_file.suffix == '.json' or cache_file.name.endswith('.pdf.json'):
                    return self._extract_json_pdf_toc(cache_file)
        
        return None
    
    def _extract_cached_pdf_toc(self, cache_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract TOC from cached PDF content (.cache format).
        
        Data-driven extraction from cached Python-readable content.
        """
        entries = []
        entry_id = 1
        
        try:
            # Try to read as text file first
            with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Analyze content for TOC patterns
            entries = self._analyze_cached_content_for_toc(content, entry_id)
            
            if entries:
                logger.info(f"Extracted {len(entries)} entries from cached content")
                
        except Exception as e:
            logger.error(f"Error reading cached PDF content: {e}")
            
        return entries
    
    def _extract_pickled_pdf_toc(self, pkl_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract TOC from pickled PDF content (.pkl format).
        """
        entries = []
        
        try:
            import pickle
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different pickle data structures
            if isinstance(data, dict):
                # Look for TOC-related keys
                if 'toc' in data:
                    entries = self._process_pickled_toc_data(data['toc'])
                elif 'outline' in data:
                    entries = self._process_pickled_toc_data(data['outline'])
                elif 'chapters' in data:
                    entries = self._process_pickled_chapters_data(data['chapters'])
                elif 'content' in data:
                    # Analyze content text
                    content = str(data['content'])
                    entries = self._analyze_cached_content_for_toc(content, 1)
                else:
                    # Try to extract from all text data
                    all_text = ' '.join(str(v) for v in data.values() if isinstance(v, (str, list)))
                    entries = self._analyze_cached_content_for_toc(all_text, 1)
                    
            elif isinstance(data, list):
                # Try to process as list of content
                entries = self._process_pickled_list_data(data)
                
            elif isinstance(data, str):
                # Process as text content
                entries = self._analyze_cached_content_for_toc(data, 1)
            
            if entries:
                logger.info(f"Extracted {len(entries)} entries from pickled content")
                
        except Exception as e:
            logger.error(f"Error reading pickled PDF content: {e}")
            
        return entries
    
    def _extract_json_pdf_toc(self, json_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract TOC from JSON PDF content (.json format).
        """
        entries = []
        
        try:
            import json
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Look for TOC-related keys
                if 'toc' in data:
                    entries = self._process_json_toc_data(data['toc'])
                elif 'outline' in data:
                    entries = self._process_json_toc_data(data['outline'])
                elif 'chapters' in data:
                    entries = self._process_json_chapters_data(data['chapters'])
                elif 'content' in data:
                    # Analyze content
                    content = str(data['content'])
                    entries = self._analyze_cached_content_for_toc(content, 1)
                else:
                    # Extract from all text fields
                    all_text = self._extract_text_from_json(data)
                    entries = self._analyze_cached_content_for_toc(all_text, 1)
                    
            elif isinstance(data, list):
                entries = self._process_json_list_data(data)
            
            if entries:
                logger.info(f"Extracted {len(entries)} entries from JSON content")
                
        except Exception as e:
            logger.error(f"Error reading JSON PDF content: {e}")
            
        return entries
    
    def _analyze_cached_content_for_toc(self, content: str, start_entry_id: int) -> List[TOCEntry]:
        """
        Goal: Analyze cached content text for TOC patterns.
        
        Data-driven analysis of cached PDF content to identify chapter/section structure.
        """
        entries = []
        entry_id = start_entry_id
        
        # Split content into manageable chunks
        lines = content.split('\n')
        
        # Enhanced patterns for cached content
        toc_patterns = [
            # Standard chapter/section patterns
            r'^(?:Chapter\s+)?(\d+)[\.\:\s]+(.+?)(?:\s+\d+\s*)?$',
            r'^(Chapter\s+\d+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            r'^(\d+\.\s+.+?)(?:\s+\d+\s*)?$',
            r'^(\d+\.\d+\s+.+?)(?:\s+\d+\s*)?$',
            
            # Part/Section/Appendix patterns
            r'^(Part\s+[IVX]+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            r'^(Section\s+\d+[\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            r'^(Appendix\s+[A-Z][\.\:\s]+.+?)(?:\s+\d+\s*)?$',
            
            # Content-based patterns
            r'^([A-Z][A-Z\s]{5,50})$',  # ALL CAPS headings
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,6})$',  # Title Case headings
            
            # TOC-specific patterns in cached content
            r'(?:^|\n)(Table\s+of\s+Contents?.*?)(?:\n|$)',
            r'(?:^|\n)(Contents?.*?)(?:\n|$)',
        ]
        
        toc_section_found = False
        current_toc_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Detect TOC section
            if re.search(r'\b(?:table\s+of\s+)?contents?\b', line, re.IGNORECASE):
                toc_section_found = True
                continue
            
            # If we're in TOC section, collect lines
            if toc_section_found:
                if len(line) < 5:
                    continue
                    
                # Stop if we hit common post-TOC sections
                if any(end_marker in line.lower() for end_marker in 
                       ['preface', 'introduction', 'chapter 1', 'part i', 'acknowledgments']):
                    break
                    
                current_toc_lines.append(line)
                
                # Process if we have enough lines
                if len(current_toc_lines) > 20:
                    break
        
        # Process TOC lines if found
        if current_toc_lines:
            entries.extend(self._process_toc_lines(current_toc_lines, entry_id))
        else:
            # Fall back to general content analysis
            entries.extend(self._analyze_general_content_structure(lines[:500], entry_id))
        
        return entries
    
    def _process_toc_lines(self, toc_lines: List[str], start_entry_id: int) -> List[TOCEntry]:
        """
        Goal: Process identified TOC lines into structured entries.
        """
        entries = []
        entry_id = start_entry_id
        
        for line in toc_lines:
            # Skip very short lines or page numbers only
            if len(line) < 5 or line.isdigit():
                continue
            
            # Extract title and page number
            page_match = re.search(r'\b(\d+)\s*$', line)
            page_number = int(page_match.group(1)) if page_match else None
            
            # Clean title (remove page number)
            title = re.sub(r'\s+\d+\s*$', '', line).strip()
            
            if not title:
                continue
            
            # Determine level based on formatting and content
            level = self._determine_content_level(title, line)
            
            entries.append(TOCEntry(
                title=title,
                level=level,
                page_number=page_number,
                entry_id=f"entry_{entry_id}"
            ))
            entry_id += 1
        
        return entries
    
    def _analyze_general_content_structure(self, lines: List[str], start_entry_id: int) -> List[TOCEntry]:
        """
        Goal: Analyze general content structure when no explicit TOC is found.
        """
        entries = []
        entry_id = start_entry_id
        
        # Look for structural patterns in content
        for line in lines:
            line = line.strip()
            
            if len(line) < 5 or len(line) > 100:
                continue
            
            # Check for chapter/section indicators
            if re.match(r'^(?:Chapter|Section|Part|Unit)\s+\d+', line, re.IGNORECASE):
                level = 1 if 'chapter' in line.lower() or 'part' in line.lower() else 2
                entries.append(TOCEntry(
                    title=line,
                    level=level,
                    entry_id=f"entry_{entry_id}"
                ))
                entry_id += 1
            
            # Check for numbered sections
            elif re.match(r'^\d+\.\s+', line):
                entries.append(TOCEntry(
                    title=line,
                    level=2,
                    entry_id=f"entry_{entry_id}"
                ))
                entry_id += 1
        
        return entries
    
    def _determine_content_level(self, title: str, original_line: str) -> int:
        """
        Goal: Determine hierarchical level for cached content entries.
        """
        title_lower = title.lower()
        
        # Level 1: Chapters, Parts, Major sections
        if any(word in title_lower for word in ['chapter', 'part', 'unit', 'section i', 'appendix']):
            return 1
        
        # Level 2: Numbered sections
        if re.match(r'^\d+\.\d+', title) or 'section' in title_lower:
            return 2
        
        # Level 3: Subsections
        if re.match(r'^\d+\.\d+\.\d+', title) or title_lower.startswith('subsection'):
            return 3
        
        # Default to level 2 for most content
        return 2
    
    def _process_pickled_toc_data(self, toc_data) -> List[TOCEntry]:
        """Goal: Process structured TOC data from pickle format."""
        entries = []
        entry_id = 1
        
        if isinstance(toc_data, list):
            for item in toc_data:
                if isinstance(item, dict) and 'title' in item:
                    entries.append(TOCEntry(
                        title=item['title'],
                        level=item.get('level', 1),
                        page_number=item.get('page', None),
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    title = str(item[1]) if len(item) > 1 else str(item[0])
                    level = item[0] if isinstance(item[0], int) else 1
                    entries.append(TOCEntry(
                        title=title,
                        level=level,
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
        
        return entries
    
    def _process_json_toc_data(self, toc_data) -> List[TOCEntry]:
        """Goal: Process structured TOC data from JSON format."""
        return self._process_pickled_toc_data(toc_data)  # Same logic
    
    def _process_pickled_chapters_data(self, chapters_data) -> List[TOCEntry]:
        """Goal: Process chapter-specific data from pickle format."""
        entries = []
        entry_id = 1
        
        if isinstance(chapters_data, list):
            for i, chapter in enumerate(chapters_data):
                if isinstance(chapter, dict):
                    title = chapter.get('title', f'Chapter {i+1}')
                    entries.append(TOCEntry(
                        title=title,
                        level=1,
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
                elif isinstance(chapter, str):
                    entries.append(TOCEntry(
                        title=chapter,
                        level=1,
                        entry_id=f"entry_{entry_id}"
                    ))
                    entry_id += 1
                    
        return entries
    
    def _process_json_chapters_data(self, chapters_data) -> List[TOCEntry]:
        """Goal: Process chapter-specific data from JSON format."""
        return self._process_pickled_chapters_data(chapters_data)
    
    def _process_pickled_list_data(self, list_data) -> List[TOCEntry]:
        """Goal: Process list data from pickle format."""
        entries = []
        entry_id = 1
        
        for item in list_data:
            if isinstance(item, dict) and 'title' in item:
                entries.append(TOCEntry(
                    title=item['title'],
                    level=item.get('level', 1),
                    entry_id=f"entry_{entry_id}"
                ))
                entry_id += 1
            elif isinstance(item, str) and len(item) > 3:
                # Analyze string for chapter/section patterns
                level = 1 if any(word in item.lower() for word in ['chapter', 'part']) else 2
                entries.append(TOCEntry(
                    title=item,
                    level=level,
                    entry_id=f"entry_{entry_id}"
                ))
                entry_id += 1
                
        return entries
    
    def _process_json_list_data(self, list_data) -> List[TOCEntry]:
        """Goal: Process list data from JSON format."""
        return self._process_pickled_list_data(list_data)
    
    def _extract_text_from_json(self, data, max_chars: int = 10000) -> str:
        """Goal: Extract text content from JSON structure."""
        text_parts = []
        
        def extract_recursive(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item, depth + 1)
        
        extract_recursive(data)
        full_text = ' '.join(text_parts)
        
        return full_text[:max_chars]  # Limit size
    
    def _process_pdf_outline(self, outline: List) -> List[TOCEntry]:
        """
        Goal: Process PDF outline/bookmarks into TOC entries.
        """
        entries = []
        entry_id = 1
        
        def process_outline_item(item, level: int = 1, parent_id: str = None):
            nonlocal entry_id
            
            if hasattr(item, 'title'):
                entry = TOCEntry(
                    title=item.title,
                    level=level,
                    entry_id=f"entry_{entry_id}",
                    parent_id=parent_id
                )
                entries.append(entry)
                current_id = entry.entry_id
                entry_id += 1
                
                # Process children if they exist
                if hasattr(item, '__iter__') and not isinstance(item, str):
                    for child in item:
                        process_outline_item(child, level + 1, current_id)
        
        for item in outline:
            process_outline_item(item)
            
        return entries
    
    def _extract_pdf_text_patterns(self, pdf_reader) -> List[TOCEntry]:
        """
        Goal: Extract TOC from PDF text using pattern matching.
        """
        entries = []
        entry_id = 1
        
        # Common TOC patterns
        patterns = [
            r'^(Chapter\s+\d+[:\.\-\s]+.+)$',
            r'^(Section\s+\d+[:\.\-\s]+.+)$',
            r'^(\d+\.\s+.+)$',
            r'^(\d+\.\d+\s+.+)$',
            r'^([A-Z][A-Z\s&]+)$',  # ALL CAPS headings
        ]
        
        for page_num, page in enumerate(pdf_reader.pages[:10]):  # Check first 10 pages for TOC
            try:
                text = page.extract_text()
                lines = text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if len(line) < 3 or len(line) > 100:
                        continue
                    
                    for pattern_idx, pattern in enumerate(patterns):
                        match = re.match(pattern, line, re.MULTILINE)
                        if match:
                            title = match.group(1).strip()
                            level = min(pattern_idx + 1, 4)
                            
                            entry = TOCEntry(
                                title=title,
                                level=level,
                                page_number=page_num + 1,
                                entry_id=f"entry_{entry_id}"
                            )
                            entries.append(entry)
                            entry_id += 1
                            break
                            
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        return entries
    
    def _extract_pdf_fallback(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Fallback PDF extraction without PyPDF2.
        """
        # For now, return empty list. Could implement alternative PDF libraries here.
        logger.warning(f"PDF extraction not available for {file_path}")
        return []
    
    def _extract_xml_toc(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract table of contents from XML/CNXML files.
        
        Handles OpenStax collection.xml format and other XML structures.
        """
        entries = []
        entry_id = 1
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Store file path for module extraction context
            self._current_file_path = file_path
            
            # Check if this is an OpenStax collection.xml file
            if 'collection' in file_path.name.lower() or 'col:collection' in str(root.tag):
                return self._extract_openstax_collection_toc(root, entry_id)
            
            # Generic XML TOC extraction
            return self._extract_generic_xml_toc(root, entry_id)
            
        except Exception as e:
            logger.error(f"Error extracting XML TOC from {file_path}: {e}")
            return []
    
    def _extract_openstax_collection_toc(self, root, entry_id: int) -> List[TOCEntry]:
        """
        Goal: Extract TOC from OpenStax collection.xml files.
        
        This method is fully data-driven - it reads the actual modular content
        structure and extracts real chapter/section titles from the modules.
        """
        entries = []
        
        # Define namespace mappings for OpenStax files
        namespaces = {
            'col': 'http://cnx.rice.edu/collxml',
            'md': 'http://cnx.rice.edu/mdml'
        }
        
        # Extract main title
        main_title = root.find('.//md:title', namespaces)
        if main_title is not None and main_title.text:
            entries.append(TOCEntry(
                title=main_title.text.strip(),
                level=0,  # Book title level
                entry_id=f"entry_{entry_id}"
            ))
            entry_id += 1
        
        # Data-driven extraction: Read actual module content for chapter titles
        entries.extend(self._extract_openstax_modules_data_driven(root, namespaces, entry_id))
        
        return entries
    
    def _extract_openstax_modules_data_driven(self, root, namespaces: dict, entry_id: int) -> List[TOCEntry]:
        """
        Goal: Data-driven extraction of OpenStax module content with FULL recursive depth.
        
        Now supports up to 10 levels of nesting to capture all possible structures.
        """
        entries = []
        
        # Get the path to collection.xml from the file_path context
        collection_path = getattr(self, '_current_file_path', None)
        if not collection_path:
            logger.warning("Collection path not available for module extraction")
            return self._extract_openstax_fallback_structure(root, namespaces, entry_id)
        
        modules_dir = collection_path.parent.parent / "modules"
        if not modules_dir.exists():
            logger.warning(f"Modules directory not found: {modules_dir}")
            return self._extract_openstax_fallback_structure(root, namespaces, entry_id)
        
        # Start recursive extraction from root
        entry_id_ref = [entry_id]  # Use list for mutable reference
        self._extract_subcollections_recursive(root, namespaces, modules_dir, entries, entry_id_ref, level=1, parent_id=None)
        
        # If no subcollections found, process modules directly
        if len(entries) == 0:
            modules = root.findall('.//col:module', namespaces)
            for module in modules:
                document_id = module.get('document')
                if document_id:
                    module_title = self._read_module_content(modules_dir / document_id / "index.cnxml")
                    if module_title:
                        module_entry = TOCEntry(
                            title=module_title,
                            level=1,
                            entry_id=f"entry_{entry_id_ref[0]}"
                        )
                        entries.append(module_entry)
                        entry_id_ref[0] += 1
        
        return entries
    
    def _extract_subcollections_recursive(self, element, namespaces: dict, modules_dir: Path, 
                                        entries: List, entry_id_ref: List[int], level: int, 
                                        parent_id: Optional[str], max_depth: int = 10):
        """
        Goal: Recursively extract subcollections and modules at ANY depth.
        
        This replaces the previous hardcoded 3-level extraction with unlimited recursion.
        """
        if level > max_depth:  # Prevent infinite recursion
            return
        
        # Find direct subcollections at this level
        if level == 1:
            # Top level - look for subcollections directly under root content
            subcollections = element.findall('./col:content/col:subcollection', namespaces)
        else:
            # Nested level - look for subcollections directly under this element
            subcollections = element.findall('./col:content/col:subcollection', namespaces)
        
        for subcoll in subcollections:
            # Extract title for this subcollection
            title_elem = subcoll.find('./md:title', namespaces)
            if title_elem is not None and title_elem.text:
                # Create entry for this subcollection
                subcoll_entry = TOCEntry(
                    title=title_elem.text.strip(),
                    level=level,
                    parent_id=parent_id,
                    entry_id=f"entry_{entry_id_ref[0]}"
                )
                entries.append(subcoll_entry)
                current_entry_id = subcoll_entry.entry_id
                entry_id_ref[0] += 1
                
                # Process direct modules in this subcollection
                direct_modules = subcoll.findall('./col:content/col:module', namespaces)
                for module in direct_modules:
                    document_id = module.get('document')
                    if document_id:
                        module_title = self._read_module_content(modules_dir / document_id / "index.cnxml")
                        if module_title:
                            # Process internal module structure for even deeper levels
                            module_internal_levels = self._extract_module_internal_structure(
                                modules_dir / document_id / "index.cnxml"
                            )
                            
                            if module_internal_levels:
                                # Add module with its internal structure
                                for internal_level, internal_titles in module_internal_levels.items():
                                    for internal_title in internal_titles:
                                        module_entry = TOCEntry(
                                            title=internal_title,
                                            level=level + 1 + internal_level,
                                            parent_id=current_entry_id,
                                            entry_id=f"entry_{entry_id_ref[0]}"
                                        )
                                        entries.append(module_entry)
                                        entry_id_ref[0] += 1
                            else:
                                # Standard module entry
                                module_entry = TOCEntry(
                                    title=module_title,
                                    level=level + 1,
                                    parent_id=current_entry_id,
                                    entry_id=f"entry_{entry_id_ref[0]}"
                                )
                                entries.append(module_entry)
                                entry_id_ref[0] += 1
                
                # RECURSIVE CALL: Process nested subcollections
                self._extract_subcollections_recursive(
                    subcoll, namespaces, modules_dir, entries, entry_id_ref, 
                    level + 1, current_entry_id, max_depth
                )
    
    def _extract_module_internal_structure(self, module_path: Path) -> Optional[Dict[int, List[str]]]:
        """
        Goal: Extract internal hierarchical structure from within a module.
        
        This captures section/subsection structure within individual modules.
        """
        if not module_path.exists():
            return None
        
        try:
            tree = ET.parse(module_path)
            root = tree.getroot()
            
            namespaces = {
                'cnxml': 'http://cnx.rice.edu/cnxml',
                'md': 'http://cnx.rice.edu/mdml'
            }
            
            internal_structure = {}
            
            # Level 0: Main sections
            sections = root.findall('.//cnxml:section', namespaces)
            if sections:
                internal_structure[0] = []
                for section in sections:
                    title_elem = section.find('./cnxml:title', namespaces)
                    if title_elem is not None and title_elem.text:
                        internal_structure[0].append(title_elem.text.strip())
                        
                        # Level 1: Subsections within sections
                        subsections = section.findall('./cnxml:section', namespaces)
                        if subsections:
                            if 1 not in internal_structure:
                                internal_structure[1] = []
                            for subsection in subsections:
                                sub_title_elem = subsection.find('./cnxml:title', namespaces)
                                if sub_title_elem is not None and sub_title_elem.text:
                                    internal_structure[1].append(sub_title_elem.text.strip())
                                    
                                    # Level 2: Sub-subsections
                                    subsubsections = subsection.findall('./cnxml:section', namespaces)
                                    if subsubsections:
                                        if 2 not in internal_structure:
                                            internal_structure[2] = []
                                        for subsubsection in subsubsections:
                                            subsub_title_elem = subsubsection.find('./cnxml:title', namespaces)
                                            if subsub_title_elem is not None and subsub_title_elem.text:
                                                internal_structure[2].append(subsub_title_elem.text.strip())
            
            return internal_structure if internal_structure else None
            
        except Exception as e:
            logger.debug(f"Error extracting module internal structure {module_path}: {e}")
            return None
    
    def _read_module_content(self, module_path: Path) -> Optional[str]:
        """
        Goal: Read actual module content to extract title.
        
        Data-driven approach that reads the CNXML file and extracts the real title.
        """
        if not module_path.exists():
            return None
        
        try:
            tree = ET.parse(module_path)
            root = tree.getroot()
            
            # Define CNXML namespaces
            namespaces = {
                'md': 'http://cnx.rice.edu/mdml',
                'cnxml': 'http://cnx.rice.edu/cnxml'
            }
            
            # Try to find the title in metadata
            title_elem = root.find('.//md:title', namespaces)
            if title_elem is not None and title_elem.text:
                return title_elem.text.strip()
            
            # Fallback: look for h1 or title elements in content
            content_title = root.find('.//cnxml:title', namespaces)
            if content_title is not None and content_title.text:
                return content_title.text.strip()
            
            # Last resort: look for any title-like element
            for elem in root.iter():
                if elem.tag.endswith('title') and elem.text and elem.text.strip():
                    return elem.text.strip()
                    
        except Exception as e:
            logger.debug(f"Error reading module {module_path}: {e}")
            
        return None
    
    def _extract_openstax_fallback_structure(self, root, namespaces: dict, entry_id: int) -> List[TOCEntry]:
        """
        Goal: Fallback method when modules directory is not accessible.
        """
        entries = []
        
        # Extract subcollections (chapters) using metadata only
        subcollections = root.findall('.//col:subcollection', namespaces)
        
        for subcoll in subcollections:
            chapter_title = subcoll.find('./md:title', namespaces)
            if chapter_title is not None and chapter_title.text:
                chapter_entry = TOCEntry(
                    title=chapter_title.text.strip(),
                    level=1,
                    entry_id=f"entry_{entry_id}"
                )
                entries.append(chapter_entry)
                entry_id += 1
                
                # Count modules for reference
                modules = subcoll.findall('.//col:module', namespaces)
                if modules:
                    for i, module in enumerate(modules[:5]):  # Limit for fallback
                        document_id = module.get('document', f'module_{i+1}')
                        module_entry = TOCEntry(
                            title=f"Section {i+1} (Module {document_id})",
                            level=2,
                            parent_id=chapter_entry.entry_id,
                            entry_id=f"entry_{entry_id}"
                        )
                        entries.append(module_entry)
                        entry_id += 1
        
        return entries
    
    def _extract_generic_xml_toc(self, root, entry_id: int) -> List[TOCEntry]:
        """
        Goal: Extract TOC from generic XML files.
        """
        entries = []
        
        # Common XML TOC elements
        toc_elements = [
            './/title',
            './/chapter',
            './/section', 
            './/subsection',
            './/h1',
            './/h2', 
            './/h3',
            './/h4',
            './/h5',
            './/h6'
        ]
        
        for element_path in toc_elements:
            elements = root.findall(element_path)
            
            for elem in elements:
                if elem.text and elem.text.strip():
                    title = elem.text.strip()
                    
                    # Determine level based on element type
                    level = self._get_xml_element_level(elem.tag, element_path)
                    
                    entry = TOCEntry(
                        title=title,
                        level=level,
                        entry_id=f"entry_{entry_id}"
                    )
                    entries.append(entry)
                    entry_id += 1
        
        # Remove duplicates while preserving order
        seen_titles = set()
        unique_entries = []
        for entry in entries:
            if entry.title not in seen_titles:
                seen_titles.add(entry.title)
                unique_entries.append(entry)
        
        return unique_entries
    
    def _get_xml_element_level(self, tag: str, element_path: str) -> int:
        """
        Goal: Determine hierarchical level for XML elements.
        """
        level_map = {
            'title': 1,
            'chapter': 1,
            'section': 2,
            'subsection': 3,
            'h1': 1,
            'h2': 2,
            'h3': 3,
            'h4': 4,
            'h5': 5,
            'h6': 6
        }
        
        return level_map.get(tag.lower(), 2)
    
    def _extract_markdown_toc(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract table of contents from Markdown files.
        
        Parses Markdown heading syntax to build hierarchical TOC.
        """
        entries = []
        entry_id = 1
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract Markdown headings
            heading_pattern = r'^(#{1,6})\s+(.+)$'
            matches = re.findall(heading_pattern, content, re.MULTILINE)
            
            for hash_marks, title in matches:
                level = len(hash_marks)
                
                entry = TOCEntry(
                    title=title.strip(),
                    level=level,
                    entry_id=f"entry_{entry_id}"
                )
                entries.append(entry)
                entry_id += 1
            
            return entries
            
        except Exception as e:
            logger.error(f"Error extracting Markdown TOC: {e}")
            return []
    
    def _extract_rst_toc(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract table of contents from reStructuredText files.
        
        Parses RST heading syntax using underline characters.
        """
        entries = []
        entry_id = 1
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i > 0 and len(line.strip()) > 0:
                    char = line.strip()[0]
                    
                    # Check if line is all the same character (RST heading underline)
                    if char in '=-~^"' and all(c == char for c in line.strip()):
                        # Previous line should be the heading
                        if i - 1 >= 0 and len(line.strip()) >= len(lines[i-1].strip()):
                            title = lines[i-1].strip()
                            
                            if title:
                                level = self._get_rst_heading_level(char)
                                
                                entry = TOCEntry(
                                    title=title,
                                    level=level,
                                    entry_id=f"entry_{entry_id}"
                                )
                                entries.append(entry)
                                entry_id += 1
            
            return entries
            
        except Exception as e:
            logger.error(f"Error extracting RST TOC: {e}")
            return []
    
    def _get_rst_heading_level(self, char: str) -> int:
        """
        Goal: Map RST underline characters to heading levels.
        """
        level_map = {'=': 1, '-': 2, '~': 3, '^': 4, '"': 5}
        return level_map.get(char, 3)
    
    def _extract_latex_toc(self, file_path: Path) -> List[TOCEntry]:
        """
        Goal: Extract table of contents from LaTeX files.
        
        Parses LaTeX sectioning commands to build hierarchical TOC.
        """
        entries = []
        entry_id = 1
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # LaTeX sectioning commands
            latex_patterns = [
                (r'\\part\*?\{([^}]+)\}', 0),
                (r'\\chapter\*?\{([^}]+)\}', 1),
                (r'\\section\*?\{([^}]+)\}', 2),
                (r'\\subsection\*?\{([^}]+)\}', 3),
                (r'\\subsubsection\*?\{([^}]+)\}', 4),
                (r'\\paragraph\*?\{([^}]+)\}', 5),
                (r'\\subparagraph\*?\{([^}]+)\}', 6)
            ]
            
            for pattern, level in latex_patterns:
                matches = re.findall(pattern, content)
                
                for title in matches:
                    entry = TOCEntry(
                        title=title.strip(),
                        level=level + 1,  # Adjust to 1-based indexing
                        entry_id=f"entry_{entry_id}"
                    )
                    entries.append(entry)
                    entry_id += 1
            
            # Sort entries by appearance order in document
            entries.sort(key=lambda x: content.find(x.title))
            
            return entries
            
        except Exception as e:
            logger.error(f"Error extracting LaTeX TOC: {e}")
            return []