"""
Advanced text extraction system for multiple textbook formats.

This module provides robust text extraction capabilities that preserve mathematical
notation, formatting, and structure across PDF, EPUB, XML, LaTeX, and other formats.

Goal: Extract high-quality searchable text while preserving educational content integrity.
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass
import zipfile
import tempfile
import shutil
import time

from .config import OpenBooksConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Represents extracted content from a textbook."""
    source_path: str
    format_type: str  # 'pdf', 'epub', 'xml', 'latex', 'cnxml', 'markdown'
    title: str
    authors: List[str]
    chapters: List[Dict[str, Any]]
    raw_text: str
    mathematical_notation: List[Dict[str, str]]
    images: List[Dict[str, str]]
    metadata: Dict[str, Any]
    extraction_stats: Dict[str, Any]
    content_hash: str


@dataclass 
class Chapter:
    """Represents a chapter or section within a textbook."""
    number: str
    title: str
    content: str
    subsections: List[Dict[str, str]]
    formulas: List[str]
    figures: List[Dict[str, str]]
    exercises: List[Dict[str, str]]


class TextExtractor:
    """
    Advanced text extraction system supporting multiple formats.
    
    Supports:
    - PDF files with mathematical notation preservation
    - EPUB files with structure extraction
    - OpenStax CNX XML files
    - LaTeX source files
    - Markdown files
    - HTML content
    """
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize text extractor with configuration."""
        self.config = config
        self.extraction_cache = {}
        
        # Initialize format-specific extractors
        self._init_pdf_extractor()
        self._init_epub_extractor()
        self._init_xml_extractor()
        
        logger.info("TextExtractor initialized with support for multiple formats")
    
    def extract_content(self, source_path: str) -> Optional[ExtractedContent]:
        """
        Extract content from a textbook file or directory.
        
        Args:
            source_path: Path to textbook file or directory
            
        Returns:
            ExtractedContent object or None if extraction fails
        """
        source_path = Path(source_path)
        
        # Check cache first
        cache_key = self._get_cache_key(source_path)
        if cache_key in self.extraction_cache:
            logger.debug(f"Using cached extraction for {source_path}")
            return self.extraction_cache[cache_key]
        
        # Determine format and extract
        format_type = self._detect_format(source_path)
        logger.info(f"Extracting content from {source_path} (format: {format_type})")
        
        try:
            if format_type == 'pdf':
                content = self._extract_pdf(source_path)
            elif format_type == 'epub':
                content = self._extract_epub(source_path)
            elif format_type == 'cnxml':
                content = self._extract_cnxml_directory(source_path)
            elif format_type == 'xml':
                content = self._extract_xml(source_path)
            elif format_type == 'latex':
                content = self._extract_latex(source_path)
            elif format_type == 'markdown':
                content = self._extract_markdown(source_path)
            elif format_type == 'html':
                content = self._extract_html(source_path)
            else:
                logger.warning(f"Unsupported format: {format_type} for {source_path}")
                return None
            
            # Cache the result
            if content:
                self.extraction_cache[cache_key] = content
                logger.info(f"Successfully extracted {len(content.raw_text)} characters from {source_path}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {source_path}: {e}")
            return None
    
    def _detect_format(self, source_path: Path) -> str:
        """Detect the format of a textbook file or directory."""
        if source_path.is_file():
            suffix = source_path.suffix.lower()
            if suffix == '.pdf':
                return 'pdf'
            elif suffix == '.epub':
                return 'epub'
            elif suffix in ['.xml', '.cnxml']:
                return 'xml'
            elif suffix in ['.tex', '.latex']:
                return 'latex'
            elif suffix in ['.md', '.markdown']:
                return 'markdown'
            elif suffix in ['.html', '.htm']:
                return 'html'
        
        elif source_path.is_dir():
            # Check for git repository (common case for OpenStax books)
            if (source_path / '.git').exists():
                # Check for CNX directory structure
                if (source_path / 'collections').exists() or (source_path / 'modules').exists():
                    return 'cnxml'
                # Check for OpenStax repository structure
                elif (source_path / 'media').exists() or any(source_path.glob('**/*.cnxml')):
                    return 'cnxml'
                # Check for LaTeX project
                elif any(source_path.glob('**/*.tex')):
                    return 'latex'
                # Check for HTML/web content
                elif (source_path / 'index.html').exists() or any(source_path.glob('**/*.html')):
                    return 'html'
                # Check for Markdown documentation
                elif any(source_path.glob('**/*.md')):
                    return 'markdown'
                # Default to cnxml for OpenStax-style repositories
                else:
                    return 'cnxml'
            # Check for CNX directory structure
            elif (source_path / 'collections').exists() or (source_path / 'modules').exists():
                return 'cnxml'
            # Check for LaTeX project
            elif any(source_path.glob('*.tex')):
                return 'latex'
            # Check for HTML/web content
            elif (source_path / 'index.html').exists():
                return 'html'
        
        return 'unknown'
    
    def _init_pdf_extractor(self) -> None:
        """Initialize PDF extraction capabilities."""
        try:
            import fitz  # PyMuPDF
            self.pdf_available = True
            logger.debug("PDF extraction available (PyMuPDF)")
        except ImportError:
            self.pdf_available = False
            logger.warning("PDF extraction not available - install PyMuPDF for PDF support")
    
    def _init_epub_extractor(self) -> None:
        """Initialize EPUB extraction capabilities."""
        try:
            import ebooklib
            from ebooklib import epub
            self.epub_available = True
            logger.debug("EPUB extraction available")
        except ImportError:
            self.epub_available = False
            logger.warning("EPUB extraction not available - install ebooklib for EPUB support")
    
    def _init_xml_extractor(self) -> None:
        """Initialize XML extraction capabilities."""
        # XML parsing is built into Python
        self.xml_available = True
        logger.debug("XML extraction available")
    
    def _extract_pdf(self, pdf_path: Path) -> Optional[ExtractedContent]:
        """Extract content from PDF file with mathematical notation preservation."""
        if not self.pdf_available:
            logger.error("PDF extraction not available")
            return None
        
        try:
            import fitz
            
            doc = fitz.open(str(pdf_path))
            chapters = []
            raw_text = ""
            mathematical_notation = []
            images = []
            
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', '')
            }
            
            current_chapter = None
            chapter_number = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with formatting
                text = page.get_text()
                raw_text += text + "\n"
                
                # Detect chapter breaks
                chapter_match = re.search(r'^(Chapter|CHAPTER)\s+(\d+)', text, re.MULTILINE)
                if chapter_match:
                    # Save previous chapter
                    if current_chapter:
                        chapters.append(current_chapter)
                    
                    # Start new chapter
                    chapter_number += 1
                    chapter_title = self._extract_chapter_title(text)
                    current_chapter = {
                        'number': chapter_match.group(2),
                        'title': chapter_title,
                        'content': text,
                        'page_start': page_num + 1,
                        'subsections': [],
                        'formulas': self._extract_formulas(text),
                        'figures': []
                    }
                elif current_chapter:
                    current_chapter['content'] += "\n" + text
                    current_chapter['formulas'].extend(self._extract_formulas(text))
                
                # Extract mathematical notation
                math_elements = self._extract_math_from_text(text)
                mathematical_notation.extend(math_elements)
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n < 5:  # Skip CMYK images
                            img_data = {
                                'page': page_num + 1,
                                'index': img_index,
                                'width': pix.width,
                                'height': pix.height,
                                'format': 'png'
                            }
                            images.append(img_data)
                        pix = None
                    except Exception as e:
                        logger.debug(f"Error extracting image on page {page_num}: {e}")
            
            # Add final chapter
            if current_chapter:
                chapters.append(current_chapter)
            
            doc.close()
            
            # Generate content hash
            content_hash = hashlib.md5(raw_text.encode()).hexdigest()
            
            extraction_stats = {
                'pages_processed': doc.page_count,
                'chapters_found': len(chapters),
                'formulas_found': len(mathematical_notation),
                'images_found': len(images),
                'text_length': len(raw_text)
            }
            
            return ExtractedContent(
                source_path=str(pdf_path),
                format_type='pdf',
                title=metadata.get('title', pdf_path.stem),
                authors=[metadata.get('author', '')] if metadata.get('author') else [],
                chapters=chapters,
                raw_text=raw_text,
                mathematical_notation=mathematical_notation,
                images=images,
                metadata=metadata,
                extraction_stats=extraction_stats,
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return None
    
    def _extract_cnxml_directory(self, cnxml_dir: Path) -> Optional[ExtractedContent]:
        """Extract content from OpenStax CNX directory structure."""
        try:
            # Find the collection XML file
            collections_dir = cnxml_dir / 'collections'
            collection_file = None
            
            if collections_dir.exists():
                collection_files = list(collections_dir.glob('*.xml'))
                if collection_files:
                    collection_file = collection_files[0]
            
            if not collection_file:
                logger.warning(f"No collection file found in {cnxml_dir}")
                return None
            
            # Parse collection structure
            tree = ET.parse(collection_file)
            root = tree.getroot()
            
            # Extract metadata
            title = ""
            authors = []
            
            # Find title
            title_elem = root.find('.//{http://cnx.rice.edu/collxml}title')
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()
            
            # Find authors
            for author_elem in root.findall('.//{http://cnx.rice.edu/collxml}person'):
                name_elem = author_elem.find('{http://cnx.rice.edu/collxml}name')
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            
            # Extract chapter structure
            chapters = []
            raw_text = ""
            
            for chapter_elem in root.findall('.//{http://cnx.rice.edu/collxml}subcollection'):
                # Look for title in both md and col namespaces
                chapter_title_elem = chapter_elem.find('{http://cnx.rice.edu/mdml}title')
                if chapter_title_elem is None:
                    chapter_title_elem = chapter_elem.find('{http://cnx.rice.edu/collxml}title')
                
                chapter_title = chapter_title_elem.text if chapter_title_elem is not None else f"Chapter {len(chapters) + 1}"
                
                chapter_content = ""
                subsections = []
                
                # Process modules in chapter
                for module_elem in chapter_elem.findall('.//{http://cnx.rice.edu/collxml}module'):
                    module_id = module_elem.get('document')
                    if module_id:
                        module_content = self._extract_cnxml_module(cnxml_dir, module_id)
                        if module_content:
                            chapter_content += module_content + "\n\n"
                            subsections.append({
                                'id': module_id,
                                'content': module_content[:500] + "..." if len(module_content) > 500 else module_content
                            })
                
                if chapter_content:
                    chapters.append({
                        'number': str(len(chapters) + 1),
                        'title': chapter_title,
                        'content': chapter_content,
                        'subsections': subsections,
                        'formulas': self._extract_formulas(chapter_content),
                        'figures': []
                    })
                    raw_text += chapter_content + "\n\n"
            
            # Generate content hash
            content_hash = hashlib.md5(raw_text.encode()).hexdigest()
            
            extraction_stats = {
                'chapters_found': len(chapters),
                'text_length': len(raw_text),
                'modules_processed': sum(len(ch['subsections']) for ch in chapters)
            }
            
            return ExtractedContent(
                source_path=str(cnxml_dir),
                format_type='cnxml',
                title=title or cnxml_dir.name,
                authors=authors,
                chapters=chapters,
                raw_text=raw_text,
                mathematical_notation=self._extract_math_from_text(raw_text),
                images=[],
                metadata={'collection_file': str(collection_file)},
                extraction_stats=extraction_stats,
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting CNXML directory {cnxml_dir}: {e}")
            return None
    
    def _extract_cnxml_module(self, cnxml_dir: Path, module_id: str) -> Optional[str]:
        """Extract text content from a CNXML module."""
        try:
            # Look for module file in various possible locations
            module_file = cnxml_dir / f"{module_id}.cnxml"
            if not module_file.exists():
                # Try in modules subdirectory (flat structure)
                modules_dir = cnxml_dir / 'modules'
                if modules_dir.exists():
                    module_file = modules_dir / f"{module_id}.cnxml"
                    
                    if not module_file.exists():
                        # Try in modules/{module_id}/index.cnxml (OpenStax standard)
                        module_file = modules_dir / module_id / "index.cnxml"
            
            if not module_file.exists():
                logger.debug(f"Module file not found: {module_id}")
                return None
            
            # Parse CNXML content
            tree = ET.parse(module_file)
            root = tree.getroot()
            
            # Extract text content, removing XML tags
            content = ET.tostring(root, encoding='unicode', method='text')
            
            # Clean up the text
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
            
        except Exception as e:
            logger.debug(f"Error extracting module {module_id}: {e}")
            return None
    
    def _extract_epub(self, epub_path: Path) -> Optional[ExtractedContent]:
        """Extract content from EPUB file."""
        if not self.epub_available:
            logger.error("EPUB extraction not available")
            return None
        
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(str(epub_path))
            
            # Extract metadata
            title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else epub_path.stem
            authors = [author[0] for author in book.get_metadata('DC', 'creator')]
            
            chapters = []
            raw_text = ""
            
            # Extract content from each item
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text()
                    
                    # Clean up text
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if text:
                        chapter = {
                            'number': str(len(chapters) + 1),
                            'title': item.get_name(),
                            'content': text,
                            'subsections': [],
                            'formulas': self._extract_formulas(text),
                            'figures': []
                        }
                        chapters.append(chapter)
                        raw_text += text + "\n\n"
            
            # Generate content hash
            content_hash = hashlib.md5(raw_text.encode()).hexdigest()
            
            extraction_stats = {
                'chapters_found': len(chapters),
                'text_length': len(raw_text)
            }
            
            return ExtractedContent(
                source_path=str(epub_path),
                format_type='epub',
                title=title,
                authors=authors,
                chapters=chapters,
                raw_text=raw_text,
                mathematical_notation=self._extract_math_from_text(raw_text),
                images=[],
                metadata={},
                extraction_stats=extraction_stats,
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting EPUB {epub_path}: {e}")
            return None
    
    def _extract_xml(self, xml_path: Path) -> Optional[ExtractedContent]:
        """Extract content from XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract text content
            raw_text = ET.tostring(root, encoding='unicode', method='text')
            raw_text = re.sub(r'\s+', ' ', raw_text).strip()
            
            # Generate content hash
            content_hash = hashlib.md5(raw_text.encode()).hexdigest()
            
            return ExtractedContent(
                source_path=str(xml_path),
                format_type='xml',
                title=xml_path.stem,
                authors=[],
                chapters=[{
                    'number': '1',
                    'title': 'Content',
                    'content': raw_text,
                    'subsections': [],
                    'formulas': self._extract_formulas(raw_text),
                    'figures': []
                }],
                raw_text=raw_text,
                mathematical_notation=self._extract_math_from_text(raw_text),
                images=[],
                metadata={},
                extraction_stats={'text_length': len(raw_text)},
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting XML {xml_path}: {e}")
            return None
    
    def _extract_latex(self, latex_path: Path) -> Optional[ExtractedContent]:
        """Extract content from LaTeX file or directory."""
        try:
            if latex_path.is_file():
                content = latex_path.read_text(encoding='utf-8')
            else:
                # Find main LaTeX file
                main_files = list(latex_path.glob('*.tex'))
                if not main_files:
                    return None
                content = main_files[0].read_text(encoding='utf-8')
            
            # Remove LaTeX commands but preserve math
            text = self._clean_latex_text(content)
            
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            return ExtractedContent(
                source_path=str(latex_path),
                format_type='latex',
                title=latex_path.name,
                authors=[],
                chapters=[{
                    'number': '1',
                    'title': 'Content',
                    'content': text,
                    'subsections': [],
                    'formulas': self._extract_latex_math(content),
                    'figures': []
                }],
                raw_text=text,
                mathematical_notation=self._extract_math_from_text(text),
                images=[],
                metadata={},
                extraction_stats={'text_length': len(text)},
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting LaTeX {latex_path}: {e}")
            return None
    
    def _extract_markdown(self, md_path: Path) -> Optional[ExtractedContent]:
        """Extract content from Markdown file."""
        try:
            content = md_path.read_text(encoding='utf-8')
            
            # Remove markdown formatting
            import re
            text = re.sub(r'[#*`_\[\]()]', '', content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            return ExtractedContent(
                source_path=str(md_path),
                format_type='markdown',
                title=md_path.stem,
                authors=[],
                chapters=[{
                    'number': '1',
                    'title': 'Content',
                    'content': text,
                    'subsections': [],
                    'formulas': self._extract_formulas(text),
                    'figures': []
                }],
                raw_text=text,
                mathematical_notation=self._extract_math_from_text(text),
                images=[],
                metadata={},
                extraction_stats={'text_length': len(text)},
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting Markdown {md_path}: {e}")
            return None
    
    def _extract_html(self, html_path: Path) -> Optional[ExtractedContent]:
        """Extract content from HTML file or directory."""
        try:
            if html_path.is_file():
                content = html_path.read_text(encoding='utf-8')
            else:
                # Find index.html
                index_file = html_path / 'index.html'
                if index_file.exists():
                    content = index_file.read_text(encoding='utf-8')
                else:
                    return None
            
            # Parse HTML and extract text
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
            except ImportError:
                # Fallback: remove HTML tags with regex
                text = re.sub(r'<[^>]+>', '', content)
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            return ExtractedContent(
                source_path=str(html_path),
                format_type='html',
                title=html_path.name,
                authors=[],
                chapters=[{
                    'number': '1',
                    'title': 'Content',
                    'content': text,
                    'subsections': [],
                    'formulas': self._extract_formulas(text),
                    'figures': []
                }],
                raw_text=text,
                mathematical_notation=self._extract_math_from_text(text),
                images=[],
                metadata={},
                extraction_stats={'text_length': len(text)},
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error extracting HTML {html_path}: {e}")
            return None
    
    def _extract_chapter_title(self, text: str) -> str:
        """Extract chapter title from text using multiple patterns."""
        lines = text.split('\n')
        
        # Try multiple patterns to extract meaningful titles
        patterns = [
            r'Chapter\s+\d+[:\s]+(.+)',  # Chapter 1: Title
            r'(\w+(?:\s+\w+){1,5})\s*\n',  # First meaningful line
            r'^([A-Z][^.!?]*[.!?]?)\s*$',  # Capitalized sentence
            r'Section\s+[\d.]+[:\s]+(.+)',  # Section 1.1: Title
            r'^(.{10,80})\s*$'  # Any reasonable length line
        ]
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line or len(line) < 5:
                continue
                
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Filter out common non-title content
                    if not any(skip in title.lower() for skip in ['teacher support', 'learning objectives', 'by the end']):
                        if len(title) > 5 and len(title) < 100:
                            return title
        
        # Fallback: use first substantial line
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                return line
                
        return "Content Section"
    
    def _extract_formulas(self, text: str) -> List[str]:
        """Extract mathematical formulas from text."""
        formulas = []
        
        # LaTeX math patterns
        latex_patterns = [
            r'\$\$([^$]+)\$\$',  # Display math
            r'\$([^$]+)\$',      # Inline math
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}',
            r'\\begin\{math\}(.*?)\\end\{math\}'
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            formulas.extend(matches)
        
        # MathML patterns
        mathml_pattern = r'<math[^>]*>(.*?)</math>'
        mathml_matches = re.findall(mathml_pattern, text, re.DOTALL)
        formulas.extend(mathml_matches)
        
        return [f.strip() for f in formulas if f.strip()]
    
    def _extract_math_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract mathematical notation with context."""
        math_elements = []
        formulas = self._extract_formulas(text)
        
        for i, formula in enumerate(formulas):
            math_elements.append({
                'id': f"math_{i}",
                'content': formula,
                'type': 'formula',
                'context': self._get_formula_context(text, formula)
            })
        
        return math_elements
    
    def _extract_latex_math(self, latex_content: str) -> List[str]:
        """Extract LaTeX mathematical expressions."""
        formulas = []
        
        # Extract equations
        equation_patterns = [
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
            r'\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}',
            r'\\\[(.*?)\\\]',
            r'\$\$(.*?)\$\$'
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, latex_content, re.DOTALL)
            formulas.extend(matches)
        
        return [f.strip() for f in formulas if f.strip()]
    
    def _clean_latex_text(self, latex_content: str) -> str:
        """Clean LaTeX content to extract readable text."""
        # Remove comments
        text = re.sub(r'%.*', '', latex_content)
        
        # Remove common LaTeX commands but keep content
        text = re.sub(r'\\(?:section|subsection|subsubsection|chapter|title|author)\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\(?:textbf|textit|emph|texttt)\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\(?:label|ref|cite)\{[^}]+\}', '', text)
        
        # Remove environments but keep content
        text = re.sub(r'\\begin\{[^}]+\}', '', text)
        text = re.sub(r'\\end\{[^}]+\}', '', text)
        
        # Remove remaining LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})*', '', text)
        text = re.sub(r'[{}\\]', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_formula_context(self, text: str, formula: str) -> str:
        """Get contextual text around a formula."""
        formula_pos = text.find(formula)
        if formula_pos == -1:
            return ""
        
        # Get 100 characters before and after
        start = max(0, formula_pos - 100)
        end = min(len(text), formula_pos + len(formula) + 100)
        
        context = text[start:end].strip()
        return context
    
    def _get_cache_key(self, source_path: Path) -> str:
        """Generate cache key for extraction results."""
        try:
            stat = source_path.stat()
            key_data = f"{source_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(source_path).encode()).hexdigest()
    
    def save_extracted_content(self, content: ExtractedContent, output_dir: Path) -> Path:
        """Save extracted content to structured format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from content hash
        filename = f"{content.content_hash}.json"
        output_file = output_dir / filename
        
        # Convert to JSON-serializable format
        content_dict = {
            'source_path': content.source_path,
            'format_type': content.format_type,
            'title': content.title,
            'authors': content.authors,
            'chapters': content.chapters,
            'raw_text': content.raw_text,
            'mathematical_notation': content.mathematical_notation,
            'images': content.images,
            'metadata': content.metadata,
            'extraction_stats': content.extraction_stats,
            'content_hash': content.content_hash,
            'extracted_at': time.time()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(content_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted content to {output_file}")
        return output_file