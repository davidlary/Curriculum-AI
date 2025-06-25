#!/usr/bin/env python3
"""
Step 2: TOC Extraction Module (Simplified Version)
Extracts table of contents from discovered books.

This simplified version works with the existing file structure and focuses on
extracting TOCs from OpenStax books and other available formats.
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import time
import xml.etree.ElementTree as ET
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing core infrastructure
try:
    from core.toc_extractor import TOCExtractor, TOCEntry as CoreTOCEntry, BookTOC
    from core.config import OpenBooksConfig
    CORE_AVAILABLE = True
    logger.info("Successfully imported core TOC infrastructure")
except ImportError as e:
    logger.warning(f"Core modules not available: {e}")
    CORE_AVAILABLE = False

# Directory structure
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "Cache" / "TOCs"
OUTPUT_DIR = BASE_DIR / "TOCs"
TEMP_DIR = BASE_DIR / "temp" / "toc_extraction"

# Create directories
for dir_path in [CACHE_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class TOCEntry:
    """Simple TOC entry structure."""
    title: str
    level: int
    page_number: Optional[int] = None
    section_number: Optional[str] = None
    entry_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'level': self.level,
            'page_number': self.page_number,
            'section_number': self.section_number,
            'entry_id': self.entry_id
        }

@dataclass
class TOCExtractionResult:
    """Result of TOC extraction for a single book."""
    book_id: str
    book_title: str
    educational_level: str
    extraction_status: str
    toc_entries: List[Dict[str, Any]]
    extraction_time: float
    quality_score: float
    hierarchy_depth: int
    total_topics: int
    error_message: str = ""

@dataclass
class TOCExtractionMetrics:
    """Metrics for the TOC extraction process."""
    total_books_processed: int
    successful_extractions: int
    failed_extractions: int
    total_processing_time: float
    average_time_per_book: float
    total_toc_entries: int
    average_hierarchy_depth: float

class SimpleTOCExtractor:
    """TOC extractor that uses core infrastructure for real content extraction."""
    
    def __init__(self):
        self.cache_ttl_hours = 168  # Cache valid for 1 week
        self.cache_version = "1.0"
        
        if CORE_AVAILABLE:
            self.core_extractor = TOCExtractor()
            logger.info("SimpleTOCExtractor initialized with core infrastructure")
        else:
            self.core_extractor = None
            logger.info("SimpleTOCExtractor initialized in fallback mode")

    def extract_tocs_from_discovery(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """Extract TOCs from all books discovered in Step 1."""
        start_time = time.time()
        logger.info(f"Starting TOC extraction for {discipline} in {language}")
        
        # Load discovered books from Step 1
        books_file = BASE_DIR / "Books" / f"{discipline}_{language}_books_discovered.json"
        if not books_file.exists():
            raise FileNotFoundError(f"Books discovery file not found: {books_file}")
        
        with open(books_file, 'r', encoding='utf-8') as f:
            discovery_data = json.load(f)
        
        books = discovery_data['books']
        logger.info(f"Found {len(books)} books to process")
        
        # Extract TOCs from each book
        extraction_results = []
        for book in books:
            result = self._extract_single_toc(book)
            extraction_results.append(result)
            logger.info(f"Processed {book['title']}: {result.extraction_status}")
        
        # Analyze overlaps between books
        logger.info("Analyzing TOC overlaps between books...")
        overlap_analysis = self._analyze_toc_overlaps(extraction_results)
        
        # Print detailed analysis for each book
        for result in extraction_results:
            if result.extraction_status == 'success':
                book_title = result.book_title
                overlap_info = overlap_analysis.get(book_title, {})
                
                total_toc = len(result.toc_entries) if result.toc_entries else 0
                overlapping = overlap_info.get('overlapping_count', 0)
                unique = overlap_info.get('unique_count', 0)
                
                print(f"\n📖 {book_title}:")
                print(f"   (1) Total TOC entries: {total_toc}")
                overlap_pct = (overlapping/total_toc*100) if total_toc > 0 else 0
                unique_pct = (unique/total_toc*100) if total_toc > 0 else 0
                print(f"   (2) Overlapping with other books: {overlapping} ({overlap_pct:.1f}%)")
                print(f"   (3) Unique to this book: {unique} ({unique_pct:.1f}%)")
                
                if overlap_info.get('overlap_details'):
                    print(f"   📊 Example overlaps:")
                    for detail in overlap_info['overlap_details'][:3]:  # Show top 3
                        other_books = ', '.join([book[:30] + '...' if len(book) > 30 else book 
                                               for book in detail['overlapping_books'][:2]])  # Show up to 2 book names
                        if len(detail['overlapping_books']) > 2:
                            other_books += f" (+{len(detail['overlapping_books'])-2} more)"
                        print(f"      • '{detail['title'][:50]}...' also in: {other_books}")
        
        # Organize results by educational level
        tocs_by_level = self._organize_tocs_by_level(extraction_results)
        
        # Calculate metrics
        metrics = self._calculate_extraction_metrics(extraction_results, start_time)
        
        # Prepare result
        result = {
            'discipline': discipline,
            'language': language,
            'extraction_timestamp': datetime.now().isoformat(),
            'tocs_by_level': tocs_by_level,
            'extraction_results': [asdict(r) for r in extraction_results],
            'overlap_analysis': overlap_analysis,
            'metrics': asdict(metrics),
            'total_books_processed': len(books),
            'successful_extractions': metrics.successful_extractions
        }
        
        # Save to output directory
        output_file = OUTPUT_DIR / f"{discipline}_{language}_tocs_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"TOC extraction completed: {metrics.successful_extractions}/{len(books)} successful")
        return result

    def _analyze_toc_overlaps(self, extraction_results: List[TOCExtractionResult]) -> Dict[str, Any]:
        """Analyze overlaps between TOC entries across books."""
        
        # Collect all TOC entries with book information
        all_entries = {}  # title -> list of (book_title, entry)
        book_entries = {}  # book_title -> list of normalized titles
        
        for result in extraction_results:
            if result.extraction_status == 'success' and result.toc_entries:
                book_title = result.book_title
                book_entries[book_title] = []
                
                for entry in result.toc_entries:
                    # Normalize title for comparison (handle both dict and object formats)
                    if hasattr(entry, 'title'):
                        title = entry.title
                    elif isinstance(entry, dict):
                        title = entry.get('title', '')
                    else:
                        title = str(entry)
                    
                    normalized_title = self._normalize_toc_title(title)
                    
                    if normalized_title not in all_entries:
                        all_entries[normalized_title] = []
                    
                    all_entries[normalized_title].append((book_title, entry))
                    book_entries[book_title].append(normalized_title)
        
        # Analyze overlaps for each book
        overlap_analysis = {}
        
        for book_title, entry_titles in book_entries.items():
            overlapping_entries = []
            unique_entries = []
            overlap_details = []
            
            for title in entry_titles:
                books_with_title = [book for book, _ in all_entries[title]]
                
                if len(books_with_title) > 1:
                    # This title appears in multiple books
                    overlapping_entries.append(title)
                    other_books = [book for book in books_with_title if book != book_title]
                    
                    overlap_details.append({
                        'title': title,
                        'overlapping_books': other_books,
                        'total_occurrences': len(books_with_title)
                    })
                else:
                    # This title is unique to this book
                    unique_entries.append(title)
            
            # Sort overlap details by number of occurrences (most common first)
            overlap_details.sort(key=lambda x: x['total_occurrences'], reverse=True)
            
            overlap_analysis[book_title] = {
                'total_entries': len(entry_titles),
                'overlapping_count': len(overlapping_entries),
                'unique_count': len(unique_entries),
                'overlap_percentage': (len(overlapping_entries) / len(entry_titles)) * 100 if entry_titles else 0,
                'uniqueness_percentage': (len(unique_entries) / len(entry_titles)) * 100 if entry_titles else 0,
                'overlap_details': overlap_details[:10]  # Keep top 10 for storage
            }
        
        return overlap_analysis

    def _normalize_toc_title(self, title: str) -> str:
        """Normalize TOC title for comparison across books."""
        import re
        
        # Convert to lowercase
        normalized = title.lower().strip()
        
        # Remove common prefixes and suffixes
        normalized = re.sub(r'^(chapter\s+\d+[\.\:]?\s*)', '', normalized)
        normalized = re.sub(r'^(section\s+\d+[\.\:]?\s*)', '', normalized)
        normalized = re.sub(r'^(\d+[\.\:]?\s*)', '', normalized)
        normalized = re.sub(r'^(introduction\s+to\s+)', '', normalized)
        normalized = re.sub(r'^(basic\s+)', '', normalized)
        normalized = re.sub(r'^(advanced\s+)', '', normalized)
        
        # Remove version indicators
        normalized = re.sub(r'\s+(i{1,3}|iv|v|vi{1,3}|1|2|3|4|5)$', '', normalized)
        normalized = re.sub(r'\s+\(part\s+\d+\)', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation for better matching
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized

    def _extract_single_toc(self, book: Dict[str, Any]) -> TOCExtractionResult:
        """Extract TOC from a single book."""
        start_time = time.time()
        book_id = book['id']
        book_title = book['title']
        book_url = book['url']
        
        try:
            book_path = Path(book_url)
            
            if not book_path.exists():
                return self._create_failed_result(book, start_time, "Book path does not exist")
            
            # Determine extraction method based on book format/structure
            toc_entries = []
            
            if book['source'] == 'openstax' and book['format'] == 'xml_collection':
                # Handle individual XML collection files
                toc_entries = self._extract_xml_collection_toc(book_path)
            elif book['source'] == 'openstax' and book['format'] == 'directory':
                toc_entries = self._extract_openstax_toc(book_path)
            elif book['format'] == 'pdf':
                toc_entries = self._extract_pdf_toc(book_path)
            elif book_path.is_dir():
                toc_entries = self._extract_directory_toc(book_path)
            else:
                toc_entries = self._extract_file_toc(book_path)
            
            # Calculate quality metrics
            quality_score = self._calculate_toc_quality(toc_entries)
            hierarchy_depth = self._calculate_hierarchy_depth(toc_entries)
            
            return TOCExtractionResult(
                book_id=book_id,
                book_title=book_title,
                educational_level=book['educational_level'],
                extraction_status='success',
                toc_entries=[entry.to_dict() for entry in toc_entries],
                extraction_time=time.time() - start_time,
                quality_score=quality_score,
                hierarchy_depth=hierarchy_depth,
                total_topics=len(toc_entries)
            )
            
        except Exception as e:
            logger.error(f"Error extracting TOC from {book_title}: {e}")
            return self._create_failed_result(book, start_time, str(e))

    def _extract_xml_collection_toc(self, xml_file: Path) -> List[TOCEntry]:
        """Extract TOC from a single XML collection file using core infrastructure."""
        toc_entries = []
        
        if self.core_extractor and xml_file.exists():
            logger.info(f"Extracting TOC from XML collection: {xml_file}")
            
            try:
                # Use core TOCExtractor with proper parameters
                book_toc = self.core_extractor.extract_toc(
                    xml_file, 
                    language="English",
                    discipline="Physics",  
                    level="Unknown"
                )
                
                if book_toc and book_toc.entries:
                    # Convert core TOCEntry objects to our simplified format
                    for core_entry in book_toc.entries:
                        simple_entry = TOCEntry(
                            title=core_entry.title,
                            level=core_entry.level,
                            page_number=core_entry.page_number,
                            section_number=core_entry.section_number,
                            entry_id=core_entry.entry_id or f"entry_{len(toc_entries)+1}"
                        )
                        toc_entries.append(simple_entry)
                    
                    logger.info(f"Successfully extracted {len(toc_entries)} TOC entries from {xml_file}")
                else:
                    logger.warning(f"No TOC entries found in {xml_file}")
                    
            except Exception as e:
                logger.error(f"Error using core extractor for {xml_file}: {e}")
                # Fallback to manual parsing if core extractor fails
                toc_entries = self._parse_openstax_xml_fallback(xml_file)
        else:
            # Fallback method if core extractor not available
            logger.warning("Core extractor not available, using fallback method")
            toc_entries = self._parse_openstax_xml_fallback(xml_file)
        
        return toc_entries

    def _extract_openstax_toc(self, book_path: Path) -> List[TOCEntry]:
        """Extract TOC from OpenStax book directory using core infrastructure."""
        toc_entries = []
        
        if self.core_extractor:
            # Use core infrastructure for proper extraction
            collections_dir = book_path / "collections"
            if collections_dir.exists():
                for xml_file in collections_dir.glob("*.xml"):
                    logger.info(f"Extracting TOC from {xml_file} using core infrastructure")
                    
                    # Use core TOCExtractor with proper parameters
                    book_toc = self.core_extractor.extract_toc(
                        xml_file, 
                        language="English",  # Default, could be parameterized
                        discipline="Physics",  # Default, could be parameterized  
                        level="Unknown"  # Will be determined by book structure
                    )
                    
                    if book_toc and book_toc.entries:
                        # Convert core TOCEntry objects to our simplified format
                        for core_entry in book_toc.entries:
                            simple_entry = TOCEntry(
                                title=core_entry.title,
                                level=core_entry.level,
                                page_number=core_entry.page_number,
                                section_number=core_entry.section_number,
                                entry_id=core_entry.entry_id
                            )
                            toc_entries.append(simple_entry)
                        
                        logger.info(f"Successfully extracted {len(toc_entries)} TOC entries using core infrastructure")
                        break  # Use first successful extraction
        
        # Fallback to basic extraction if core fails
        if not toc_entries:
            logger.warning("Core extraction failed, using fallback method")
            collections_dir = book_path / "collections"
            if collections_dir.exists():
                for xml_file in collections_dir.glob("*.xml"):
                    toc_entries.extend(self._parse_openstax_xml_fallback(xml_file))
            
            # If no XML found, try to extract from directory structure
            if not toc_entries:
                modules_dir = book_path / "modules"
                if modules_dir.exists():
                    toc_entries = self._extract_from_modules_directory(modules_dir)
        
        # If still no TOC, create basic structure from README
        if not toc_entries:
            readme_file = book_path / "README.md"
            if readme_file.exists():
                toc_entries = self._extract_from_readme(readme_file)
        
        return toc_entries

    def _parse_openstax_xml_fallback(self, xml_file: Path) -> List[TOCEntry]:
        """Parse OpenStax collection XML file for TOC."""
        toc_entries = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Remove namespace prefixes for easier parsing
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]
            
            # Find all content elements
            content_elements = root.findall('.//content')
            subcollection_elements = root.findall('.//subcollection')
            
            entry_id = 0
            
            # Process subcollections (chapters)
            for subcoll in subcollection_elements:
                entry_id += 1
                title = subcoll.get('title', f'Chapter {entry_id}')
                
                entry = TOCEntry(
                    title=title,
                    level=1,
                    entry_id=f"chapter_{entry_id}"
                )
                toc_entries.append(entry)
                
                # Process content within subcollection (sections)
                sub_content = subcoll.findall('.//content')
                for i, content in enumerate(sub_content, 1):
                    content_title = content.get('title', f'Section {entry_id}.{i}')
                    
                    sub_entry = TOCEntry(
                        title=content_title,
                        level=2,
                        section_number=f"{entry_id}.{i}",
                        entry_id=f"section_{entry_id}_{i}"
                    )
                    toc_entries.append(sub_entry)
            
            # Process standalone content elements
            for i, content in enumerate(content_elements, 1):
                if content.getparent().tag != 'subcollection':  # Not already processed
                    title = content.get('title', f'Topic {i}')
                    
                    entry = TOCEntry(
                        title=title,
                        level=1,
                        entry_id=f"topic_{i}"
                    )
                    toc_entries.append(entry)
                    
        except Exception as e:
            logger.error(f"Error parsing XML {xml_file}: {e}")
        
        return toc_entries

    def _extract_from_modules_directory(self, modules_dir: Path) -> List[TOCEntry]:
        """Extract TOC from modules directory structure."""
        toc_entries = []
        
        # Get all module directories
        module_dirs = [d for d in modules_dir.iterdir() if d.is_dir()]
        module_dirs.sort()
        
        for i, module_dir in enumerate(module_dirs, 1):
            # Try to get title from index.cnxml or other files
            title = self._get_module_title(module_dir)
            if not title:
                title = module_dir.name.replace('_', ' ').replace('-', ' ').title()
            
            entry = TOCEntry(
                title=title,
                level=1,
                entry_id=f"module_{i}"
            )
            toc_entries.append(entry)
        
        return toc_entries

    def _get_module_title(self, module_dir: Path) -> Optional[str]:
        """Extract title from module directory."""
        # Look for index.cnxml
        index_file = module_dir / "index.cnxml"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex to find title
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', content)
                if title_match:
                    return title_match.group(1).strip()
                    
            except Exception as e:
                logger.debug(f"Error reading {index_file}: {e}")
        
        return None

    def _extract_from_readme(self, readme_file: Path) -> List[TOCEntry]:
        """Extract basic TOC from README file."""
        toc_entries = []
        
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for markdown headers
            lines = content.split('\n')
            entry_id = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    entry_id += 1
                    
                    # Count header level
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('# ').strip()
                    
                    if title and len(title) > 2:
                        entry = TOCEntry(
                            title=title,
                            level=min(level, 3),  # Cap at level 3
                            entry_id=f"readme_{entry_id}"
                        )
                        toc_entries.append(entry)
                        
        except Exception as e:
            logger.error(f"Error reading README {readme_file}: {e}")
        
        return toc_entries

    def _extract_pdf_toc(self, pdf_path: Path) -> List[TOCEntry]:
        """Extract TOC from PDF file (basic implementation)."""
        toc_entries = []
        
        # For now, create a basic structure based on filename
        # This could be enhanced with actual PDF parsing
        title = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
        
        entry = TOCEntry(
            title=title,
            level=1,
            entry_id="pdf_main"
        )
        toc_entries.append(entry)
        
        return toc_entries

    def _extract_directory_toc(self, dir_path: Path) -> List[TOCEntry]:
        """Extract TOC from general directory structure."""
        toc_entries = []
        
        # Look for common document files
        doc_files = []
        for ext in ['.md', '.txt', '.rst', '.html']:
            doc_files.extend(dir_path.glob(f'*{ext}'))
        
        for i, doc_file in enumerate(doc_files, 1):
            title = doc_file.stem.replace('_', ' ').replace('-', ' ').title()
            
            entry = TOCEntry(
                title=title,
                level=1,
                entry_id=f"doc_{i}"
            )
            toc_entries.append(entry)
        
        return toc_entries

    def _extract_file_toc(self, file_path: Path) -> List[TOCEntry]:
        """Extract TOC from a single file."""
        toc_entries = []
        
        title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
        
        entry = TOCEntry(
            title=title,
            level=1,
            entry_id="file_main"
        )
        toc_entries.append(entry)
        
        return toc_entries

    def _calculate_toc_quality(self, toc_entries: List[TOCEntry]) -> float:
        """Calculate quality score for extracted TOC."""
        if not toc_entries:
            return 0.0
        
        score = 0.5  # Base score
        
        # Number of entries bonus
        if len(toc_entries) >= 5:
            score += 0.2
        elif len(toc_entries) >= 2:
            score += 0.1
        
        # Hierarchy depth bonus
        levels = set(entry.level for entry in toc_entries)
        if len(levels) > 1:
            score += 0.2
        
        # Title quality bonus
        avg_title_length = sum(len(entry.title) for entry in toc_entries) / len(toc_entries)
        if avg_title_length >= 10:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_hierarchy_depth(self, toc_entries: List[TOCEntry]) -> int:
        """Calculate maximum hierarchy depth."""
        if not toc_entries:
            return 0
        return max(entry.level for entry in toc_entries)

    def _organize_tocs_by_level(self, results: List[TOCExtractionResult]) -> Dict[str, List[Dict]]:
        """Organize TOCs by educational level."""
        tocs_by_level = {}
        
        for result in results:
            if result.extraction_status == 'success':
                level = result.educational_level
                if level not in tocs_by_level:
                    tocs_by_level[level] = []
                
                tocs_by_level[level].append({
                    'book_id': result.book_id,
                    'book_title': result.book_title,
                    'toc_entries': result.toc_entries,
                    'quality_score': result.quality_score,
                    'hierarchy_depth': result.hierarchy_depth
                })
        
        return tocs_by_level

    def _calculate_extraction_metrics(self, results: List[TOCExtractionResult], start_time: float) -> TOCExtractionMetrics:
        """Calculate metrics for the extraction process."""
        total_time = time.time() - start_time
        
        successful = [r for r in results if r.extraction_status == 'success']
        failed = [r for r in results if r.extraction_status == 'failed']
        
        total_entries = sum(r.total_topics for r in successful)
        avg_depth = sum(r.hierarchy_depth for r in successful) / len(successful) if successful else 0
        
        return TOCExtractionMetrics(
            total_books_processed=len(results),
            successful_extractions=len(successful),
            failed_extractions=len(failed),
            total_processing_time=total_time,
            average_time_per_book=total_time / len(results) if results else 0,
            total_toc_entries=total_entries,
            average_hierarchy_depth=avg_depth
        )

    def _create_failed_result(self, book: Dict[str, Any], start_time: float, error_msg: str) -> TOCExtractionResult:
        """Create a failed extraction result."""
        return TOCExtractionResult(
            book_id=book['id'],
            book_title=book['title'],
            educational_level=book['educational_level'],
            extraction_status='failed',
            toc_entries=[],
            extraction_time=time.time() - start_time,
            quality_score=0.0,
            hierarchy_depth=0,
            total_topics=0,
            error_message=error_msg
        )

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract TOCs from discovered books")
    parser.add_argument("--discipline", required=True, help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        extractor = SimpleTOCExtractor()
        result = extractor.extract_tocs_from_discovery(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nTOC Extraction Summary for {args.discipline} ({args.language}):")
        print(f"Books processed: {result['total_books_processed']}")
        print(f"Successful extractions: {result['successful_extractions']}")
        print(f"Total TOC entries: {result['metrics']['total_toc_entries']}")
        print(f"Average hierarchy depth: {result['metrics']['average_hierarchy_depth']:.1f}")
        print(f"Processing time: {result['metrics']['total_processing_time']:.2f}s")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_tocs_extracted.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during TOC extraction: {e}")
        exit(1)

if __name__ == "__main__":
    main()