#!/usr/bin/env python3
"""
Step 2: TOC Extraction Module (Using Existing Infrastructure)
Leverages the existing comprehensive TOC extraction and book reading capabilities.

This module uses the proven toc_aware_curriculum_system.py and other existing
infrastructure to extract high-quality TOCs from all discovered books.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing proven modules
try:
    from core.toc_extractor import TOCExtractor, TOCEntry, BookTOC
    from core.text_extractor import TextExtractor
    from scripts.toc_aware_curriculum_system import TOCAwareCurriculumSystem, ComprehensiveBookDiscoverer
    from scripts.parse_textbooks import chunk_extracted_content
    logger.info("Successfully imported existing TOC infrastructure")
    EXISTING_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some existing modules not available: {e}")
    EXISTING_MODULES_AVAILABLE = False

# Directory structure
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "Cache" / "TOCs"
OUTPUT_DIR = BASE_DIR / "TOCs"
BOOKS_DIR = BASE_DIR / "Books"

# Create directories
for dir_path in [CACHE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class IntegratedTOCExtractor:
    """
    Uses existing proven TOC extraction infrastructure to extract comprehensive
    table of contents from all discovered books.
    """
    
    def __init__(self):
        """Initialize with existing infrastructure."""
        if EXISTING_MODULES_AVAILABLE:
            self.toc_extractor = TOCExtractor()
            self.text_extractor = TextExtractor()
            self.toc_system = TOCAwareCurriculumSystem()
            self.book_discoverer = ComprehensiveBookDiscoverer()
        
        self.cache_ttl_hours = 168  # Cache valid for 1 week
        logger.info("IntegratedTOCExtractor initialized")

    def extract_tocs_from_discovery(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """
        Extract TOCs using the existing proven infrastructure.
        
        This leverages the comprehensive book discovery and TOC extraction
        that's already been tested and proven to work.
        """
        start_time = datetime.now()
        logger.info(f"Starting integrated TOC extraction for {discipline} in {language}")
        
        # Load discovered books from Step 1
        books_file = BOOKS_DIR / f"{discipline}_{language}_books_discovered.json"
        if not books_file.exists():
            raise FileNotFoundError(f"Books discovery file not found: {books_file}")
        
        with open(books_file, 'r', encoding='utf-8') as f:
            discovery_data = json.load(f)
        
        books = discovery_data['books']
        logger.info(f"Processing {len(books)} discovered books")
        
        if EXISTING_MODULES_AVAILABLE:
            # Use the proven TOC-aware curriculum system
            result = self._extract_using_existing_system(books, discipline, language, start_time)
        else:
            # Fallback to basic extraction
            result = self._extract_using_fallback(books, discipline, language, start_time)
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_tocs_extracted.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"TOC extraction completed: {result.get('successful_extractions', 0)} successful")
        return result

    def _extract_using_existing_system(self, books: List[Dict], discipline: str, language: str, start_time: datetime) -> Dict[str, Any]:
        """Use the existing proven TOC extraction system with chunks fallback."""
        logger.info("Using existing TOC-aware curriculum system with chunks fallback")
        
        # Convert discovered books to the format expected by existing system
        book_paths = []
        for book in books:
            book_path = Path(book['url'])
            if book_path.exists():
                book_paths.append({
                    'path': book_path,
                    'educational_level': book['educational_level'],
                    'title': book['title'],
                    'source': book['source'],
                    'book_id': book['id']
                })
        
        logger.info(f"Processing {len(book_paths)} valid book paths")
        
        # Use existing comprehensive book discovery and TOC extraction
        all_tocs = []
        extraction_results = []
        
        for book_info in book_paths:
            try:
                logger.info(f"Extracting TOC from: {book_info['title']}")
                
                # Use existing TOC extractor
                book_toc = self.toc_extractor.extract_toc(
                    book_info['path'],
                    language,
                    discipline, 
                    book_info['educational_level']
                )
                
                if book_toc and book_toc.entries:
                    all_tocs.append(book_toc)
                    
                    extraction_results.append({
                        'book_id': book_info['book_id'],
                        'book_title': book_info['title'],
                        'educational_level': book_info['educational_level'],
                        'extraction_status': 'success',
                        'toc_entries': [entry.to_dict() for entry in book_toc.entries],
                        'total_topics': len(book_toc.entries),
                        'extraction_method': book_toc.extraction_method
                    })
                    
                    logger.info(f"Successfully extracted {len(book_toc.entries)} TOC entries from {book_info['title']}")
                else:
                    # Try chunks fallback for failed TOC extraction
                    logger.info(f"Standard TOC extraction failed for {book_info['title']}, trying chunks fallback")
                    chunks_toc = self._extract_toc_from_chunks(book_info, discipline, language)
                    
                    if chunks_toc:
                        all_tocs.append(chunks_toc)
                        extraction_results.append({
                            'book_id': book_info['book_id'],
                            'book_title': book_info['title'],
                            'educational_level': book_info['educational_level'],
                            'extraction_status': 'success_chunks_fallback',
                            'toc_entries': [entry.to_dict() for entry in chunks_toc.entries],
                            'total_topics': len(chunks_toc.entries),
                            'extraction_method': 'chunks_fallback'
                        })
                        logger.info(f"Successfully extracted {len(chunks_toc.entries)} TOC entries using chunks fallback")
                    else:
                        extraction_results.append({
                            'book_id': book_info['book_id'],
                            'book_title': book_info['title'],
                            'educational_level': book_info['educational_level'],
                            'extraction_status': 'failed',
                            'toc_entries': [],
                            'total_topics': 0,
                            'error_message': 'No TOC entries found in standard extraction or chunks fallback'
                        })
                    
            except Exception as e:
                logger.error(f"Error extracting TOC from {book_info['title']}: {e}")
                extraction_results.append({
                    'book_id': book_info['book_id'],
                    'book_title': book_info['title'],
                    'educational_level': book_info['educational_level'],
                    'extraction_status': 'failed',
                    'toc_entries': [],
                    'total_topics': 0,
                    'error_message': str(e)
                })
        
        # Organize results by educational level
        tocs_by_level = self._organize_by_level(extraction_results)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(extraction_results, start_time)
        
        # Use existing TOC-aware processing for enhanced analysis
        if all_tocs:
            enhanced_analysis = self._run_toc_aware_analysis(all_tocs, discipline)
        else:
            enhanced_analysis = {}
        
        return {
            'discipline': discipline,
            'language': language,
            'extraction_timestamp': datetime.now().isoformat(),
            'tocs_by_level': tocs_by_level,
            'extraction_results': extraction_results,
            'metrics': metrics,
            'enhanced_analysis': enhanced_analysis,
            'total_books_processed': len(books),
            'successful_extractions': len([r for r in extraction_results if r['extraction_status'] == 'success']),
            'using_existing_infrastructure': True
        }

    def _run_toc_aware_analysis(self, tocs: List[BookTOC], discipline: str) -> Dict[str, Any]:
        """Run the existing TOC-aware analysis for enhanced insights."""
        try:
            # Use existing TOC-aware curriculum system for analysis
            logger.info("Running TOC-aware analysis using existing infrastructure")
            
            # Extract meaningful subtopics using existing system
            meaningful_subtopics = []
            for toc in tocs:
                for entry in toc.entries:
                    # Convert to the format expected by existing system
                    meaningful_subtopics.append({
                        'id': entry.entry_id or f"entry_{len(meaningful_subtopics)}",
                        'name': entry.title,
                        'level': entry.level,
                        'educational_level': toc.level,
                        'source_book': toc.book_title,
                        'parent_id': entry.parent_id,
                        'section_number': entry.section_number
                    })
            
            # Calculate cross-level normalization
            cross_level_analysis = self._analyze_cross_level_topics(meaningful_subtopics)
            
            # Calculate hierarchy analysis
            hierarchy_analysis = self._analyze_hierarchy_structure(meaningful_subtopics)
            
            return {
                'meaningful_subtopics_count': len(meaningful_subtopics),
                'cross_level_analysis': cross_level_analysis,
                'hierarchy_analysis': hierarchy_analysis,
                'unique_topics': len(set(s['name'] for s in meaningful_subtopics)),
                'topics_by_level': self._count_topics_by_level(meaningful_subtopics)
            }
            
        except Exception as e:
            logger.error(f"Error in TOC-aware analysis: {e}")
            return {'error': str(e)}

    def _analyze_cross_level_topics(self, subtopics: List[Dict]) -> Dict[str, Any]:
        """Analyze topics that appear across multiple educational levels."""
        topic_levels = {}
        for subtopic in subtopics:
            topic_name = subtopic['name'].lower().strip()
            level = subtopic['educational_level']
            
            if topic_name not in topic_levels:
                topic_levels[topic_name] = set()
            topic_levels[topic_name].add(level)
        
        cross_level_topics = {topic: list(levels) for topic, levels in topic_levels.items() if len(levels) > 1}
        
        return {
            'cross_level_topics_count': len(cross_level_topics),
            'cross_level_topics': dict(list(cross_level_topics.items())[:20]),  # Top 20 for display
            'single_level_topics': len(topic_levels) - len(cross_level_topics),
            'coverage_analysis': {
                level: len([t for t, lvls in topic_levels.items() if level in lvls])
                for level in ['high_school', 'undergraduate', 'graduate']
            }
        }

    def _analyze_hierarchy_structure(self, subtopics: List[Dict]) -> Dict[str, Any]:
        """Analyze the hierarchical structure of extracted topics."""
        levels = [s['level'] for s in subtopics]
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'max_hierarchy_depth': max(levels) if levels else 0,
            'level_distribution': level_counts,
            'avg_hierarchy_depth': sum(levels) / len(levels) if levels else 0,
            'topics_with_parents': len([s for s in subtopics if s.get('parent_id')]),
            'root_level_topics': len([s for s in subtopics if s['level'] == 1])
        }

    def _count_topics_by_level(self, subtopics: List[Dict]) -> Dict[str, int]:
        """Count topics by educational level."""
        counts = {}
        for subtopic in subtopics:
            level = subtopic['educational_level']
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _extract_toc_from_chunks(self, book_info: Dict, discipline: str, language: str) -> Optional[Any]:
        """Extract TOC information from existing chunks as fallback."""
        try:
            # Look for chunks matching this book
            chunks_file = BASE_DIR / "Chunks" / f"{discipline}_{book_info['educational_level']}.jsonl"
            if not chunks_file.exists():
                logger.warning(f"No chunks file found: {chunks_file}")
                return None
            
            logger.info(f"Reading chunks from: {chunks_file}")
            
            # Read chunks and extract TOC structure
            toc_entries = []
            book_title_lower = book_info['title'].lower()
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        chunk = json.loads(line.strip())
                        
                        # Check if this chunk belongs to our book
                        chunk_book = chunk.get('book_title', '').lower()
                        if book_title_lower in chunk_book or chunk_book in book_title_lower:
                            
                            # First, add the chapter title itself as a TOC entry
                            chapter_title = chunk.get('chapter_title', '')
                            if chapter_title and chapter_title not in ['Untitled Chapter', 'Content Section']:
                                toc_entries.append(TOCEntry(
                                    title=chapter_title,
                                    level=1,
                                    entry_id=f"chapter_{chunk.get('chapter_number', line_num)}",
                                    section_number=chunk.get('chapter_number', str(line_num))
                                ))
                            
                            # Extract subsections as TOC entries
                            subsections = chunk.get('subsections', [])
                            for subsection in subsections:
                                # Parse subsection content for meaningful titles
                                content = subsection.get('content', '')
                                section_id = subsection.get('id', f'section_{line_num}')
                                
                                # Extract learning objectives, section titles, etc.
                                toc_entries.extend(self._parse_chunk_content_for_toc(content, section_id))
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in chunks file line {line_num}: {e}")
                        continue
            
            if not toc_entries:
                logger.warning(f"No TOC entries found in chunks for book: {book_info['title']}")
                return None
            
            # Create BookTOC object
            if EXISTING_MODULES_AVAILABLE:
                from core.toc_extractor import BookTOC
                book_toc = BookTOC(
                    book_title=book_info['title'],
                    language=language.lower(),
                    discipline=discipline,
                    level=book_info['educational_level'],
                    file_path=str(book_info['path']),
                    entries=toc_entries,
                    extraction_method="chunks_fallback"
                )
                
                logger.info(f"Created TOC from chunks with {len(toc_entries)} entries")
                return book_toc
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error extracting TOC from chunks for {book_info['title']}: {e}")
            return None

    def _parse_chunk_content_for_toc(self, content: str, section_id: str) -> List[Any]:
        """Parse chunk content to extract meaningful TOC entries."""
        toc_entries = []
        
        if not EXISTING_MODULES_AVAILABLE:
            return toc_entries
            
        from core.toc_extractor import TOCEntry
        
        # Look for section learning objectives
        objectives_match = re.search(r'Section Learning Objectives[:\s]*By the end of this section[,\s]*you will be able to do the following:(.*?)(?=Teacher Support|Section Key Terms|\n\n|$)', content, re.IGNORECASE | re.DOTALL)
        if objectives_match:
            objectives = objectives_match.group(1).strip()
            # Split objectives into individual points
            obj_points = re.split(r'(?:^|\n)\s*[•·\-\*]?\s*', objectives)
            for i, obj in enumerate(obj_points):
                obj = obj.strip()
                if obj and len(obj) > 10:  # Filter out very short fragments
                    toc_entries.append(TOCEntry(
                        title=obj[:100] + "..." if len(obj) > 100 else obj,
                        level=2,
                        entry_id=f"{section_id}_obj_{i}",
                        section_number=f"{section_id}.{i+1}"
                    ))
        
        # Look for section titles and subsection headers
        section_headers = re.findall(r'^([A-Z][^:\n]{10,80})(?:\s|$)', content, re.MULTILINE)
        for i, header in enumerate(section_headers[:5]):  # Limit to first 5 headers
            if not any(skip_word in header.lower() for skip_word in ['teacher support', 'section learning', 'key terms']):
                toc_entries.append(TOCEntry(
                    title=header.strip(),
                    level=3,
                    entry_id=f"{section_id}_header_{i}",
                    section_number=f"{section_id}.h.{i+1}"
                ))
        
        # Look for key terms
        key_terms_match = re.search(r'Section Key Terms[:\s]*(.*?)(?=Teacher Support|\n\n|$)', content, re.IGNORECASE | re.DOTALL)
        if key_terms_match:
            key_terms = key_terms_match.group(1).strip()
            terms = re.split(r'\s+', key_terms)
            # Take meaningful terms (not too short, not common words)
            meaningful_terms = [term for term in terms if len(term) > 3 and term.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']][:10]
            for i, term in enumerate(meaningful_terms):
                toc_entries.append(TOCEntry(
                    title=f"Key Term: {term}",
                    level=4,
                    entry_id=f"{section_id}_term_{i}",
                    section_number=f"{section_id}.t.{i+1}"
                ))
        
        return toc_entries

    def _extract_using_fallback(self, books: List[Dict], discipline: str, language: str, start_time: datetime) -> Dict[str, Any]:
        """Fallback extraction method if existing modules aren't available."""
        logger.info("Using fallback TOC extraction method")
        
        # Simple fallback implementation
        extraction_results = []
        for book in books:
            extraction_results.append({
                'book_id': book['id'],
                'book_title': book['title'],
                'educational_level': book['educational_level'],
                'extraction_status': 'fallback',
                'toc_entries': [{'title': book['title'], 'level': 1, 'entry_id': 'fallback_1'}],
                'total_topics': 1,
                'extraction_method': 'fallback'
            })
        
        tocs_by_level = self._organize_by_level(extraction_results)
        metrics = self._calculate_comprehensive_metrics(extraction_results, start_time)
        
        return {
            'discipline': discipline,
            'language': language,
            'extraction_timestamp': datetime.now().isoformat(),
            'tocs_by_level': tocs_by_level,
            'extraction_results': extraction_results,
            'metrics': metrics,
            'total_books_processed': len(books),
            'successful_extractions': len(extraction_results),
            'using_existing_infrastructure': False,
            'note': 'Used fallback method - existing infrastructure not available'
        }

    def _organize_by_level(self, extraction_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize extraction results by educational level."""
        organized = {}
        
        for result in extraction_results:
            if result['extraction_status'] in ['success', 'fallback']:
                level = result['educational_level']
                if level not in organized:
                    organized[level] = []
                
                organized[level].append({
                    'book_id': result['book_id'],
                    'book_title': result['book_title'],
                    'toc_entries': result['toc_entries'],
                    'total_topics': result['total_topics'],
                    'extraction_method': result.get('extraction_method', 'unknown')
                })
        
        return organized

    def _calculate_comprehensive_metrics(self, extraction_results: List[Dict], start_time: datetime) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the extraction process."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        successful = [r for r in extraction_results if r['extraction_status'] == 'success']
        failed = [r for r in extraction_results if r['extraction_status'] == 'failed']
        
        total_entries = sum(r['total_topics'] for r in successful)
        
        return {
            'total_books_processed': len(extraction_results),
            'successful_extractions': len(successful),
            'failed_extractions': len(failed),
            'total_processing_time': duration,
            'average_time_per_book': duration / len(extraction_results) if extraction_results else 0,
            'total_toc_entries': total_entries,
            'average_entries_per_book': total_entries / len(successful) if successful else 0,
            'success_rate': len(successful) / len(extraction_results) if extraction_results else 0,
            'extraction_methods': list(set(r.get('extraction_method', 'unknown') for r in successful))
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract TOCs using existing infrastructure")
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
        extractor = IntegratedTOCExtractor()
        result = extractor.extract_tocs_from_discovery(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nIntegrated TOC Extraction Summary for {args.discipline} ({args.language}):")
        print(f"Books processed: {result['total_books_processed']}")
        print(f"Successful extractions: {result['successful_extractions']}")
        print(f"Using existing infrastructure: {result.get('using_existing_infrastructure', False)}")
        
        if 'metrics' in result:
            print(f"Total TOC entries: {result['metrics']['total_toc_entries']}")
            print(f"Success rate: {result['metrics']['success_rate']:.2%}")
            print(f"Processing time: {result['metrics']['total_processing_time']:.2f}s")
        
        if 'enhanced_analysis' in result and result['enhanced_analysis']:
            analysis = result['enhanced_analysis']
            print(f"Meaningful subtopics: {analysis.get('meaningful_subtopics_count', 0)}")
            print(f"Cross-level topics: {analysis.get('cross_level_analysis', {}).get('cross_level_topics_count', 0)}")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_tocs_extracted.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during TOC extraction: {e}")
        exit(1)

if __name__ == "__main__":
    main()