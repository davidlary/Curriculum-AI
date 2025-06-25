#!/usr/bin/env python3
"""
Step 1: Book Discovery Module (Simplified Version)
Automatically identifies all available books for a discipline at all educational levels.

This simplified version works with the existing codebase structure.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import time
import xml.etree.ElementTree as ET

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory structure
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "Cache" / "Books"
OUTPUT_DIR = BASE_DIR / "Books"
BOOKS_DIR = BASE_DIR / "Books"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class BookMetadata:
    """Enhanced book metadata structure."""
    id: str
    title: str
    authors: List[str]
    educational_level: str
    language: str
    discipline: str
    subdiscipline: str
    source: str
    url: str
    format: str
    quality_score: float
    last_updated: str
    discovery_timestamp: str
    file_size_mb: Optional[float] = None
    page_count: Optional[int] = None
    is_elective: bool = False

@dataclass
class DiscoveryMetrics:
    """Simple metrics for book discovery."""
    total_books_found: int
    books_by_level: Dict[str, int]
    books_by_source: Dict[str, int]
    discovery_duration: float
    coverage_completeness: float

class SimpleBookDiscoverer:
    """Simplified book discoverer that works with existing file structure."""
    
    def __init__(self):
        self.books_base_dir = BOOKS_DIR
        
        # Educational level keywords
        self.level_keywords = {
            'high_school': ['high school', 'secondary', 'grade', 'ap', 'honors'],
            'undergraduate': ['college', 'university', 'undergraduate', 'bachelor', 'intro'],
            'graduate': ['graduate', 'masters', 'doctoral', 'phd', 'advanced'],
            'professional': ['professional', 'practitioner', 'handbook', 'reference']
        }
        
        logger.info("SimpleBookDiscoverer initialized")

    def discover_books(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """Discover books from the existing Books directory structure."""
        start_time = time.time()
        logger.info(f"Starting book discovery for {discipline} in {language}")
        logger.info(f"Looking in books directory: {self.books_base_dir}")
        
        discovered_books = []
        seen_titles = set()  # Track titles to avoid duplicates
        
        # Debug: List available directories
        if self.books_base_dir.exists():
            logger.info(f"Books base directory exists. Contents: {[d.name for d in self.books_base_dir.iterdir() if d.is_dir()]}")
        else:
            logger.error(f"Books base directory does not exist: {self.books_base_dir}")
            return self._create_empty_result(discipline, language, start_time)
        
        # Navigate the Books directory structure
        language_dir = self.books_base_dir / language.lower()
        if not language_dir.exists():
            logger.warning(f"Language directory not found: {language_dir}")
            # Try with different casing
            for lang_dir in self.books_base_dir.iterdir():
                if lang_dir.is_dir() and lang_dir.name.lower() == language.lower():
                    language_dir = lang_dir
                    logger.info(f"Found language directory with different casing: {language_dir}")
                    break
            else:
                logger.error(f"No language directory found for {language}")
                return self._create_empty_result(discipline, language, start_time)
        
        logger.info(f"Using language directory: {language_dir}")
        logger.info(f"Available disciplines in {language}: {[d.name for d in language_dir.iterdir() if d.is_dir()]}")
        
        # Look for discipline directory
        discipline_dir = None
        for dir_path in language_dir.iterdir():
            if dir_path.is_dir():
                logger.debug(f"Checking directory: {dir_path.name} for discipline: {discipline}")
                if discipline.lower() in dir_path.name.lower():
                    discipline_dir = dir_path
                    logger.info(f"Found discipline directory: {discipline_dir}")
                    break
        
        if not discipline_dir:
            logger.error(f"Discipline directory not found for {discipline} in {language_dir}")
            logger.info(f"Available disciplines: {[d.name for d in language_dir.iterdir() if d.is_dir()]}")
            return self._create_empty_result(discipline, language, start_time)
        
        logger.info(f"Using discipline directory: {discipline_dir}")
        logger.info(f"Available levels: {[d.name for d in discipline_dir.iterdir() if d.is_dir()]}")
        
        # Discover books at all educational levels
        for level_dir in discipline_dir.iterdir():
            if level_dir.is_dir():
                level_name = self._classify_educational_level(level_dir.name)
                logger.info(f"Processing level directory: {level_dir.name} -> {level_name}")
                books_in_level = self._discover_books_in_level(level_dir, level_name, discipline, language)
                
                # Add books while avoiding duplicates
                unique_count = 0
                for book in books_in_level:
                    title_key = book.title.lower().strip()
                    if title_key not in seen_titles:
                        discovered_books.append(book)
                        seen_titles.add(title_key)
                        unique_count += 1
                        logger.debug(f"Added book: {book.title}")
                    else:
                        logger.info(f"Skipping duplicate book: {book.title}")
                        
                logger.info(f"Found {len(books_in_level)} books in {level_dir.name} ({unique_count} unique)")
        
        # Calculate metrics
        metrics = self._calculate_metrics(discovered_books, start_time)
        
        # Create result
        result = {
            'discipline': discipline,
            'language': language,
            'discovery_timestamp': datetime.now().isoformat(),
            'books': [asdict(book) for book in discovered_books],
            'metrics': asdict(metrics),
            'total_books': len(discovered_books)
        }
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_books_discovered.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Discovery completed: {len(discovered_books)} books found")
        return result

    def _discover_books_in_level(self, level_dir: Path, level_name: str, discipline: str, language: str) -> List[BookMetadata]:
        """Discover books in a specific educational level directory."""
        books = []
        
        # Look for book directories and files
        for item in level_dir.rglob("*"):
            if item.is_file():
                # Skip README files entirely
                if self._is_readme_file(item):
                    logger.debug(f"Skipping README file: {item}")
                    continue
                
                # Check for collection XML files (individual books within bundles)
                if self._is_collection_file(item):
                    book = self._create_book_metadata_from_collection(item, level_name, discipline, language)
                    if book:
                        books.append(book)
                elif self._is_book_file(item):
                    book = self._create_book_metadata(item, level_name, discipline, language)
                    if book:
                        books.append(book)
            elif item.is_dir() and self._is_book_directory(item):
                # Only process directories that don't contain collection files
                # (to avoid duplicating books that are handled as collections)
                if not self._has_collection_files(item):
                    book = self._create_book_metadata_from_dir(item, level_name, discipline, language)
                    if book:
                        books.append(book)
        
        return books

    def _is_readme_file(self, file_path: Path) -> bool:
        """Check if a file is a README file that should be ignored."""
        readme_patterns = ['readme', 'README']
        return any(pattern in file_path.name for pattern in readme_patterns)
    
    def _is_collection_file(self, file_path: Path) -> bool:
        """Check if a file is a collection XML file (individual book)."""
        return (file_path.suffix.lower() == '.xml' and 
                'collection' in file_path.name.lower() and 
                'collections' in str(file_path.parent))
    
    def _has_collection_files(self, dir_path: Path) -> bool:
        """Check if a directory contains collection files."""
        collections_dir = dir_path / 'collections'
        if collections_dir.exists():
            return any(f.suffix.lower() == '.xml' and 'collection' in f.name.lower() 
                      for f in collections_dir.iterdir())
        return False

    def _is_book_file(self, file_path: Path) -> bool:
        """Check if a file is likely a book."""
        book_extensions = {'.pdf', '.epub', '.mobi', '.txt'}
        return file_path.suffix.lower() in book_extensions

    def _is_book_directory(self, dir_path: Path) -> bool:
        """Check if a directory contains a book."""
        # Look for typical book directory indicators
        indicators = ['README.md', 'LICENSE', 'modules', 'collections', 'media']
        return any((dir_path / indicator).exists() for indicator in indicators)

    def _create_book_metadata(self, file_path: Path, level: str, discipline: str, language: str) -> Optional[BookMetadata]:
        """Create book metadata from a file."""
        try:
            title = file_path.stem
            book_id = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
            
            # Determine format
            format_type = file_path.suffix.lower().lstrip('.')
            
            # Calculate file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else None
            
            # Enhanced educational level classification
            enhanced_level = self._enhanced_educational_level_classification(title, file_path.parent.name)
            
            # Mark elective status
            is_elective = self._is_elective_subject(title, discipline)
            
            return BookMetadata(
                id=book_id,
                title=title,
                authors=[],
                educational_level=enhanced_level,
                language=language,
                discipline=discipline,
                subdiscipline=self._classify_subdiscipline(title, discipline),
                source='local_files',
                url=str(file_path),
                format=format_type,
                quality_score=self._calculate_quality_score(file_path, title),
                last_updated=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                discovery_timestamp=datetime.now().isoformat(),
                file_size_mb=file_size_mb,
                is_elective=is_elective
            )
        except Exception as e:
            logger.error(f"Error creating metadata for {file_path}: {e}")
            return None

    def _create_book_metadata_from_collection(self, xml_file: Path, level: str, discipline: str, language: str) -> Optional[BookMetadata]:
        """Create book metadata from a collection XML file."""
        try:
            # Parse XML to extract title
            title = self._extract_title_from_collection(xml_file)
            if not title:
                title = xml_file.stem.replace('-', ' ').title()
            
            book_id = hashlib.md5(str(xml_file).encode()).hexdigest()[:8]
            
            # Enhanced educational level classification - get the actual level directory
            level_dir_name = xml_file.parent.parent.parent.name  # Go up to HighSchool/University level
            enhanced_level = self._enhanced_educational_level_classification(title, level_dir_name)
            
            # Mark elective status
            is_elective = self._is_elective_subject(title, discipline)
            
            return BookMetadata(
                id=book_id,
                title=title,
                authors=[],
                educational_level=enhanced_level,
                language=language,
                discipline=discipline,
                subdiscipline=self._classify_subdiscipline(title, discipline),
                source='openstax',
                url=str(xml_file),
                format='xml_collection',
                quality_score=0.95,  # High quality for structured OpenStax collections
                last_updated=datetime.fromtimestamp(xml_file.stat().st_mtime).isoformat(),
                discovery_timestamp=datetime.now().isoformat(),
                is_elective=is_elective
            )
        except Exception as e:
            logger.error(f"Error creating metadata for collection {xml_file}: {e}")
            return None
    
    def _extract_title_from_collection(self, xml_file: Path) -> Optional[str]:
        """Extract title from collection XML file."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Look for title in metadata section
            # Handle namespace
            namespaces = {
                'col': 'http://cnx.rice.edu/collxml',
                'md': 'http://cnx.rice.edu/mdml'
            }
            
            title_elem = root.find('.//md:title', namespaces)
            if title_elem is not None:
                return title_elem.text
            
            # Fallback: look for title without namespace
            for elem in root.iter():
                if elem.tag.endswith('title') and elem.text:
                    return elem.text
                    
            return None
        except Exception as e:
            logger.warning(f"Could not parse XML title from {xml_file}: {e}")
            return None

    def _create_book_metadata_from_dir(self, dir_path: Path, level: str, discipline: str, language: str) -> Optional[BookMetadata]:
        """Create book metadata from a directory (only for directories without collections)."""
        try:
            title = dir_path.name.replace('osbooks-', '').replace('-', ' ').title()
            book_id = hashlib.md5(str(dir_path).encode()).hexdigest()[:8]
            
            # Enhanced educational level classification
            enhanced_level = self._enhanced_educational_level_classification(title, dir_path.parent.name)
            
            # Mark elective status
            is_elective = self._is_elective_subject(title, discipline)
            
            return BookMetadata(
                id=book_id,
                title=title,
                authors=[],
                educational_level=enhanced_level,
                language=language,
                discipline=discipline,
                subdiscipline=self._classify_subdiscipline(title, discipline),
                source='openstax',
                url=str(dir_path),
                format='directory',
                quality_score=self._calculate_quality_score_dir(dir_path, title),
                last_updated=datetime.fromtimestamp(dir_path.stat().st_mtime).isoformat(),
                discovery_timestamp=datetime.now().isoformat(),
                is_elective=is_elective
            )
        except Exception as e:
            logger.error(f"Error creating metadata for directory {dir_path}: {e}")
            return None

    def _classify_educational_level(self, dir_name: str) -> str:
        """Classify educational level from directory name."""
        dir_lower = dir_name.lower()
        
        for level, keywords in self.level_keywords.items():
            if any(keyword in dir_lower for keyword in keywords):
                return level
        
        # Default mapping
        if 'high' in dir_lower:
            return 'high_school'
        elif 'university' in dir_lower or 'college' in dir_lower:
            return 'undergraduate'
        elif 'graduate' in dir_lower:
            return 'graduate'
        else:
            return 'undergraduate'
    
    def _enhanced_educational_level_classification(self, title: str, directory_hint: str = "") -> str:
        """Enhanced educational level classification using existing sophisticated logic."""
        text = f"{title} {directory_hint}".lower()
        
        # High school indicators (most specific first)
        high_school_patterns = [
            'high school', 'high-school', 'highschool', 'hs ',
            'ap course', 'ap physics', 'ap biology', 'ap chemistry', 'ap®', 'ap ',
            'for ap courses', 'advanced placement',
            'pre-algebra', 'prealgebra', 'basic music theory',
            'introductory physics', 'fundamentals of'
        ]
        
        # Graduate level indicators
        graduate_patterns = [
            'graduate', 'phd', 'doctoral', 'advanced research',
            'signal processing', 'machine learning', 'research methods'
        ]
        
        # University level indicators (more specific)
        university_patterns = [
            'university physics', 'college physics', 'calculus-based',
            'undergraduate', 'college', 'university'
        ]
        
        # Check for AP courses - these are high school level
        for pattern in high_school_patterns:
            if pattern in text:
                return 'high_school'
        
        # Check for graduate indicators
        for pattern in graduate_patterns:
            if pattern in text:
                return 'graduate'
        
        # Physics-specific rules based on existing logic
        if 'physics' in text:
            if 'college physics' in text or 'university physics' in text:
                # University-level physics courses
                return 'undergraduate'
            elif any(term in text for term in ['physics', 'mechanics']) and 'volume' in text:
                # Multi-volume university physics series
                return 'undergraduate'
            elif text.strip().lower() == 'physics' and 'highschool' in directory_hint.lower():
                # Generic "Physics" book in high school directory
                return 'high_school'
        
        # Directory-based classification as fallback
        if 'highschool' in directory_hint.lower() or 'high' in directory_hint.lower():
            return 'high_school'
        elif 'university' in directory_hint.lower() or 'college' in directory_hint.lower():
            return 'undergraduate'
        
        # Default to undergraduate for most OpenStax content
        return 'undergraduate'
    
    def _is_elective_subject(self, title: str, discipline: str) -> bool:
        """Determine if a subject is an elective rather than core curriculum."""
        title_lower = title.lower()
        
        # Elective subjects within Physics
        physics_electives = [
            'astronomy', 'astrophysics', 'cosmology',
            'nuclear physics', 'particle physics',
            'biophysics', 'geophysics',
            'history of physics'
        ]
        
        if discipline.lower() == 'physics':
            for elective in physics_electives:
                if elective in title_lower:
                    return True
        
        return False

    def _classify_subdiscipline(self, title: str, discipline: str) -> str:
        """Classify subdiscipline based on title."""
        title_lower = title.lower()
        
        if discipline.lower() == 'physics':
            physics_subdisciplines = {
                'mechanics': ['mechanics', 'dynamics', 'kinematics'],
                'thermodynamics': ['thermodynamics', 'thermal'],
                'electromagnetism': ['electromagnetism', 'electricity', 'magnetism'],
                'quantum': ['quantum', 'atomic'],
                'optics': ['optics', 'light'],
                'astronomy': ['astronomy', 'astrophysics']
            }
            
            for subdiscipline, keywords in physics_subdisciplines.items():
                if any(keyword in title_lower for keyword in keywords):
                    return subdiscipline
        
        return 'general'

    def _calculate_quality_score(self, file_path: Path, title: str) -> float:
        """Calculate quality score for a file."""
        score = 0.5  # Base score
        
        # File size bonus
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > 1024 * 1024:  # > 1MB
                score += 0.2
        
        # Format bonus
        if file_path.suffix.lower() == '.pdf':
            score += 0.2
        
        # Title quality
        if len(title) > 10:
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_quality_score_dir(self, dir_path: Path, title: str) -> float:
        """Calculate quality score for a directory."""
        score = 0.7  # Higher base for structured directories
        
        # Check for quality indicators
        if (dir_path / "README.md").exists():
            score += 0.1
        
        if (dir_path / "LICENSE").exists():
            score += 0.1
        
        if any(dir_path.glob("modules/*")):
            score += 0.1
        
        return min(score, 1.0)

    def _calculate_metrics(self, books: List[BookMetadata], start_time: float) -> DiscoveryMetrics:
        """Calculate discovery metrics."""
        duration = time.time() - start_time
        
        # Books by level
        books_by_level = {}
        for book in books:
            level = book.educational_level
            books_by_level[level] = books_by_level.get(level, 0) + 1
        
        # Books by source
        books_by_source = {}
        for book in books:
            source = book.source
            books_by_source[source] = books_by_source.get(source, 0) + 1
        
        # Coverage (simplified)
        expected_levels = ['high_school', 'undergraduate', 'graduate']
        covered_levels = set(books_by_level.keys())
        coverage = len(covered_levels.intersection(expected_levels)) / len(expected_levels)
        
        return DiscoveryMetrics(
            total_books_found=len(books),
            books_by_level=books_by_level,
            books_by_source=books_by_source,
            discovery_duration=duration,
            coverage_completeness=coverage
        )

    def _create_empty_result(self, discipline: str, language: str, start_time: float) -> Dict[str, Any]:
        """Create empty result when no books found."""
        duration = time.time() - start_time
        
        return {
            'discipline': discipline,
            'language': language,
            'discovery_timestamp': datetime.now().isoformat(),
            'books': [],
            'metrics': asdict(DiscoveryMetrics(
                total_books_found=0,
                books_by_level={},
                books_by_source={},
                discovery_duration=duration,
                coverage_completeness=0.0
            )),
            'total_books': 0
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Discover books for curriculum generation")
    parser.add_argument("--discipline", required=True, help="Target discipline (e.g., Physics, Mathematics)")
    parser.add_argument("--language", default="English", help="Target language (default: English)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run discovery
    try:
        discoverer = SimpleBookDiscoverer()
        result = discoverer.discover_books(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nBook Discovery Summary for {args.discipline} ({args.language}):")
        print(f"Total books found: {result['total_books']}")
        print(f"Books by level: {result['metrics']['books_by_level']}")
        print(f"Books by source: {result['metrics']['books_by_source']}")
        print(f"Coverage completeness: {result['metrics']['coverage_completeness']:.2%}")
        print(f"Discovery duration: {result['metrics']['discovery_duration']:.2f}s")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_books_discovered.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during book discovery: {e}")
        exit(1)

if __name__ == "__main__":
    main()