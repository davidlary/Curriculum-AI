#!/usr/bin/env python3
"""
book_list_generator.py - Generate comprehensive book lists for the OpenBooks collection

This module scans the ./Books directory and creates BookList.tsv and BookList.json files
containing metadata for all downloaded books, enabling direct and efficient access
for contents.py and other processing tools.

Generated fields:
- book_id: Unique identifier for the book
- book_name: English name of the book
- original_name: Original name in source language (if different)
- language: Book language
- discipline: Academic discipline
- level: Educational level (University, HighSchool, etc.)
- format: Book format (git_repository, pdf)
- file_path: Path to the main content file
- collection_file: Path to collection.xml (if available)
- pdf_file: Path to PDF file (if available)
- last_updated: Last modification timestamp
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import csv

logger = logging.getLogger(__name__)


@dataclass
class BookEntry:
    """Represents a book entry in the BookList."""
    book_id: str
    book_name: str
    original_name: str
    language: str
    discipline: str
    level: str
    format: str
    file_path: str
    collection_file: Optional[str] = None
    pdf_file: Optional[str] = None
    last_updated: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BookListGenerator:
    """Generates comprehensive book lists from the Books directory."""
    
    def __init__(self, books_dir: Path):
        """Initialize with books directory path."""
        self.books_dir = books_dir
        self.books: List[BookEntry] = []
        
    def scan_books_directory(self) -> List[BookEntry]:
        """
        Scan the Books directory and identify all downloaded books.
        
        Directory structure: ./Books/<Language>/<Discipline>/<Level>/...
        """
        books = []
        
        if not self.books_dir.exists():
            logger.warning(f"Books directory does not exist: {self.books_dir}")
            return books
        
        # Traverse directory structure: Books/<Language>/<Discipline>/<Level>/
        for language_dir in self.books_dir.iterdir():
            if not language_dir.is_dir() or language_dir.name.startswith('.'):
                continue
            
            language = language_dir.name
            
            for discipline_dir in language_dir.iterdir():
                if not discipline_dir.is_dir() or discipline_dir.name.startswith('.'):
                    continue
                
                discipline = discipline_dir.name
                
                for level_dir in discipline_dir.iterdir():
                    if not level_dir.is_dir() or level_dir.name.startswith('.'):
                        continue
                    
                    level = level_dir.name
                    
                    # Scan for books in this level directory
                    books.extend(self._scan_level_directory(level_dir, language, discipline, level))
        
        logger.info(f"Found {len(books)} books in total")
        return books
    
    def _scan_level_directory(self, level_dir: Path, language: str, discipline: str, level: str) -> List[BookEntry]:
        """Scan a specific level directory for books."""
        books = []
        
        # Look for book directories (containing collection.xml or PDFs)
        for item in level_dir.iterdir():
            if not item.is_dir() or item.name.startswith('.') or item.name.lower() == 'readme':
                continue
            
            # Check for Git repository format (contains collections/)
            collections_dir = item / "collections"
            if collections_dir.exists():
                # This is a Git repository format book
                collection_files = list(collections_dir.glob("*.collection.xml"))
                for collection_file in collection_files:
                    book_entry = self._create_book_entry_from_collection(
                        collection_file, item, language, discipline, level
                    )
                    if book_entry:
                        books.append(book_entry)
            
            # Check for direct PDF files in the directory
            pdf_files = list(item.glob("*.pdf"))
            for pdf_file in pdf_files:
                if not any(book.pdf_file == str(pdf_file) for book in books):
                    book_entry = self._create_book_entry_from_pdf(
                        pdf_file, language, discipline, level
                    )
                    if book_entry:
                        books.append(book_entry)
        
        # Also check for PDFs directly in the level directory
        for pdf_file in level_dir.glob("*.pdf"):
            if not any(book.pdf_file == str(pdf_file) for book in books):
                book_entry = self._create_book_entry_from_pdf(
                    pdf_file, language, discipline, level
                )
                if book_entry:
                    books.append(book_entry)
        
        return books
    
    def _create_book_entry_from_collection(self, collection_file: Path, book_dir: Path, 
                                         language: str, discipline: str, level: str) -> Optional[BookEntry]:
        """Create a book entry from a collection.xml file."""
        try:
            # Extract book name from collection.xml
            book_name, original_name = self._extract_book_title_from_collection(collection_file)
            
            # Generate book ID
            book_id = f"{language}_{discipline}_{level}_{book_dir.name}"
            
            # Check for corresponding PDF
            pdf_file = None
            possible_pdf_paths = [
                book_dir / f"{book_dir.name}.pdf",
                book_dir.parent / f"{book_dir.name}.pdf",
                book_dir / "book.pdf"
            ]
            for pdf_path in possible_pdf_paths:
                if pdf_path.exists():
                    pdf_file = str(pdf_path)
                    break
            
            # Get last update time
            last_updated = datetime.fromtimestamp(collection_file.stat().st_mtime).isoformat()
            
            return BookEntry(
                book_id=book_id,
                book_name=book_name,
                original_name=original_name,
                language=language,
                discipline=discipline,
                level=level,
                format="git_repository",
                file_path=str(book_dir),
                collection_file=str(collection_file),
                pdf_file=pdf_file,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error processing collection file {collection_file}: {e}")
            return None
    
    def _create_book_entry_from_pdf(self, pdf_file: Path, language: str, 
                                  discipline: str, level: str) -> Optional[BookEntry]:
        """Create a book entry from a PDF file."""
        try:
            # Extract book name from PDF filename
            book_name = pdf_file.stem.replace('-', ' ').replace('_', ' ').title()
            original_name = book_name  # For PDFs, assume same name
            
            # Generate book ID
            book_id = f"{language}_{discipline}_{level}_{pdf_file.stem}"
            
            # Get last update time
            last_updated = datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
            
            return BookEntry(
                book_id=book_id,
                book_name=book_name,
                original_name=original_name,
                language=language,
                discipline=discipline,
                level=level,
                format="pdf",
                file_path=str(pdf_file),
                pdf_file=str(pdf_file),
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF file {pdf_file}: {e}")
            return None
    
    def _extract_book_title_from_collection(self, collection_file: Path) -> tuple[str, str]:
        """Extract book title from collection.xml file."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(collection_file)
            root = tree.getroot()
            
            # Define namespaces
            namespaces = {
                'md': 'http://cnx.rice.edu/mdml'
            }
            
            # Find title element
            title_elem = root.find('.//md:title', namespaces)
            if title_elem is not None and title_elem.text:
                original_title = title_elem.text.strip()
                
                # Apply translation if needed (integrate with translation service)
                english_title = self._translate_title_to_english(original_title)
                
                return english_title, original_title
            
        except Exception as e:
            logger.debug(f"Error extracting title from {collection_file}: {e}")
        
        # Fallback to filename
        book_name = collection_file.stem.replace('-', ' ').replace('_', ' ').title()
        return book_name, book_name
    
    def _translate_title_to_english(self, title: str) -> str:
        """Translate title to English if needed."""
        try:
            # Import translation service if available
            from core.translation_service import TranslationService
            
            translator = TranslationService()
            # Try to detect language and translate
            # For now, return as-is, but this can be enhanced
            return translator.translate_book_title(title, "unknown")
            
        except ImportError:
            # Translation service not available, return as-is
            return title
        except Exception:
            # Translation failed, return as-is
            return title
    
    def save_book_list(self, output_dir: Path) -> tuple[Path, Path]:
        """Save the book list to TSV and JSON formats."""
        tsv_path = output_dir / "BookList.tsv"
        json_path = output_dir / "BookList.json"
        
        # Save TSV format
        with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
            if self.books:
                fieldnames = list(self.books[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                for book in self.books:
                    writer.writerow(book.to_dict())
        
        # Save JSON format
        with open(json_path, 'w', encoding='utf-8') as f:
            books_data = [book.to_dict() for book in self.books]
            json.dump(books_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.books)} books to {tsv_path} and {json_path}")
        return tsv_path, json_path


def generate_book_list(books_dir: Path) -> tuple[Path, Path]:
    """
    Generate BookList.tsv and BookList.json files.
    
    Args:
        books_dir: Path to the Books directory
        
    Returns:
        Tuple of (tsv_path, json_path)
    """
    generator = BookListGenerator(books_dir)
    
    # Scan for books
    books = generator.scan_books_directory()
    generator.books = books
    
    # Save to files
    tsv_path, json_path = generator.save_book_list(books_dir)
    
    return tsv_path, json_path


if __name__ == "__main__":
    # Test script
    import sys
    
    books_dir = Path("./Books")
    if len(sys.argv) > 1:
        books_dir = Path(sys.argv[1])
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        tsv_path, json_path = generate_book_list(books_dir)
        print(f"✅ Generated book list files:")
        print(f"   📄 TSV: {tsv_path}")
        print(f"   📄 JSON: {json_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)