#!/usr/bin/env python3
"""
Example: How to Read OpenStax Textbooks from ./Books Directory

This example demonstrates how to use the existing core modules to read
and extract content from OpenStax textbook repositories stored in ./Books.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import OpenBooksConfig
from core.text_extractor import TextExtractor
from core.toc_extractor import TOCExtractor

def read_book_example():
    """Example of reading an OpenStax textbook from the Books directory."""
    
    # Initialize configuration
    config = OpenBooksConfig()
    
    # Initialize extractors
    text_extractor = TextExtractor(config)
    toc_extractor = TOCExtractor()
    
    # Example: Read a physics textbook
    books_dir = Path("./Books")
    
    # Find an available physics book
    physics_books = []
    if books_dir.exists():
        physics_paths = books_dir.glob("*/Physics/*/osbooks-*")
        physics_books = list(physics_paths)
    
    if not physics_books:
        print("No physics textbooks found in ./Books directory")
        print("Run 'python GetOpenBooks.py --subjects Physics' first to download books")
        return
    
    # Use the first physics book found
    book_path = physics_books[0]
    print(f"Reading textbook: {book_path.name}")
    print(f"Full path: {book_path}")
    
    # Extract Table of Contents
    print("\n=== TABLE OF CONTENTS ===")
    try:
        # Look for collection XML file
        collections_dir = book_path / 'collections'
        if collections_dir.exists():
            collection_files = list(collections_dir.glob('*.xml'))
            if collection_files:
                collection_file = collection_files[0]
                print(f"Found collection file: {collection_file.name}")
                
                # Extract language and discipline from path for TOC extractor
                path_parts = book_path.parts
                language = path_parts[-4] if len(path_parts) >= 4 else "english"
                discipline = path_parts[-3] if len(path_parts) >= 3 else "Physics"
                level = path_parts[-2] if len(path_parts) >= 2 else "University"
                
                toc = toc_extractor.extract_toc(collection_file, language, discipline, level)
                if toc:
                    print(f"Book Title: {toc.book_title}")
                    print(f"Language: {toc.language}")
                    print(f"Discipline: {toc.discipline}")
                    print(f"Level: {toc.level}")
                    print(f"Total Entries: {len(toc.entries)}")
                    
                    # Display first few TOC entries
                    print("\nFirst few chapters/sections:")
                    for i, entry in enumerate(toc.entries[:5]):
                        indent = "  " * (entry.level - 1)
                        section_num = entry.section_number or f"{i+1}"
                        print(f"{indent}{section_num}. {entry.title}")
                else:
                    print("Could not extract table of contents")
            else:
                print("No collection XML files found")
        else:
            print("No collections directory found")
    except Exception as e:
        print(f"Error extracting TOC: {e}")
    
    # Extract Full Text Content
    print("\n=== TEXT CONTENT ===")
    try:
        content = text_extractor.extract_content(str(book_path))
        if content:
            print(f"Format: {content.format_type}")
            print(f"Title: {content.title}")
            print(f"Authors: {', '.join(content.authors)}")
            print(f"Number of chapters: {len(content.chapters)}")
            print(f"Total text length: {len(content.raw_text)} characters")
            
            # Show first paragraph of text
            if content.raw_text:
                first_paragraph = content.raw_text[:500] + "..." if len(content.raw_text) > 500 else content.raw_text
                print(f"\nFirst paragraph:\n{first_paragraph}")
            
            # Show mathematical notation if found
            if content.mathematical_notation:
                print(f"\nMathematical formulas found: {len(content.mathematical_notation)}")
                for i, formula in enumerate(content.mathematical_notation[:3]):
                    print(f"  Formula {i+1}: {formula}")
            
            # Show images if found
            if content.images:
                print(f"\nImages found: {len(content.images)}")
                for i, image in enumerate(content.images[:3]):
                    print(f"  Image {i+1}: {image.get('path', 'Unknown path')}")
        else:
            print("Could not extract text content")
    except Exception as e:
        print(f"Error extracting content: {e}")

def list_available_books():
    """List all available textbooks in the Books directory."""
    
    books_dir = Path("./Books")
    if not books_dir.exists():
        print("./Books directory not found")
        print("Run 'python GetOpenBooks.py' to download textbooks first")
        return
    
    print("=== AVAILABLE TEXTBOOKS ===")
    
    # Group by language and subject
    books_by_language = {}
    
    for language_dir in books_dir.iterdir():
        if not language_dir.is_dir():
            continue
            
        books_by_language[language_dir.name] = {}
        
        for subject_dir in language_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            books_by_language[language_dir.name][subject_dir.name] = []
            
            for level_dir in subject_dir.iterdir():
                if not level_dir.is_dir():
                    continue
                    
                for book_dir in level_dir.iterdir():
                    if book_dir.is_dir() and book_dir.name.startswith('osbooks-'):
                        books_by_language[language_dir.name][subject_dir.name].append({
                            'name': book_dir.name,
                            'level': level_dir.name,
                            'path': book_dir
                        })
    
    # Display organized list
    for language, subjects in books_by_language.items():
        if not subjects:
            continue
            
        print(f"\n{language.upper()}:")
        for subject, books in subjects.items():
            if not books:
                continue
                
            print(f"  {subject}:")
            for book in books:
                print(f"    📚 {book['name']} ({book['level']})")
                print(f"       Path: {book['path']}")

def search_in_book(book_path: Path, search_term: str):
    """Search for a term within a specific textbook."""
    
    print(f"\n=== SEARCHING FOR '{search_term}' ===")
    
    config = OpenBooksConfig()
    text_extractor = TextExtractor(config)
    
    try:
        content = text_extractor.extract_content(str(book_path))
        if not content:
            print("Could not extract content for searching")
            return
        
        # Simple text search
        raw_text = content.raw_text.lower()
        search_term_lower = search_term.lower()
        
        if search_term_lower in raw_text:
            # Find context around matches
            matches = []
            start = 0
            while True:
                pos = raw_text.find(search_term_lower, start)
                if pos == -1:
                    break
                    
                # Get context (100 chars before and after)
                context_start = max(0, pos - 100)
                context_end = min(len(raw_text), pos + len(search_term) + 100)
                context = content.raw_text[context_start:context_end]
                
                matches.append({
                    'position': pos,
                    'context': context
                })
                
                start = pos + 1
                if len(matches) >= 5:  # Limit to first 5 matches
                    break
            
            print(f"Found {len(matches)} matches:")
            for i, match in enumerate(matches):
                print(f"\nMatch {i+1}:")
                print(f"  {match['context']}")
        else:
            print(f"No matches found for '{search_term}'")
            
    except Exception as e:
        print(f"Error searching: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenStax Textbook Reader Example")
    parser.add_argument("--list", action="store_true", help="List available textbooks")
    parser.add_argument("--read", action="store_true", help="Read example textbook")
    parser.add_argument("--search", type=str, help="Search for term in first physics book")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_books()
    elif args.search:
        books_dir = Path("./Books")
        physics_books = list(books_dir.glob("*/Physics/*/osbooks-*"))
        if physics_books:
            search_in_book(physics_books[0], args.search)
        else:
            print("No physics books found to search in")
    else:
        read_book_example()