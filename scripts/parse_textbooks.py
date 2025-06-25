import os
import sys
import json
import re
from pathlib import Path
import logging

# Add the core module to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.text_extractor import TextExtractor
from core.toc_extractor import TOCExtractor
from core.config import OpenBooksConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOOKS_DIR = Path("Books")
CHUNKS_DIR = Path("Chunks")
CHUNKS_DIR.mkdir(exist_ok=True)

def extract_title_from_content(content: str) -> str:
    """Extract a meaningful title from content text."""
    if not content:
        return None
    
    lines = content.split('\n')
    
    # Patterns to identify good titles
    title_patterns = [
        r'^([A-Z][A-Za-z\s]{10,80}[?!.]?)\s*$',  # Capitalized sentence ending with punctuation
        r'^([A-Z][A-Za-z\s&\'-]{5,60})\s*$',  # Capitalized phrase
        r'Introduction[:\s]*([A-Z][^.!?]*)',  # Introduction: Title
        r'Chapter\s+\d+[:\s]*([A-Z][^.!?]*)',  # Chapter N: Title
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+)*)\s*$',  # Title Case phrases
    ]
    
    # Skip common non-title patterns
    skip_patterns = [
        r'teacher support', r'learning objectives', r'by the end',
        r'section \d+', r'figure \d+', r'table \d+', r'example \d+',
        r'^\d+\.', r'page \d+', r'copyright', r'all rights reserved'
    ]
    
    for line in lines[:15]:  # Check first 15 lines
        line = line.strip()
        if len(line) < 5 or len(line) > 100:
            continue
            
        # Skip lines with skip patterns
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
        
        # Try title patterns
        for pattern in title_patterns:
            match = re.search(pattern, line)
            if match:
                title = match.group(1).strip()
                if title and len(title) > 5:
                    return title
    
    # Fallback: look for any capitalized substantial content
    for line in lines[:10]:
        line = line.strip()
        if (len(line) > 10 and len(line) < 80 and 
            line[0].isupper() and 
            not any(char in line for char in ['(', ')', '@', 'http', 'www'])):
            return line
    
    return None

def chunk_extracted_content(extracted_content, max_chunk_size=1000, overlap=100):
    """
    Intelligently chunk extracted content preserving structure and context.
    
    Args:
        extracted_content: ExtractedContent object from TextExtractor
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
    
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    
    if not extracted_content or not extracted_content.chapters:
        logger.warning("No chapters found in extracted content")
        return chunks
    
    chunk_id = 0
    
    for chapter_idx, chapter in enumerate(extracted_content.chapters):
        chapter_title = chapter.get('title', f'Chapter {chapter_idx + 1}')
        chapter_content = chapter.get('content', '')
        chapter_number = chapter.get('number', str(chapter_idx + 1))
        
        # If title is generic, try to extract a better one from content
        if chapter_title in ['Untitled Chapter', f'Chapter {chapter_idx + 1}', 'Content Section']:
            better_title = extract_title_from_content(chapter_content)
            if better_title:
                chapter_title = better_title
        
        if not chapter_content.strip():
            continue
        
        # For shorter chapters, keep as single chunk
        if len(chapter_content) <= max_chunk_size:
            chunks.append({
                'chunk_id': chunk_id,
                'chapter_number': chapter_number,
                'chapter_title': chapter_title,
                'content': chapter_content,
                'chunk_type': 'full_chapter',
                'subsections': chapter.get('subsections', []),
                'formulas': chapter.get('formulas', [])
            })
            chunk_id += 1
        else:
            # Split longer chapters into meaningful chunks
            chapter_chunks = split_chapter_content(
                chapter_content, 
                chapter_title, 
                chapter_number,
                max_chunk_size, 
                overlap
            )
            
            for i, chunk_content in enumerate(chapter_chunks):
                chunks.append({
                    'chunk_id': chunk_id,
                    'chapter_number': chapter_number,
                    'chapter_title': chapter_title,
                    'content': chunk_content,
                    'chunk_type': 'chapter_section',
                    'section_index': i,
                    'total_sections': len(chapter_chunks),
                    'subsections': chapter.get('subsections', []) if i == 0 else [],
                    'formulas': extract_formulas_from_text(chunk_content)
                })
                chunk_id += 1
    
    logger.info(f"Created {len(chunks)} content chunks from {len(extracted_content.chapters)} chapters")
    return chunks

def split_chapter_content(content, title, number, max_size, overlap):
    """
    Split chapter content into logical chunks preserving paragraph boundaries.
    """
    chunks = []
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph would exceed max size, start new chunk
        if current_chunk and len(current_chunk) + len(para) > max_size:
            chunks.append(current_chunk.strip())
            
            # Add overlap from end of previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_formulas_from_text(text):
    """Extract mathematical formulas from text content."""
    import re
    formulas = []
    
    # Basic formula patterns
    patterns = [
        r'\$\$([^$]+)\$\$',  # Display math
        r'\$([^$]+)\$',      # Inline math
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        formulas.extend([m.strip() for m in matches if m.strip()])
    
    return formulas

def parse_textbooks(discipline="Physics", language="english"):
    """
    Parse textbooks using proper TextExtractor and TOCExtractor for full content extraction.
    
    This replaces the previous simple file scanning approach with comprehensive
    textbook content extraction that handles OpenStax CNXML format properly.
    """
    books_path = BOOKS_DIR / language / discipline
    if not books_path.exists():
        logger.error(f"Path not found: {books_path}")
        available_disciplines = [d.name for d in (BOOKS_DIR / language).iterdir() if d.is_dir()]
        logger.info(f"Available disciplines: {available_disciplines}")
        return

    # Initialize extractors with config
    config = OpenBooksConfig()
    text_extractor = TextExtractor(config)
    toc_extractor = TOCExtractor()
    
    logger.info(f"Starting textbook parsing for {discipline} ({language})")
    
    for level_dir in books_path.iterdir():
        if not level_dir.is_dir():
            continue
            
        level = level_dir.name
        output_file = CHUNKS_DIR / f"{discipline}_{level}.jsonl"
        
        logger.info(f"Processing level: {level}")
        
        total_chunks = 0
        total_books = 0
        
        with open(output_file, "w", encoding="utf-8") as out_f:
            for repo in level_dir.iterdir():
                if not repo.is_dir():
                    continue
                    
                # Check if this is a git repository (OpenStax books)
                if not (repo / ".git").exists():
                    logger.debug(f"Skipping non-git directory: {repo}")
                    continue
                
                logger.info(f"📘 Processing textbook: {repo.name}")
                
                try:
                    # Extract full textbook content using TextExtractor
                    extracted_content = text_extractor.extract_content(str(repo))
                    
                    if not extracted_content:
                        logger.warning(f"Failed to extract content from {repo}")
                        continue
                    
                    # Also extract TOC for additional structure
                    collection_file = repo / "collections"
                    if collection_file.exists():
                        collection_xml = list(collection_file.glob("*.xml"))
                        if collection_xml:
                            toc = toc_extractor.extract_toc(
                                collection_xml[0], language, discipline, level
                            )
                            if toc:
                                logger.info(f"Extracted TOC with {toc.total_entries} entries")
                    
                    # Convert extracted content to chunks
                    content_chunks = chunk_extracted_content(extracted_content)
                    
                    if not content_chunks:
                        logger.warning(f"No content chunks created for {repo}")
                        continue
                    
                    # Write chunks to output file
                    for chunk in content_chunks:
                        record = {
                            "discipline": discipline,
                            "language": language,
                            "level": level,
                            "book_title": extracted_content.title,
                            "book_format": extracted_content.format_type,
                            "source_path": extracted_content.source_path,
                            "chunk_id": chunk['chunk_id'],
                            "chapter_number": chunk['chapter_number'],
                            "chapter_title": chunk['chapter_title'],
                            "chunk_type": chunk['chunk_type'],
                            "text": chunk['content'],
                            "formulas": chunk.get('formulas', []),
                            "subsections": chunk.get('subsections', []),
                            "extraction_stats": extracted_content.extraction_stats,
                            "content_hash": extracted_content.content_hash
                        }
                        
                        # Add section info for multi-part chapters
                        if chunk['chunk_type'] == 'chapter_section':
                            record['section_index'] = chunk['section_index']
                            record['total_sections'] = chunk['total_sections']
                        
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    total_chunks += len(content_chunks)
                    total_books += 1
                    
                    logger.info(f"✅ Extracted {len(content_chunks)} chunks from {repo.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {repo}: {e}")
                    continue
        
        logger.info(f"📊 Level {level} complete: {total_books} books, {total_chunks} chunks saved to {output_file}")

if __name__ == '__main__':
    parse_textbooks()