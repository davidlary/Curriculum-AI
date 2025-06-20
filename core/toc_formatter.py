"""
Table of Contents Formatter - Core functionality for formatting and saving TOC data.

Goal: Provide standardized formatting and export capabilities for table of contents data
in multiple formats (TSV, JSON) with consistent structure.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from .toc_extractor import BookTOC, TOCEntry

logger = logging.getLogger(__name__)


class TOCFormatter:
    """
    Goal: Format and save table of contents data in multiple formats.
    
    Provides standardized export capabilities for TOC data with support for
    TSV and JSON formats, maintaining data integrity and readability.
    """
    
    def __init__(self):
        """Initialize TOC formatter."""
        logger.info("TOCFormatter initialized")
    
    def save_toc(self, toc: BookTOC, output_dir: Path, formats: List[str] = None) -> Dict[str, Path]:
        """
        Goal: Save table of contents in specified formats.
        
        Args:
            toc: BookTOC object to save
            output_dir: Directory to save files
            formats: List of formats to save ('tsv', 'json'). If None, saves both.
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = ['tsv', 'json']
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Generate base filename from book title (sanitized)
        base_filename = self._sanitize_filename(toc.book_title)
        
        try:
            if 'tsv' in formats:
                tsv_path = self._save_tsv(toc, output_dir, base_filename)
                saved_files['tsv'] = tsv_path
                logger.info(f"Saved TSV: {tsv_path}")
            
            if 'json' in formats:
                json_path = self._save_json(toc, output_dir, base_filename)
                saved_files['json'] = json_path
                logger.info(f"Saved JSON: {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving TOC for {toc.book_title}: {e}")
            raise
        
        return saved_files
    
    def _save_tsv(self, toc: BookTOC, output_dir: Path, base_filename: str) -> Path:
        """
        Goal: Save table of contents as TSV (Tab-Separated Values) file.
        
        TSV format provides easy import into spreadsheets and databases while
        maintaining readability and data structure.
        """
        tsv_path = output_dir / f"{base_filename}.tsv"
        
        # Define TSV headers
        headers = [
            'entry_id',
            'title',
            'level',
            'section_number',
            'page_number',
            'parent_id'
        ]
        
        with open(tsv_path, 'w', newline='', encoding='utf-8') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            
            # Write headers
            writer.writerow(headers)
            
            # Write TOC entries
            for entry in toc.entries:
                self._write_entry_tsv(writer, entry)
        
        return tsv_path
    
    def _write_entry_tsv(self, writer, entry: TOCEntry):
        """
        Goal: Write a single TOC entry and its children to TSV.
        
        Recursively writes entry and all child entries maintaining hierarchy.
        """
        # Write current entry
        row = [
            entry.entry_id,
            entry.title,
            entry.level,
            entry.section_number or '',
            entry.page_number or '',
            entry.parent_id or ''
        ]
        writer.writerow(row)
        
        # Write children recursively
        for child in entry.children:
            self._write_entry_tsv(writer, child)
    
    def _save_json(self, toc: BookTOC, output_dir: Path, base_filename: str) -> Path:
        """
        Goal: Save table of contents as JSON file.
        
        JSON format provides complete data structure preservation with
        hierarchical relationships and metadata.
        """
        json_path = output_dir / f"{base_filename}.json"
        
        # Convert to dictionary with complete metadata
        toc_data = toc.to_dict()
        
        # Add formatting metadata
        toc_data['format_info'] = {
            'format_version': '1.0',
            'generated_by': 'OpenBooks TOC Extractor',
            'structure': 'hierarchical'
        }
        
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(toc_data, json_file, indent=2, ensure_ascii=False)
        
        return json_path
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Goal: Sanitize filename for safe file system usage.
        
        Removes or replaces characters that may cause issues in file systems.
        """
        # Replace problematic characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')
        
        # Remove extra spaces and trim
        safe_filename = ' '.join(safe_filename.split())
        
        # Limit length
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]
        
        # Ensure not empty
        if not safe_filename:
            safe_filename = 'untitled'
        
        return safe_filename
    
    def load_toc_json(self, json_path: Path) -> BookTOC:
        """
        Goal: Load table of contents from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            BookTOC object reconstructed from JSON data
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            # Reconstruct TOC entries
            entries = []
            for entry_data in data.get('entries', []):
                entry = self._dict_to_toc_entry(entry_data)
                entries.append(entry)
            
            # Create BookTOC object
            toc = BookTOC(
                book_title=data.get('book_title', ''),
                language=data.get('language', ''),
                discipline=data.get('discipline', ''),
                level=data.get('level', ''),
                file_path=data.get('file_path', ''),
                entries=entries,
                extraction_method=data.get('extraction_method', ''),
                total_entries=data.get('total_entries', 0)
            )
            
            logger.info(f"Loaded TOC from {json_path}: {toc.total_entries} entries")
            return toc
            
        except Exception as e:
            logger.error(f"Error loading TOC from {json_path}: {e}")
            raise
    
    def _dict_to_toc_entry(self, entry_data: Dict[str, Any]) -> TOCEntry:
        """
        Goal: Convert dictionary data to TOCEntry object.
        
        Recursively reconstructs TOCEntry objects from dictionary data.
        """
        # Create children first
        children = []
        for child_data in entry_data.get('children', []):
            child = self._dict_to_toc_entry(child_data)
            children.append(child)
        
        # Create TOCEntry
        entry = TOCEntry(
            title=entry_data.get('title', ''),
            level=entry_data.get('level', 1),
            page_number=entry_data.get('page_number'),
            section_number=entry_data.get('section_number'),
            parent_id=entry_data.get('parent_id'),
            entry_id=entry_data.get('entry_id', ''),
            children=children
        )
        
        return entry
    
    def create_combined_report(self, tocs: List[BookTOC], output_path: Path) -> Path:
        """
        Goal: Create combined report from multiple TOCs.
        
        Generates summary report with statistics and overview of all processed books.
        """
        try:
            report_data = {
                'summary': {
                    'total_books': len(tocs),
                    'total_entries': sum(toc.total_entries for toc in tocs),
                    'languages': list(set(toc.language for toc in tocs)),
                    'disciplines': list(set(toc.discipline for toc in tocs)),
                    'levels': list(set(toc.level for toc in tocs))
                },
                'books': []
            }
            
            # Add individual book summaries
            for toc in tocs:
                book_summary = {
                    'title': toc.book_title,
                    'language': toc.language,
                    'discipline': toc.discipline,
                    'level': toc.level,
                    'entries_count': toc.total_entries,
                    'extraction_method': toc.extraction_method,
                    'file_path': toc.file_path
                }
                report_data['books'].append(book_summary)
            
            # Sort books by discipline, then level, then title
            report_data['books'].sort(key=lambda x: (x['discipline'], x['level'], x['title']))
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as report_file:
                json.dump(report_data, report_file, indent=2, ensure_ascii=False)
            
            logger.info(f"Created combined report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating combined report: {e}")
            raise