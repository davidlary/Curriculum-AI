#!/usr/bin/env python3
"""
Enhanced Curriculum AI Admin Panel with 7-Step Modular System Visualization

This enhanced version provides rich visualizations for:
1. Book Discovery visualization
2. TOC Extraction progress and results
3. Topic Normalization with semantic alignment visualization
4. Interactive pipeline management
5. Real-time quality metrics dashboards

Usage:
    streamlit run ui/enhanced_admin_app.py
"""

import streamlit as st
import json
import os
import subprocess
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import time
from collections import defaultdict, Counter

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import cache manager and quality validator
from core.cache_manager import get_cache_manager, CacheManager
from core.quality_validator import QualityValidator

# Configure logging
logger = logging.getLogger(__name__)

# Configure page (will be set in main() when run directly)

# Standardized Directory Structure
BASE_DIR = Path(".")
BOOKS_DIR = BASE_DIR / "Books"           # Book discovery and metadata
CHUNKS_DIR = BASE_DIR / "Chunks"         # Text chunks from processing
TOCS_DIR = BASE_DIR / "TOCs"             # Table of contents extraction
CURRICULUM_DIR = BASE_DIR / "Curriculum" # Final curriculum outputs
CACHE_DIR = BASE_DIR / "Cache"           # Temporary cache and processing
QUESTIONS_DIR = BASE_DIR / "Questions"   # Generated questions
EMBEDDINGS_DIR = BASE_DIR / "Embeddings" # Vector embeddings

class EnhancedCurriculumAdmin:
    """Enhanced admin class with 7-step modular system support."""
    
    def __init__(self):
        # Primary data directories
        self.data_dirs = {
            'books': BOOKS_DIR,
            'tocs': TOCS_DIR,
            'chunks': CHUNKS_DIR,
            'curriculum': CURRICULUM_DIR,
            'questions': QUESTIONS_DIR,
            'embeddings': EMBEDDINGS_DIR
        }
        
        # Cache directories for processing
        self.cache_dirs = {
            'books': CACHE_DIR / "Books",
            'tocs': CACHE_DIR / "TOCs", 
            'normalization': CACHE_DIR / "Normalization",
            'prerequisites': CACHE_DIR / "Prerequisites",
            'sequences': CACHE_DIR / "Sequences",
            'adaptivity': CACHE_DIR / "Adaptivity"
        }
        
        # Create all directories
        for dir_dict in [self.data_dirs, self.cache_dirs]:
            for directory in dir_dict.values():
                directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager and quality validator
        self.cache_manager = get_cache_manager(CACHE_DIR)
        self.quality_validator = QualityValidator(CACHE_DIR / "QualityValidation")
        
        self.step_commands = {
            "step1_discovery": ["python", "scripts/step1_book_discovery.py"],
            "step2_toc": ["python", "scripts/step2_toc_extraction.py"],
            "step3_classification": ["python", "scripts/step3_core_elective_classification.py"],
            "step4_hierarchy": ["python", "scripts/step4_six_level_hierarchy.py"],
            "step5_prerequisites": ["python", "scripts/step5_prerequisites_dependencies.py"],
            "step6_standards": ["python", "scripts/step6_standards_mapping.py"],
            "step7_export": ["python", "scripts/step7_multi_format_export.py"],
            "master_orchestrator": ["python", "scripts/master_curriculum_orchestrator.py"]
        }

    def get_available_disciplines(self) -> List[str]:
        """Get available disciplines from various sources."""
        disciplines = set()
        
        # From chunks
        if CHUNKS_DIR.exists():
            for chunk_file in CHUNKS_DIR.glob("*_*.jsonl"):
                discipline = chunk_file.stem.split('_')[0]
                disciplines.add(discipline)
        
        # From discovered books
        books_cache = self.cache_dirs['books']
        if books_cache.exists():
            for book_file in books_cache.glob("*_discovered.json"):
                discipline = book_file.stem.split('_')[0]
                disciplines.add(discipline)
        
        return sorted(list(disciplines))

    def _get_step_output_file(self, step: str, discipline: str, language: str) -> Optional[Path]:
        """Get the expected output file path for a given step using standardized directories."""
        file_mappings = {
            # Step 1: Book discovery -> Books directory
            "step1_discovery": BOOKS_DIR / f"{discipline}_{language}_books_discovered.json",
            # Step 2: TOC extraction -> TOCs directory  
            "step2_toc": TOCS_DIR / f"{discipline}_{language}_tocs_extracted.json",
            # Steps 3-7: Curriculum processing -> Curriculum directory
            "step3_classification": CURRICULUM_DIR / f"{discipline}_{language}_classified_curriculum.json",
            "step4_hierarchy": CURRICULUM_DIR / f"{discipline}_{language}_six_level_hierarchy.json",
            "step5_prerequisites": CURRICULUM_DIR / f"{discipline}_{language}_prerequisites_mapped.json",
            "step6_standards": CURRICULUM_DIR / f"{discipline}_{language}_standards_mapped.json",
            "step7_export": CURRICULUM_DIR / f"{discipline}_{language}_complete_curriculum.json",
            "master_orchestrator": CURRICULUM_DIR / f"{discipline}_{language}_complete_curriculum.json"
        }
        return file_mappings.get(step)

    def run_modular_step(self, step: str, discipline: str, language: str = "English", 
                        force_refresh: bool = False, **kwargs) -> Dict[str, Any]:
        """Run a specific step in the 7-step modular system."""
        if step not in self.step_commands:
            return {"success": False, "error": f"Unknown step: {step}"}
        
        # Build command with correct arguments based on step type
        cmd = self.step_commands[step].copy()
        
        # Steps 1-2 use --discipline and --language
        # Steps 3-7 use --input and --output
        if step in ["step1_discovery", "step2_toc"]:
            cmd.extend(["--discipline", discipline, "--language", language])
        else:
            # Steps 3-7 use input/output pattern with standardized directories
            input_output_mappings = {
                "step3_classification": {
                    "input": str(TOCS_DIR / f"{discipline}_{language}_tocs_extracted.json"),
                    "output": str(CURRICULUM_DIR / f"{discipline}_{language}_classified_curriculum.json")
                },
                "step4_hierarchy": {
                    "input": str(CURRICULUM_DIR / f"{discipline}_{language}_classified_curriculum.json"),
                    "output": str(CURRICULUM_DIR / f"{discipline}_{language}_six_level_hierarchy.json")
                },
                "step5_prerequisites": {
                    "input": str(CURRICULUM_DIR / f"{discipline}_{language}_six_level_hierarchy.json"),
                    "output": str(CURRICULUM_DIR / f"{discipline}_{language}_prerequisites_mapped.json")
                },
                "step6_standards": {
                    "input": str(CURRICULUM_DIR / f"{discipline}_{language}_prerequisites_mapped.json"),
                    "output": str(CURRICULUM_DIR / f"{discipline}_{language}_standards_mapped.json")
                },
                "step7_export": {
                    "input": str(CURRICULUM_DIR / f"{discipline}_{language}_standards_mapped.json"),
                    "output": str(CURRICULUM_DIR / f"{discipline}_{language}_complete_curriculum")
                },
                "master_orchestrator": {
                    "input": str(TOCS_DIR / f"{discipline}_{language}_tocs_extracted.json"),
                    "output": str(CURRICULUM_DIR)  # Directory for orchestrator
                }
            }
            
            # Add input/output arguments for steps 3-7
            if step in input_output_mappings:
                mapping = input_output_mappings[step]
                cmd.extend(["--input", mapping["input"]])
                cmd.extend(["--output", mapping["output"]])
            
            # Add config file for steps 3-7
            cmd.extend(["--config", "config/curriculum_config.json"])
        
        # Handle force refresh by deleting cached output files instead of passing --force-refresh
        # (since most step scripts don't support --force-refresh argument)
        if force_refresh:
            # Get the expected output file for this step and delete it to force regeneration
            output_file = self._get_step_output_file(step, discipline, language)
            if output_file and output_file.exists():
                try:
                    output_file.unlink()
                    logger.info(f"Deleted cached output file for force refresh: {output_file}")
                except Exception as e:
                    logger.warning(f"Could not delete cached file {output_file}: {e}")
        
        # Add --no-llm flag if LLM is disabled
        if kwargs.get('disable_llm', False):
            cmd.append("--no-llm")
        
        # Add step-specific arguments only (excluding openai_api_key which is handled via environment)
        step_specific_args = {
            "step1_discovery": [],  # Step 1 doesn't need additional args
            "step2_toc": [],  # Step 2 doesn't need additional args
            "step3_classification": [],  # Step 3 uses LLM via environment variable only
            "step4_hierarchy": [],  # Step 4 uses LLM via environment variable only
            "step5_prerequisites": [],  # Step 5 uses LLM via environment variable only
            "step6_standards": [],  # Step 6 uses LLM via environment variable only
            "step7_export": ["formats"],  # Step 7 should always export all formats
            "master_orchestrator": []  # Master orchestrator uses LLM via environment variable only
        }
        
        # Special handling for step7_export to always export all formats
        if step == "step7_export":
            cmd.extend(["--formats", "tsv", "json", "dot", "duckdb"])
        
        allowed_args = step_specific_args.get(step, [])
        for key, value in kwargs.items():
            if key in allowed_args and value and key != "formats":  # Skip formats for step7 since we handle it above
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Set environment
        env = os.environ.copy()
        if 'openai_api_key' in kwargs and kwargs['openai_api_key']:
            env["OPENAI_API_KEY"] = kwargs['openai_api_key']
        
        try:
            with st.spinner(f"Running {step}..."):
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=3600,  # 60 minute timeout for stable LLM processing
                    cwd=os.getcwd()  # Ensure correct working directory
                )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired as e:
            return {
                "success": False, 
                "error": f"Process timed out after {e.timeout} seconds",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Subprocess execution failed: {str(e)}",
                "command": " ".join(cmd)
            }

    def load_step_results(self, step: str, discipline: str, language: str = "English") -> Optional[Dict[str, Any]]:
        """Load results from a completed step using standardized directory structure."""
        # Use the same mapping as _get_step_output_file for consistency
        file_path = self._get_step_output_file(step, discipline, language)
        if not file_path or not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {step} results: {e}")
            return None

def create_book_discovery_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 1: Book Discovery."""
    st.subheader("📚 Step 1: Book Discovery Results")
    
    if not data or 'books' not in data:
        st.warning("No book discovery data available")
        return
    
    books = data['books']
    metrics = data.get('metrics', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Books Found", len(books))
    with col2:
        st.metric("Unique Sources", len(metrics.get('books_by_source', {})))
    with col3:
        st.metric("Educational Levels", len(metrics.get('books_by_level', {})))
    with col4:
        st.metric("Coverage Score", f"{metrics.get('coverage_completeness', 0):.1%}")
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Books by educational level
        level_data = metrics.get('books_by_level', {})
        if level_data:
            fig = px.bar(
                x=list(level_data.keys()),
                y=list(level_data.values()),
                title="Books by Educational Level",
                labels={'x': 'Educational Level', 'y': 'Number of Books'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Books by source
        source_data = metrics.get('books_by_source', {})
        if source_data:
            fig = px.pie(
                values=list(source_data.values()),
                names=list(source_data.keys()),
                title="Books by Source"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Quality distribution
    quality_dist = metrics.get('quality_distribution', {})
    if quality_dist:
        st.subheader("Quality Distribution")
        fig = px.bar(
            x=list(quality_dist.keys()),
            y=list(quality_dist.values()),
            title="Book Quality Distribution",
            color=list(quality_dist.keys()),
            color_discrete_map={'high': 'green', 'medium': 'orange', 'low': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed book list
    with st.expander("📖 Detailed Book List"):
        if books:
            books_df = pd.DataFrame(books)
            st.dataframe(
                books_df[['title', 'educational_level', 'source', 'quality_score', 'language']],
                use_container_width=True
            )

def create_topic_alignment_matrix(all_books: List[Dict]) -> Dict[str, Any]:
    """Create a topic alignment matrix for cross-book analysis."""
    from difflib import SequenceMatcher
    import re
    
    # Extract unique topics from all books
    all_topics = {}
    topic_sources = {}
    
    for book in all_books:
        book_key = f"{book['book_title']} ({book['level']})"
        toc_entries = book.get('toc_entries', [])
        
        for entry in toc_entries:
            topic_title = entry.get('title', '').strip()
            if topic_title and len(topic_title) > 3:  # Filter out very short titles
                # Normalize topic title for comparison
                normalized = re.sub(r'[^\w\s]', '', topic_title.lower()).strip()
                
                if normalized not in all_topics:
                    all_topics[normalized] = {
                        'canonical_title': topic_title,
                        'books': set(),
                        'levels': set(),
                        'variations': set()
                    }
                    topic_sources[normalized] = []
                
                all_topics[normalized]['books'].add(book_key)
                all_topics[normalized]['levels'].add(book['level'])
                all_topics[normalized]['variations'].add(topic_title)
                topic_sources[normalized].append({
                    'book': book_key,
                    'level': book['level'],
                    'title': topic_title,
                    'hierarchy_level': entry.get('level', 1)
                })
    
    # Calculate topic similarities and alignments
    alignment_matrix = {}
    cross_level_topics = {}
    
    for topic, data in all_topics.items():
        if len(data['levels']) > 1:  # Topic appears across multiple levels
            cross_level_topics[topic] = {
                'books': list(data['books']),
                'levels': list(data['levels']),
                'variations': list(data['variations']),
                'sources': topic_sources[topic]
            }
    
    return {
        'all_topics': all_topics,
        'cross_level_topics': cross_level_topics,
        'topic_sources': topic_sources,
        'total_topics': len(all_topics),
        'cross_level_count': len(cross_level_topics)
    }

def create_topic_similarity_heatmap(alignment_data: Dict[str, Any]) -> None:
    """Create a topic similarity heatmap visualization."""
    cross_level_topics = alignment_data.get('cross_level_topics', {})
    
    if not cross_level_topics:
        st.info("No cross-level topic alignments detected. This may indicate that books use different terminology for similar concepts.")
        return
    
    # Create alignment summary
    st.success(f"Found {len(cross_level_topics)} topics that appear across multiple educational levels!")
    
    # Display cross-level topics
    alignment_df = []
    for topic, data in cross_level_topics.items():
        alignment_df.append({
            'Topic': data['variations'][0] if data['variations'] else topic,
            'Educational Levels': ', '.join(sorted(data['levels'])),
            'Books': len(data['books']),
            'Variations': len(data['variations'])
        })
    
    if alignment_df:
        df = pd.DataFrame(alignment_df)
        st.dataframe(df, use_container_width=True)
        
        # Show detailed variations for a selected topic
        if len(alignment_df) > 0:
            st.subheader("🔍 Topic Variation Analysis")
            selected_topic_idx = st.selectbox(
                "Select a topic to see its variations across books:",
                range(len(alignment_df)),
                format_func=lambda x: alignment_df[x]['Topic']
            )
            
            if selected_topic_idx is not None:
                topic_key = list(cross_level_topics.keys())[selected_topic_idx]
                topic_data = cross_level_topics[topic_key]
                
                st.write(f"**Topic:** {topic_data['variations'][0]}")
                st.write(f"**Appears in {len(topic_data['levels'])} educational levels:** {', '.join(sorted(topic_data['levels']))}")
                
                # Show variations by book
                for source in topic_data['sources']:
                    st.write(f"- **{source['book']}**: \"{source['title']}\" (Level {source['hierarchy_level']})")

def create_tabbed_book_interface(books: List[Dict]) -> None:
    """Create a tabbed interface showing complete TOCs for each book."""
    
    if not books:
        st.warning("No books available for display.")
        return
    
    # Identify potential duplicate books
    book_issues = []
    book_titles = [book.get('book_title', 'Unknown') for book in books]
    
    # Check for extraction issues and duplicates
    for book in books:
        title = book.get('book_title', 'Unknown')
        entries = book.get('toc_entries', [])
        
        issues = []
        if len(entries) <= 5:
            issues.append(f"⚠️ Very few TOC entries ({len(entries)})")
        
        # Check for duplicate content
        if title == "Physics (High School PDF)" and len(entries) == 1:
            issues.append("🔍 Possible extraction failure")
        
        if "College Physics" in title:
            similar_books = [b for b in books if "College Physics" in b.get('book_title', '') and b != book]
            if similar_books:
                issues.append(f"📋 Potential duplicate with {len(similar_books)} other College Physics book(s)")
        
        if issues:
            book_issues.append({
                'title': title,
                'issues': issues,
                'entries_count': len(entries)
            })
    
    # Show issues summary if any
    if book_issues:
        with st.expander("⚠️ Book Quality Issues Detected", expanded=False):
            st.warning("The following books may have extraction or duplication issues:")
            for issue in book_issues:
                st.write(f"**{issue['title']}** ({issue['entries_count']} entries)")
                for prob in issue['issues']:
                    st.write(f"  • {prob}")
    
    # Create tabs for each book
    st.info(f"📖 Browse complete Table of Contents for all {len(books)} books using the tabs below")
    
    # Create tab names (shortened for display)
    tab_names = []
    for book in books:
        title = book.get('book_title', 'Unknown')
        level = book.get('level', 'unknown')
        entries_count = len(book.get('toc_entries', []))
        
        # Shorten title for tab display
        short_title = title
        if len(title) > 25:
            short_title = title[:22] + "..."
        
        # Add warning emoji for problematic books
        warning = ""
        if any(issue['title'] == title for issue in book_issues):
            warning = "⚠️ "
        
        tab_names.append(f"{warning}{short_title} ({entries_count})")
    
    # Create the tabs
    tabs = st.tabs(tab_names)
    
    # Display content for each tab
    for i, (tab, book) in enumerate(zip(tabs, books)):
        with tab:
            display_complete_book_toc(book, books, i)


def display_complete_book_toc(book: Dict, all_books: List[Dict], book_index: int) -> None:
    """Display complete TOC for a single book in a tab."""
    
    title = book.get('book_title', 'Unknown')
    level = book.get('level', 'unknown')
    toc_entries = book.get('toc_entries', [])
    extraction_method = book.get('extraction_method', 'unknown')
    
    # Book header with metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Book Title", title)
        st.write(f"**Educational Level:** {level.replace('_', ' ').title()}")
    
    with col2:
        st.metric("📋 TOC Entries", len(toc_entries))
        st.write(f"**Extraction Method:** {extraction_method}")
    
    with col3:
        if len(toc_entries) <= 5:
            st.error("⚠️ Very few entries - Possible extraction issue")
        elif len(toc_entries) < 50:
            st.warning("⚠️ Fewer entries than expected")
        else:
            st.success("✅ Good TOC coverage")
    
    # Check for specific issues
    if title == "Physics (High School PDF)" and len(toc_entries) == 1:
        st.error("🔍 **Extraction Issue Detected**: This book has only 1 TOC entry, suggesting the PDF was not properly processed.")
        st.info("💡 **Recommendation**: Re-run Step 2 (TOC Extraction) with enhanced PDF processing for this book.")
    
    # Check for duplicates
    college_physics_books = [b for b in all_books if "College Physics" in b.get('book_title', '')]
    if len(college_physics_books) > 1 and "College Physics" in title:
        st.warning(f"📋 **Potential Duplicate**: Found {len(college_physics_books)} College Physics variants with similar content.")
        duplicate_titles = [b.get('book_title', 'Unknown') for b in college_physics_books]
        st.write("Similar books:")
        for dup_title in duplicate_titles:
            if dup_title != title:
                st.write(f"  • {dup_title}")
    
    # Display complete TOC
    if toc_entries:
        st.subheader(f"📖 Complete Table of Contents")
        
        # TOC navigation and search
        col1, col2 = st.columns(2)
        
        with col1:
            search_toc = st.text_input(
                f"🔍 Search within {title}:",
                placeholder="Enter topic keywords...",
                key=f"search_toc_{book_index}"
            )
        
        with col2:
            show_levels = st.multiselect(
                "Show levels:",
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3],
                key=f"levels_{book_index}"
            )
        
        # Filter TOC entries
        filtered_entries = toc_entries
        
        if search_toc:
            filtered_entries = [
                entry for entry in toc_entries
                if search_toc.lower() in entry.get('title', '').lower()
            ]
        
        if show_levels:
            filtered_entries = [
                entry for entry in filtered_entries
                if entry.get('level', 1) in show_levels
            ]
        
        st.write(f"**Showing {len(filtered_entries)} of {len(toc_entries)} TOC entries:**")
        
        # Create comprehensive TOC table
        toc_data = []
        for i, entry in enumerate(filtered_entries):
            level = entry.get('level', 1)
            entry_title = entry.get('title', 'Unknown')
            page_num = entry.get('page_number', 'N/A')
            section_num = entry.get('section_number', 'N/A')
            
            # Create indentation for hierarchy
            indent = "    " * (level - 1)
            level_icon = {1: "📘", 2: "📄", 3: "📝", 4: "•", 5: "◦"}.get(level, "•")
            
            toc_data.append({
                'Index': i + 1,
                'Level': level,
                'Title': f"{indent}{level_icon} {entry_title}",
                'Page': page_num,
                'Section': section_num
            })
        
        # Display as dataframe
        if toc_data:
            toc_df = pd.DataFrame(toc_data)
            
            # Apply styling based on level
            def highlight_levels(row):
                level = row['Level']
                if level == 1:
                    return ['background-color: #e3f2fd'] * len(row)
                elif level == 2:
                    return ['background-color: #fff3e0'] * len(row)
                elif level == 3:
                    return ['background-color: #f1f8e9'] * len(row)
                else:
                    return ['background-color: #fafafa'] * len(row)
            
            styled_toc = toc_df.style.apply(highlight_levels, axis=1)
            st.dataframe(styled_toc, use_container_width=True, height=400)
            
            # Export option
            if st.button(f"📥 Export {title} TOC", key=f"export_{book_index}"):
                csv_data = toc_df.to_csv(index=False)
                st.download_button(
                    label=f"Download {title} TOC as CSV",
                    data=csv_data,
                    file_name=f"{title.replace(' ', '_')}_TOC.csv",
                    mime="text/csv",
                    key=f"download_{book_index}"
                )
        
        else:
            st.warning("No TOC entries match your search criteria.")
    
    else:
        st.error("❌ No TOC entries found for this book.")
        if title == "Physics (High School PDF)":
            st.info("🔧 **Fix Suggestion**: This appears to be a PDF processing issue. The book likely contains content but the TOC extraction failed.")


def create_cross_linked_toc_tables(books: List[Dict]) -> None:
    """Create independent but cross-linked tables, one per book, for easy navigation."""
    if not books:
        st.warning("No books available for comparison")
        return
    
    try:
        # Create navigation interface
        st.info("📚 **Cross-Linked Book Navigation**: Select any book below to view its detailed TOC. Each table shows the complete structure with cross-references to related topics in other books.")
        
        # Book selection tabs
        book_names = [f"{book['book_title']} ({book['level'].replace('_', ' ').title()})" for book in books]
        
        # Create columns for book selection buttons
        cols = st.columns(min(len(books), 4))  # Max 4 columns
        selected_book_idx = 0
        
        # Use session state to remember selection
        if 'selected_book_idx' not in st.session_state:
            st.session_state.selected_book_idx = 0
            
        for i, book in enumerate(books):
            col_idx = i % len(cols)
            with cols[col_idx]:
                level_color = "#28a745" if book['level'] == 'high_school' else "#007bff"
                elective_text = " 🟡" if 'astronomy' in book.get('book_title', '').lower() else ""
                
                button_label = f"{book['book_title'][:20]}{'...' if len(book['book_title']) > 20 else ''}{elective_text}"
                
                if st.button(button_label, key=f"book_btn_{i}", help=f"View TOC for {book['book_title']}"):
                    st.session_state.selected_book_idx = i
        
        selected_book_idx = st.session_state.selected_book_idx
        selected_book = books[selected_book_idx]
        
        # Display selected book information
        st.markdown("---")
        
        # Book header with enhanced information
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            level_badge = f'<span style="background-color: {"#28a745" if selected_book["level"] == "high_school" else "#007bff"}; color: white; padding: 4px 12px; border-radius: 15px; font-size: 12px;">{selected_book["level"].replace("_", " ").title()}</span>'
            elective_badge = ""
            if 'astronomy' in selected_book.get('book_title', '').lower():
                elective_badge = ' <span style="background-color: #ffc107; color: black; padding: 3px 8px; border-radius: 10px; font-size: 11px; margin-left: 8px;">ELECTIVE</span>'
            
            st.markdown(f"### 📖 {selected_book['book_title']}")
            st.markdown(f"{level_badge}{elective_badge}", unsafe_allow_html=True)
        
        with col2:
            st.metric("TOC Entries", len(selected_book.get('toc_entries', [])))
            st.metric("Total Topics", selected_book.get('total_topics', 0))
        
        with col3:
            st.metric("Book Index", f"{selected_book_idx + 1} of {len(books)}")
            st.write(f"**Source:** {selected_book.get('source', 'Unknown')}")
        
        # Navigation arrows
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Previous Book", disabled=selected_book_idx == 0):
                st.session_state.selected_book_idx = max(0, selected_book_idx - 1)
                st.rerun()
        
        with nav_col3:
            if st.button("Next Book ➡️", disabled=selected_book_idx == len(books) - 1):
                st.session_state.selected_book_idx = min(len(books) - 1, selected_book_idx + 1)
                st.rerun()
        
        with nav_col2:
            # Quick jump selector
            jump_options = [f"{i+1}. {book['book_title'][:30]}{'...' if len(book['book_title']) > 30 else ''}" for i, book in enumerate(books)]
            jump_selection = st.selectbox(
                "Quick Jump to Book:", 
                jump_options, 
                index=selected_book_idx,
                key="book_jump_select"
            )
            if jump_options.index(jump_selection) != selected_book_idx:
                st.session_state.selected_book_idx = jump_options.index(jump_selection)
                st.rerun()
        
        # Display the TOC table for selected book
        st.markdown("---")
        create_individual_book_toc_table(selected_book, books, selected_book_idx)
        
        # Cross-reference section
        st.markdown("---")
        create_cross_reference_section(selected_book, books, selected_book_idx)
        
    except Exception as e:
        st.error(f"Error creating cross-linked tables: {e}")
        st.info("Falling back to simple book list:")
        for i, book in enumerate(books):
            st.write(f"{i+1}. **{book.get('book_title', 'Unknown')}** ({book.get('level', 'Unknown level')}) - {len(book.get('toc_entries', []))} entries")

def create_individual_book_toc_table(book: Dict, all_books: List[Dict], book_idx: int) -> None:
    """Create a detailed TOC table for an individual book with enhanced features."""
    toc_entries = book.get('toc_entries', [])
    
    if not toc_entries:
        st.warning(f"No TOC entries found for {book['book_title']}")
        return
    
    st.subheader(f"📋 Table of Contents: {book['book_title']}")
    
    # TOC Statistics
    level_counts = {}
    for entry in toc_entries:
        level = entry.get('level', 1)
        level_counts[level] = level_counts.get(level, 0) + 1
    
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total Entries", len(toc_entries))
    with stat_cols[1]:
        st.metric("Chapters", level_counts.get(1, 0))
    with stat_cols[2]:
        st.metric("Sections", level_counts.get(2, 0))
    with stat_cols[3]:
        st.metric("Subsections", level_counts.get(3, 0) + level_counts.get(4, 0))
    
    # Search and filter options
    search_term = st.text_input("🔍 Search TOC entries:", placeholder="Enter topic name...")
    
    filter_cols = st.columns(3)
    with filter_cols[0]:
        level_filter = st.selectbox("Filter by Level:", ["All", "1 (Chapters)", "2 (Sections)", "3+ (Subsections)"])
    with filter_cols[1]:
        show_count = st.slider("Show entries:", 10, min(len(toc_entries), 200), min(50, len(toc_entries)))
    
    # Filter entries
    filtered_entries = toc_entries
    if search_term:
        filtered_entries = [entry for entry in toc_entries if search_term.lower() in entry.get('title', '').lower()]
    
    if level_filter != "All":
        if level_filter == "1 (Chapters)":
            filtered_entries = [entry for entry in filtered_entries if entry.get('level', 1) == 1]
        elif level_filter == "2 (Sections)":
            filtered_entries = [entry for entry in filtered_entries if entry.get('level', 1) == 2]
        elif level_filter == "3+ (Subsections)":
            filtered_entries = [entry for entry in filtered_entries if entry.get('level', 1) >= 3]
    
    # Display results count
    if search_term or level_filter != "All":
        st.info(f"Showing {min(len(filtered_entries), show_count)} of {len(filtered_entries)} filtered entries (from {len(toc_entries)} total)")
    
    # Create the TOC table
    display_entries = filtered_entries[:show_count]
    
    if display_entries:
        # Create DataFrame for better display
        toc_data = []
        for i, entry in enumerate(display_entries):
            level = entry.get('level', 1)
            title = entry.get('title', 'Unknown')
            
            # Create indentation
            indent = "　" * (level - 1)  # Using full-width space for better alignment
            
            # Level indicator
            if level == 1:
                level_icon = "📘"
                style_class = "chapter"
            elif level == 2:
                level_icon = "📄"
                style_class = "section"
            else:
                level_icon = "📝"
                style_class = "subsection"
            
            toc_data.append({
                "Entry": f"{level_icon} {indent}{title}",
                "Level": level,
                "Type": style_class.title(),
                "Index": i + 1
            })
        
        # Display as dataframe with styling
        df = pd.DataFrame(toc_data)
        
        # Apply conditional formatting
        def highlight_levels(row):
            if row['Level'] == 1:
                return ['background-color: #e3f2fd'] * len(row)
            elif row['Level'] == 2:
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: #f1f8e9'] * len(row)
        
        styled_df = df.style.apply(highlight_levels, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Show more entries option
        if len(filtered_entries) > show_count:
            if st.button(f"Show All {len(filtered_entries)} Entries"):
                # Display all entries in an expander
                with st.expander(f"All {len(filtered_entries)} TOC Entries", expanded=True):
                    for entry in filtered_entries:
                        level = entry.get('level', 1)
                        title = entry.get('title', 'Unknown')
                        indent = "　" * (level - 1)
                        
                        if level == 1:
                            st.markdown(f"### 📘 {title}")
                        elif level == 2:
                            st.markdown(f"#### 📄 {indent}{title}")
                        else:
                            st.markdown(f"##### 📝 {indent}{title}")
    else:
        st.warning("No entries match your search criteria")
    
    # Legend
    with st.expander("📖 TOC Legend"):
        st.markdown("""
        - 📘 **Chapters** (Level 1): Main topics and major sections
        - 📄 **Sections** (Level 2): Subtopics within chapters  
        - 📝 **Subsections** (Level 3+): Detailed topics and concepts
        - 🔵 **Blue Background**: Chapter-level entries
        - 🟠 **Orange Background**: Section-level entries
        - 🟢 **Green Background**: Subsection-level entries
        """)

def create_cross_reference_section(current_book: Dict, all_books: List[Dict], current_idx: int) -> None:
    """Create cross-references to related topics in other books."""
    st.subheader("🔗 Cross-References to Other Books")
    
    other_books = [book for i, book in enumerate(all_books) if i != current_idx]
    
    if not other_books:
        st.info("This is the only book available for comparison.")
        return
    
    # Quick navigation to other books
    st.write("**Quick Navigation:**")
    nav_cols = st.columns(min(len(other_books), 3))
    
    for i, book in enumerate(other_books):
        col_idx = i % len(nav_cols)
        with nav_cols[col_idx]:
            real_idx = all_books.index(book)
            level_badge = "🟢 HS" if book['level'] == 'high_school' else "🔵 UG"
            elective_badge = " 🟡" if 'astronomy' in book.get('book_title', '').lower() else ""
            
            if st.button(f"{level_badge} {book['book_title'][:25]}{'...' if len(book['book_title']) > 25 else ''}{elective_badge}", key=f"cross_ref_{real_idx}"):
                st.session_state.selected_book_idx = real_idx
                st.rerun()
    
    # Topic overlap analysis
    st.write("**Topic Relationship Analysis:**")
    
    current_topics = set(entry.get('title', '').lower() for entry in current_book.get('toc_entries', []))
    
    overlaps = []
    for book in other_books:
        book_topics = set(entry.get('title', '').lower() for entry in book.get('toc_entries', []))
        common_topics = current_topics.intersection(book_topics)
        
        if common_topics:
            overlaps.append({
                'book': book['book_title'],
                'level': book['level'],
                'common_count': len(common_topics),
                'similarity': len(common_topics) / len(current_topics.union(book_topics)),
                'common_topics': list(common_topics)[:5]  # Show first 5
            })
    
    if overlaps:
        # Sort by similarity
        overlaps.sort(key=lambda x: x['similarity'], reverse=True)
        
        for overlap in overlaps:
            with st.expander(f"📊 {overlap['book']} - {overlap['common_count']} common topics ({overlap['similarity']:.1%} similarity)"):
                st.write("**Common Topics:**")
                for topic in overlap['common_topics']:
                    st.write(f"• {topic.title()}")
                
                if len(overlap['common_topics']) < overlap['common_count']:
                    st.write(f"... and {overlap['common_count'] - len(overlap['common_topics'])} more")
                
                # Quick jump button
                other_book_idx = next(i for i, book in enumerate(all_books) if book['book_title'] == overlap['book'])
                if st.button(f"View {overlap['book']}", key=f"jump_to_{other_book_idx}"):
                    st.session_state.selected_book_idx = other_book_idx
                    st.rerun()
    else:
        st.info("No exact topic matches found with other books. This often indicates different approaches to the same subject matter.")
    
    # Educational progression indicator
    if len(all_books) > 1:
        st.write("**Educational Progression:**")
        levels = ['high_school', 'undergraduate', 'graduate']
        current_level = current_book['level']
        
        progression_info = []
        for level in levels:
            books_at_level = [book for book in all_books if book['level'] == level]
            if books_at_level:
                is_current = level == current_level
                progression_info.append(f"{'**' if is_current else ''}{level.replace('_', ' ').title()}: {len(books_at_level)} books{'**' if is_current else ''}")
        
        st.info(" → ".join(progression_info))

def create_educational_progression_view(all_books: List[Dict]) -> None:
    """Create educational progression analysis view."""
    # Group books by educational level
    books_by_level = defaultdict(list)
    for book in all_books:
        books_by_level[book['level']].append(book)
    
    # Analyze topic progression across levels
    level_order = ['high_school', 'undergraduate', 'graduate']
    available_levels = [level for level in level_order if level in books_by_level]
    
    if len(available_levels) >= 2:
        st.write(f"**Educational Progression:** {' → '.join(level.replace('_', ' ').title() for level in available_levels)}")
        
        # Create progression flow chart
        col1, col2, col3 = st.columns(3)
        
        for i, level in enumerate(available_levels):
            books = books_by_level[level]
            total_topics = sum(book['total_topics'] for book in books)
            
            with [col1, col2, col3][i % 3]:
                st.metric(
                    f"{level.replace('_', ' ').title()}",
                    f"{len(books)} books",
                    f"{total_topics} topics"
                )
        
        # Topic complexity analysis
        st.subheader("📊 Topic Complexity by Level")
        
        complexity_data = []
        for level in available_levels:
            books = books_by_level[level]
            avg_hierarchy_depth = 0
            total_entries = 0
            
            for book in books:
                toc_entries = book.get('toc_entries', [])
                if toc_entries:
                    depths = [entry.get('level', 1) for entry in toc_entries]
                    avg_hierarchy_depth += sum(depths)
                    total_entries += len(depths)
            
            if total_entries > 0:
                avg_depth = avg_hierarchy_depth / total_entries
                complexity_data.append({
                    'Level': level.replace('_', ' ').title(),
                    'Avg Hierarchy Depth': round(avg_depth, 2),
                    'Total Topics': total_entries,
                    'Books': len(books)
                })
        
        if complexity_data:
            complexity_df = pd.DataFrame(complexity_data)
            
            # Create bar chart
            fig = px.bar(
                complexity_df,
                x='Level',
                y='Avg Hierarchy Depth',
                title="Average Topic Hierarchy Depth by Educational Level",
                hover_data=['Total Topics', 'Books']
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 educational levels to show progression analysis.")

def create_toc_extraction_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 2: TOC Extraction."""
    st.subheader("📑 Step 2: TOC Extraction Results")
    
    if not data:
        st.warning("No TOC extraction data available")
        return
    
    metrics = data.get('metrics', {})
    tocs_by_level = data.get('tocs_by_level', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Books Processed", metrics.get('total_books_processed', 0))
    with col2:
        st.metric("Successful Extractions", metrics.get('successful_extractions', 0))
    with col3:
        st.metric("Total TOC Entries", metrics.get('total_toc_entries', 0))
    with col4:
        st.metric("Avg Hierarchy Depth", f"{metrics.get('average_hierarchy_depth', 0):.1f}")
    
    # TOC Overlap Analysis
    if 'overlap_analysis' in data and data['overlap_analysis']:
        st.subheader("📊 TOC Overlap Analysis Between Books")
        st.info("This analysis shows how many TOC entries overlap between books vs. unique entries per book")
        
        overlap_data = data['overlap_analysis']
        
        # Create overlap summary table
        overlap_summary = []
        for book_title, analysis in overlap_data.items():
            overlap_summary.append({
                'Book Title': book_title,
                'Total TOC Entries': analysis.get('total_entries', 0),
                'Overlapping with Others': analysis.get('overlapping_count', 0),
                'Unique to This Book': analysis.get('unique_count', 0),
                'Overlap %': f"{analysis.get('overlap_percentage', 0):.1f}%",
                'Uniqueness %': f"{analysis.get('uniqueness_percentage', 0):.1f}%"
            })
        
        if overlap_summary:
            df = pd.DataFrame(overlap_summary)
            st.dataframe(df, use_container_width=True)
            
            # Visualization of overlap patterns
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart showing overlap vs unique entries
                fig_data = []
                for book_title, analysis in overlap_data.items():
                    fig_data.append({
                        'Book': book_title[:20] + '...' if len(book_title) > 20 else book_title,
                        'Type': 'Overlapping',
                        'Count': analysis.get('overlapping_count', 0)
                    })
                    fig_data.append({
                        'Book': book_title[:20] + '...' if len(book_title) > 20 else book_title,
                        'Type': 'Unique',
                        'Count': analysis.get('unique_count', 0)
                    })
                
                if fig_data:
                    fig_df = pd.DataFrame(fig_data)
                    fig = px.bar(
                        fig_df, 
                        x='Book', 
                        y='Count', 
                        color='Type',
                        title="Overlapping vs Unique TOC Entries by Book",
                        barmode='stack'
                    )
                    fig.update_layout(xaxis={'tickangle': 45})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart showing overall distribution
                total_overlapping = sum(analysis.get('overlapping_count', 0) for analysis in overlap_data.values())
                total_unique = sum(analysis.get('unique_count', 0) for analysis in overlap_data.values())
                
                if total_overlapping > 0 or total_unique > 0:
                    pie_data = pd.DataFrame({
                        'Type': ['Overlapping Topics', 'Unique Topics'],
                        'Count': [total_overlapping, total_unique]
                    })
                    
                    fig = px.pie(
                        pie_data, 
                        values='Count', 
                        names='Type',
                        title="Overall TOC Content Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed overlap examples
            st.subheader("🔍 Detailed Overlap Examples")
            
            # Select a book to see its overlap details
            book_options = list(overlap_data.keys())
            if book_options:
                selected_book = st.selectbox(
                    "Select a book to see detailed overlap analysis:",
                    book_options
                )
                
                if selected_book:
                    book_analysis = overlap_data[selected_book]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total TOC Entries", book_analysis.get('total_entries', 0))
                    with col2:
                        st.metric("Overlapping", f"{book_analysis.get('overlapping_count', 0)} ({book_analysis.get('overlap_percentage', 0):.1f}%)")
                    with col3:
                        st.metric("Unique", f"{book_analysis.get('unique_count', 0)} ({book_analysis.get('uniqueness_percentage', 0):.1f}%)")
                    
                    # Show specific overlap examples
                    overlap_details = book_analysis.get('overlap_details', [])
                    if overlap_details:
                        st.subheader(f"📋 Example Topics from '{selected_book}' that appear in other books:")
                        
                        for i, detail in enumerate(overlap_details[:10]):  # Show top 10
                            with st.expander(f"📄 {detail['title'][:60]}{'...' if len(detail['title']) > 60 else ''}"):
                                st.write(f"**Full title:** {detail['title']}")
                                st.write(f"**Also appears in {len(detail['overlapping_books'])} other book(s):**")
                                for other_book in detail['overlapping_books']:
                                    st.write(f"   • {other_book}")
                                st.write(f"**Total occurrences across all books:** {detail['total_occurrences']}")
                    else:
                        st.info("No overlap details available for this book.")
    else:
        st.warning("No overlap analysis data available. This may occur with older extraction results.")
    
    # Processing metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate by level
        if tocs_by_level:
            level_counts = {level: len(tocs) for level, tocs in tocs_by_level.items()}
            fig = px.bar(
                x=list(level_counts.keys()),
                y=list(level_counts.values()),
                title="Successful Extractions by Educational Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality distribution
        quality_dist = metrics.get('quality_distribution', {})
        if quality_dist:
            fig = px.pie(
                values=list(quality_dist.values()),
                names=list(quality_dist.keys()),
                title="TOC Quality Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Processing Time", f"{metrics.get('total_processing_time', 0):.2f}s")
    with perf_col2:
        st.metric("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0):.1%}")
    with perf_col3:
        st.metric("Avg Time/Book", f"{metrics.get('average_time_per_book', 0):.2f}s")
    
    # Detailed TOC analysis by level with side-by-side comparison
    st.subheader("📚 Individual TOCs Side-by-Side Comparison")
    
    # Create side-by-side TOC visualization
    all_books = []
    for level, level_tocs in tocs_by_level.items():
        for toc in level_tocs:
            all_books.append({
                'level': level,
                'book_title': toc.get('book_title', 'Unknown Book'),
                'toc_entries': toc.get('toc_entries', []),
                'extraction_method': toc.get('extraction_method', 'unknown'),
                'total_topics': toc.get('total_topics', len(toc.get('toc_entries', [])))
            })
    
    if all_books:
        # Option to show all books or select specific ones
        show_all = st.checkbox("Show all books with cross-linked navigation", value=True)
        
        if show_all:
            # Create tabbed interface with complete TOCs
            st.subheader("📚 All Books - Complete Table of Contents")
            create_tabbed_book_interface(all_books)
        else:
            # Selective comparison with cross-linked tables
            st.subheader("🔍 Select Books for Detailed Comparison")
            book_options = [f"{book['book_title']} ({book['level']})" for book in all_books]
            selected_books = st.multiselect(
                "Choose books to compare with cross-linked navigation:",
                book_options,
                default=book_options[:min(4, len(book_options))]
            )
            
            if selected_books:
                # Get selected book data
                selected_indices = [book_options.index(book) for book in selected_books]
                comparison_books = [all_books[i] for i in selected_indices]
                create_cross_linked_toc_tables(comparison_books)
    
    # Cross-level topic alignment visualization
    st.subheader("🔗 Cross-Level Topic Alignment")
    
    # Analyze topic alignment across levels
    if len(tocs_by_level) > 1:
        # Find similar topics across levels
        alignment_data = []
        high_school_topics = set()
        undergrad_topics = set()
        
        # Collect topics by level
        for level, level_tocs in tocs_by_level.items():
            for toc in level_tocs:
                toc_entries = toc.get('toc_entries', [])
                for entry in toc_entries:
                    title = entry.get('title', '').lower().strip()
                    if title:
                        if level == 'high_school':
                            high_school_topics.add(title)
                        elif level == 'undergraduate':
                            undergrad_topics.add(title)
        
        # Find common topics (simple string matching)
        common_topics = high_school_topics.intersection(undergrad_topics)
        
        if common_topics:
            st.success(f"Found {len(common_topics)} topics that appear across both high school and undergraduate levels")
            
            # Display common topics in a table
            common_df = pd.DataFrame({
                'Common Topics': sorted(list(common_topics))
            })
            st.dataframe(common_df, use_container_width=True)
        else:
            st.info("No exact topic matches found between levels (this is normal - topics often have different names at different levels)")
    
    # Enhanced Parallel TOC Alignment Analysis
    st.subheader("🔗 Advanced Cross-Book Topic Alignment")
    
    if len(all_books) >= 2:
        # Create alignment matrix for all books
        alignment_data = create_topic_alignment_matrix(all_books)
        
        # Display alignment visualization
        st.subheader("📊 Topic Alignment Heatmap")
        
        # Create topic similarity visualization
        if alignment_data:
            create_topic_similarity_heatmap(alignment_data)
        
        # Concept progression visualization
        st.subheader("📈 Educational Progression Analysis")
        create_educational_progression_view(all_books)
    
    # Expandable detailed view
    st.subheader("📋 Detailed TOC Analysis by Level")
    for level, level_tocs in tocs_by_level.items():
        with st.expander(f"{level.title()} Level - Detailed Analysis ({len(level_tocs)} books)"):
            for toc in level_tocs:
                book_title = toc.get('book_title', 'Unknown Book')
                extraction_method = toc.get('extraction_method', 'unknown')
                total_topics = toc.get('total_topics', len(toc.get('toc_entries', [])))
                
                st.markdown(f"#### 📚 {book_title}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Extraction Method", extraction_method)
                with col2:
                    st.metric("Total Topics", total_topics)
                with col3:
                    st.metric("TOC Entries", len(toc.get('toc_entries', [])))
                
                # Hierarchy analysis
                toc_entries = toc.get('toc_entries', [])
                if toc_entries:
                    levels = [entry.get('level', 1) for entry in toc_entries]
                    max_depth = max(levels) if levels else 0
                    
                    level_counts = Counter(levels)
                    st.write(f"**Hierarchy Depth:** {max_depth} levels")
                    st.write(f"**Level Distribution:** {dict(level_counts)}")
                
                st.markdown("---")

def create_topic_normalization_visualization(data: Dict[str, Any]) -> None:
    """Create rich visualizations for Step 3: Topic Normalization."""
    st.subheader("🔄 Step 3: Topic Normalization with Semantic Alignment")
    
    if not data:
        st.warning("No topic normalization data available")
        return
    
    metrics = data.get('metrics', {})
    normalized_topics = data.get('normalized_topics', [])
    topic_clusters = data.get('topic_clusters', [])
    organized = data.get('organized_by_level', {})
    by_domain = data.get('organized_by_domain', {})
    
    # Summary metrics
    st.subheader("📊 Normalization Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Raw Topics", metrics.get('total_raw_topics', 0))
    with col2:
        st.metric("Normalized Topics", metrics.get('normalized_topics', 0))
    with col3:
        reduction = metrics.get('reduction_ratio', 0)
        st.metric("Reduction Ratio", f"{reduction:.1%}")
    with col4:
        st.metric("Cross-Level Alignments", metrics.get('cross_level_alignments', 0))
    
    # Processing quality metrics
    qual_col1, qual_col2, qual_col3 = st.columns(3)
    with qual_col1:
        st.metric("Semantic Clusters", metrics.get('semantic_clusters', 0))
    with qual_col2:
        st.metric("Avg Consensus Score", f"{metrics.get('avg_consensus_score', 0):.3f}")
    with qual_col3:
        st.metric("Processing Time", f"{metrics.get('processing_time', 0):.2f}s")
    
    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🎯 Semantic Clusters", 
        "📈 Cross-Level Analysis", 
        "🏗️ Topic Hierarchy", 
        "📋 Detailed Topics"
    ])
    
    with viz_tab1:
        st.subheader("Semantic Cluster Visualization")
        
        if topic_clusters:
            # Cluster overview
            cluster_data = []
            for cluster in topic_clusters:
                cluster_data.append({
                    'Cluster ID': cluster['cluster_id'],
                    'Central Topic': cluster['central_topic'],
                    'Members': len(cluster['member_topics']),
                    'Consensus Level': cluster['consensus_level'],
                    'Educational Span': ', '.join(cluster['educational_span'])
                })
            
            cluster_df = pd.DataFrame(cluster_data)
            
            # Cluster size distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    cluster_df, 
                    x='Members', 
                    title="Cluster Size Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    cluster_df,
                    x='Members',
                    y='Consensus Level',
                    size='Members',
                    hover_data=['Central Topic'],
                    title="Cluster Quality vs Size"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Interactive cluster explorer
            st.subheader("🔍 Interactive Cluster Explorer")
            selected_cluster = st.selectbox(
                "Select a cluster to explore:",
                options=range(len(topic_clusters)),
                format_func=lambda x: f"Cluster {topic_clusters[x]['cluster_id']}: {topic_clusters[x]['central_topic']}"
            )
            
            if selected_cluster is not None:
                cluster = topic_clusters[selected_cluster]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Central Topic:** {cluster['central_topic']}")
                    st.write(f"**Educational Span:** {', '.join(cluster['educational_span'])}")
                    st.write(f"**Consensus Level:** {cluster['consensus_level']:.3f}")
                    
                    st.write("**Member Topics:**")
                    for i, (topic, similarity) in enumerate(zip(cluster['member_topics'], cluster['similarity_scores'])):
                        st.write(f"{i+1}. {topic} (similarity: {similarity:.3f})")
                
                with col2:
                    # Similarity distribution for this cluster
                    if cluster['similarity_scores']:
                        fig = px.box(
                            y=cluster['similarity_scores'],
                            title=f"Similarity Distribution<br>Cluster {cluster['cluster_id']}"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab2:
        st.subheader("Cross-Educational Level Analysis")
        
        # Topics by educational level
        if organized:
            level_counts = {level: len(topics) for level, topics in organized.items()}
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=list(level_counts.keys()),
                    y=list(level_counts.values()),
                    title="Topics by Educational Level"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cross-level topic coverage
                cross_level_topics = []
                for topic in normalized_topics:
                    if len(topic['educational_levels']) > 1:
                        cross_level_topics.append({
                            'Topic': topic['canonical_name'],
                            'Levels': len(topic['educational_levels']),
                            'Quality': topic['quality_score']
                        })
                
                if cross_level_topics:
                    cross_df = pd.DataFrame(cross_level_topics)
                    fig = px.scatter(
                        cross_df,
                        x='Levels',
                        y='Quality',
                        size='Quality',
                        hover_data=['Topic'],
                        title="Cross-Level Topic Quality"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Educational progression visualization
        st.subheader("📚 Educational Progression Flow")
        
        # Create a Sankey diagram showing topic flow across levels
        if normalized_topics:
            # Prepare data for Sankey diagram
            level_hierarchy = ['high_school', 'undergraduate', 'graduate', 'professional']
            level_positions = {level: i for i, level in enumerate(level_hierarchy)}
            
            # Count topic flows
            flows = defaultdict(int)
            for topic in normalized_topics:
                levels = sorted(topic['educational_levels'], key=lambda x: level_positions.get(x, 999))
                for i in range(len(levels) - 1):
                    flows[(levels[i], levels[i+1])] += 1
            
            if flows:
                # Create Sankey diagram
                source_indices = []
                target_indices = []
                values = []
                
                all_levels = list(set([level for flow in flows.keys() for level in flow]))
                level_to_index = {level: i for i, level in enumerate(all_levels)}
                
                for (source, target), count in flows.items():
                    source_indices.append(level_to_index[source])
                    target_indices.append(level_to_index[target])
                    values.append(count)
                
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_levels,
                        color="blue"
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values
                    )
                )])
                
                fig.update_layout(title_text="Topic Flow Across Educational Levels", font_size=10)
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab3:
        st.subheader("Topic Hierarchy and Domain Classification")
        
        # Domain distribution
        if by_domain:
            domain_counts = {domain: len(topics) for domain, topics in by_domain.items()}
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(domain_counts.values()),
                    names=list(domain_counts.keys()),
                    title="Topics by Physics Domain"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=list(domain_counts.keys()),
                    y=list(domain_counts.values()),
                    title="Domain Coverage"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Quality distribution
        quality_dist = metrics.get('quality_distribution', {})
        if quality_dist:
            st.subheader("Quality Distribution")
            fig = px.bar(
                x=list(quality_dist.keys()),
                y=list(quality_dist.values()),
                title="Normalized Topic Quality Distribution",
                color=list(quality_dist.keys()),
                color_discrete_map={'high': 'green', 'medium': 'orange', 'low': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tab4:
        st.subheader("Detailed Normalized Topics")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.selectbox(
                "Filter by Educational Level:",
                ["All"] + list(organized.keys()) if organized else ["All"]
            )
        
        with col2:
            domain_filter = st.selectbox(
                "Filter by Domain:",
                ["All"] + list(by_domain.keys()) if by_domain else ["All"]
            )
        
        with col3:
            min_quality = st.slider("Minimum Quality Score:", 0.0, 1.0, 0.0, 0.1)
        
        # Filter topics
        filtered_topics = normalized_topics.copy()
        
        if level_filter != "All":
            filtered_topics = [t for t in filtered_topics if level_filter in t['educational_levels']]
        
        if domain_filter != "All":
            # This would need domain classification in the topic data
            pass
        
        filtered_topics = [t for t in filtered_topics if t['quality_score'] >= min_quality]
        
        # Display filtered topics
        st.write(f"Showing {len(filtered_topics)} topics")
        
        for topic in filtered_topics[:20]:  # Show first 20
            with st.expander(f"📖 {topic['canonical_name']} (Quality: {topic['quality_score']:.3f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Educational Levels:** {', '.join(topic['educational_levels'])}")
                    st.write(f"**Topic Type:** {topic['topic_type']}")
                    st.write(f"**Frequency Score:** {topic['frequency_score']:.3f}")
                    st.write(f"**Consensus Score:** {topic['consensus_score']:.3f}")
                    
                    if topic['alternative_names']:
                        st.write(f"**Alternative Names:** {', '.join(topic['alternative_names'])}")
                
                with col2:
                    st.write(f"**Source Books:** {len(topic['source_books'])}")
                    if topic['source_books']:
                        for book in topic['source_books'][:3]:
                            st.write(f"- {book}")
                        if len(topic['source_books']) > 3:
                            st.write(f"... and {len(topic['source_books']) - 3} more")
                    
                    if topic['learning_objectives']:
                        st.write("**Learning Objectives:**")
                        for obj in topic['learning_objectives']:
                            st.write(f"- {obj}")
        
        if len(filtered_topics) > 20:
            st.info(f"Showing first 20 of {len(filtered_topics)} topics. Adjust filters to see more specific results.")


def create_classification_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 3: Core/Elective Classification."""
    st.subheader("🎯 Step 3: Core/Elective Classification Results")
    
    if not data:
        st.warning("No classification data available")
        return
    
    metadata = data.get('metadata', {})
    books = data.get('books', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", metadata.get('total_books', 0))
    with col2:
        st.metric("Core Books", metadata.get('core_books', 0))
    with col3:
        st.metric("Elective Books", metadata.get('elective_books', 0))
    with col4:
        confidence = metadata.get('average_confidence', 0)
        st.metric("Avg Confidence", f"{confidence:.2f}")
    
    # Classification distribution pie chart
    if books:
        classifications = []
        for book_data in books.values():
            classifications.append(book_data.get('classification', 'unknown'))
        
        class_counts = Counter(classifications)
        if class_counts:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Core vs Elective Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Classification details table
    if books:
        st.subheader("📚 Book Classification Details")
        
        book_data = []
        for book_name, book_info in books.items():
            book_data.append({
                'Book': book_name,
                'Classification': book_info.get('classification', 'unknown'),
                'Confidence': f"{book_info.get('confidence', 0):.2f}",
                'Reasoning': book_info.get('reasoning', 'N/A')[:100] + '...' if len(book_info.get('reasoning', '')) > 100 else book_info.get('reasoning', 'N/A')
            })
        
        df = pd.DataFrame(book_data)
        st.dataframe(df, use_container_width=True)


def create_hierarchy_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 4: Six-Level Hierarchy."""
    st.subheader("🏗️ Step 4: Six-Level Hierarchy Structure")
    
    if not data:
        st.warning("No hierarchy data available")
        return
    
    metadata = data.get('metadata', {})
    hierarchy = data.get('hierarchy', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Topics", metadata.get('total_topics', 0))
    with col2:
        st.metric("Hierarchy Depth", metadata.get('max_depth', 0))
    with col3:
        st.metric("Domains", metadata.get('domains_count', 0))
    with col4:
        st.metric("Categories", metadata.get('categories_count', 0))
    
    # Hierarchy level distribution
    level_counts = metadata.get('level_distribution', {})
    if level_counts:
        fig = px.bar(
            x=list(level_counts.keys()),
            y=list(level_counts.values()),
            title="Topics Distribution by Hierarchy Level"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive hierarchy explorer
    st.subheader("🔍 Interactive Hierarchy Explorer")
    
    if hierarchy:
        # Classification selector
        classification = st.selectbox(
            "Select Classification:",
            options=list(hierarchy.keys())
        )
        
        if classification and classification in hierarchy:
            class_data = hierarchy[classification]
            
            # Domain selector
            if isinstance(class_data, dict):
                domain = st.selectbox(
                    "Select Domain:",
                    options=list(class_data.keys())
                )
                
                if domain and domain in class_data:
                    st.json(class_data[domain])


def create_prerequisites_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 5: Prerequisites & Dependencies."""
    st.subheader("🔗 Step 5: Prerequisites & Dependencies Analysis")
    
    if not data:
        st.warning("No prerequisites data available")
        return
    
    metadata = data.get('metadata', {})
    graph_data = data.get('graph_data', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Topics", metadata.get('total_topics', 0))
    with col2:
        st.metric("Relationships", metadata.get('total_relationships', 0))
    with col3:
        st.metric("Cycles Removed", metadata.get('removed_cycles', 0))
    with col4:
        graph_props = metadata.get('graph_properties', {})
        st.metric("Is DAG", "✅" if graph_props.get('is_dag') else "❌")
    
    # Graph properties
    if graph_props:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Graph Properties")
            st.write(f"**Nodes:** {graph_props.get('nodes', 0)}")
            st.write(f"**Edges:** {graph_props.get('edges', 0)}")
            st.write(f"**Density:** {graph_props.get('density', 0):.4f}")
            st.write(f"**Avg Confidence:** {graph_props.get('avg_confidence', 0):.2f}")
        
        with col2:
            # Relationship types distribution
            rel_types = graph_props.get('relationship_types', {})
            if rel_types:
                fig = px.pie(
                    values=list(rel_types.values()),
                    names=list(rel_types.keys()),
                    title="Relationship Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)


def create_standards_visualization(data: Dict[str, Any]) -> None:
    """Create visualizations for Step 6: Standards Mapping."""
    st.subheader("📊 Step 6: Educational Standards Mapping")
    
    if not data:
        st.warning("No standards mapping data available")
        return
    
    metadata = data.get('metadata', {})
    standards_mapping = metadata.get('standards_mapping', {})
    stats = standards_mapping.get('statistics', {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Mappings", stats.get('total_mappings', 0))
    with col2:
        st.metric("Standards Covered", stats.get('standards_covered', 0))
    with col3:
        st.metric("Processed Topics", standards_mapping.get('processed_topics', 0))
    with col4:
        st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
    
    # Standards distribution
    mappings_by_standard = stats.get('mappings_by_standard', {})
    if mappings_by_standard:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=list(mappings_by_standard.keys()),
                y=list(mappings_by_standard.values()),
                title="Mappings by Educational Standard"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                values=list(mappings_by_standard.values()),
                names=list(mappings_by_standard.keys()),
                title="Standards Coverage Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Bloom's taxonomy distribution
    bloom_dist = stats.get('bloom_distribution', {})
    if bloom_dist:
        fig = px.bar(
            x=list(bloom_dist.keys()),
            y=list(bloom_dist.values()),
            title="Bloom's Taxonomy Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Difficulty distribution
    difficulty_dist = stats.get('difficulty_distribution', {})
    if difficulty_dist:
        fig = px.pie(
            values=list(difficulty_dist.values()),
            names=list(difficulty_dist.keys()),
            title="Difficulty Level Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def create_export_visualization(data: Dict[str, Any]) -> None:
    """Create comprehensive visualization for Step 7: Multi-Format Export."""
    st.subheader("📤 Multi-Format Curriculum Export")
    
    # Check for export files with specific patterns
    export_patterns = {
        'Complete Curriculum JSON': '*complete_curriculum.json',
        'Complete Curriculum DOT': '*complete_curriculum.dot', 
        'Complete Curriculum DuckDB': '*complete_curriculum.db',
        'TSV Exports': '*.tsv',
        'Other JSON Files': '*curriculum*.json',
        'Visualization Files': '*.png'
    }
    
    all_files = []
    format_summary = {}
    
    for category, pattern in export_patterns.items():
        matching_files = list(CURRICULUM_DIR.glob(pattern))
        format_summary[category] = len(matching_files)
        for file_path in matching_files:
            file_size = file_path.stat().st_size
            all_files.append({
                'Category': category,
                'File': file_path.name,
                'Format': file_path.suffix.upper(),
                'Size': f"{file_size / (1024*1024):.2f} MB" if file_size > 1024*1024 else f"{file_size / 1024:.1f} KB",
                'Size_MB': file_size / (1024*1024),
                'Modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                'Path': str(file_path)
            })
    
    if all_files:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Files", len(all_files))
        with col2:
            total_size = sum(f['Size_MB'] for f in all_files)
            st.metric("💾 Total Size", f"{total_size:.1f} MB")
        with col3:
            formats = len(set(f['Format'] for f in all_files))
            st.metric("📋 Formats", formats)
        with col4:
            main_files = len([f for f in all_files if 'complete_curriculum' in f['File']])
            st.metric("🎯 Main Outputs", main_files)
        
        # Required formats status
        st.subheader("📋 Export Format Status")
        
        required_formats = {
            'JSON': any('complete_curriculum.json' in f['File'] for f in all_files),
            'DOT': any('complete_curriculum.dot' in f['File'] for f in all_files),
            'DuckDB': any('complete_curriculum.db' in f['File'] for f in all_files),
            'TSV': any('complete_curriculum.tsv' in f['File'] for f in all_files)
        }
        
        status_cols = st.columns(4)
        for i, (fmt, exists) in enumerate(required_formats.items()):
            with status_cols[i]:
                if exists:
                    st.success(f"✅ {fmt}")
                else:
                    st.error(f"❌ {fmt}")
        
        # File listing
        st.subheader("📁 Generated Files")
        
        # Create DataFrame and sort by category and size
        df = pd.DataFrame(all_files)
        df = df.sort_values(['Category', 'Size_MB'], ascending=[True, False])
        
        # Display with categories
        for category in df['Category'].unique():
            with st.expander(f"📂 {category} ({len(df[df['Category'] == category])} files)"):
                category_df = df[df['Category'] == category][['File', 'Format', 'Size', 'Modified']]
                st.dataframe(category_df, use_container_width=True, hide_index=True)
        
        # Format distribution chart
        format_counts = df['Format'].value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=format_counts.values,
                names=format_counts.index,
                title="Export Formats Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Size by format
            size_by_format = df.groupby('Format')['Size_MB'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=list(size_by_format.index),
                y=list(size_by_format.values),
                title="File Size by Format (MB)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download section
        st.subheader("📥 Download Options")
        st.info("💡 **Main curriculum files:** Look for files with 'complete_curriculum' in the name for the final outputs")
        
    else:
        st.warning("No export files found. Run Step 7 to generate exports.")
        st.info("**Expected outputs:**")
        st.write("- 📄 **JSON**: Complete curriculum data")
        st.write("- 📊 **TSV**: Spreadsheet-compatible format") 
        st.write("- 🔗 **DOT**: Graph visualization format")
        st.write("- 🗄️ **DuckDB**: Database format for analysis")


def create_config_editor() -> None:
    """Create an interactive configuration editor."""
    st.subheader("⚙️ Configuration Editor")
    
    config_file = Path("config/curriculum_config.json")
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Create tabs for different configuration sections
            config_tabs = st.tabs(["General", "LLM Settings", "Processing", "Export", "Standards"])
            
            with config_tabs[0]:  # General
                st.subheader("General Settings")
                config_data['subject'] = st.text_input("Subject", value=config_data.get('subject', 'Physics'))
                config_data['language'] = st.selectbox("Language", 
                    options=["English", "Spanish", "French", "German"],
                    index=["English", "Spanish", "French", "German"].index(config_data.get('language', 'English'))
                )
                config_data['cache_directory'] = st.text_input("Cache Directory", value=config_data.get('cache_directory', 'cache'))
            
            with config_tabs[1]:  # LLM Settings
                st.subheader("LLM Configuration")
                config_data['llm_provider'] = st.selectbox("LLM Provider",
                    options=["openai", "anthropic"],
                    index=["openai", "anthropic"].index(config_data.get('llm_provider', 'openai'))
                )
                config_data['llm_model'] = st.text_input("LLM Model", value=config_data.get('llm_model', 'gpt-4'))
                config_data['max_tokens'] = st.number_input("Max Tokens", value=config_data.get('max_tokens', 4000), min_value=100, max_value=8000)
                config_data['temperature'] = st.slider("Temperature", 0.0, 1.0, config_data.get('temperature', 0.1), 0.1)
            
            with config_tabs[2]:  # Processing
                st.subheader("Processing Settings")
                config_data['confidence_threshold'] = st.slider("Confidence Threshold", 0.0, 1.0, config_data.get('confidence_threshold', 0.7), 0.05)
                config_data['cycle_detection_enabled'] = st.checkbox("Enable Cycle Detection", value=config_data.get('cycle_detection_enabled', True))
                config_data['elective_threshold'] = st.slider("Elective Classification Threshold", 0.0, 1.0, config_data.get('elective_threshold', 0.2), 0.05)
            
            with config_tabs[3]:  # Export
                st.subheader("Export Settings")
                available_formats = ["tsv", "json", "dot", "duckdb"]
                current_formats = config_data.get('export_formats', available_formats)
                config_data['export_formats'] = st.multiselect("Export Formats", available_formats, default=current_formats)
            
            with config_tabs[4]:  # Standards
                st.subheader("Standards Mapping")
                available_standards = ["MCAT", "IB_HL", "IB_SL", "A_Level", "IGCSE", "ABET", "ISO", "UNESCO"]
                current_standards = config_data.get('supported_standards', ["MCAT", "IB_HL", "IB_SL", "A_Level", "IGCSE"])
                config_data['supported_standards'] = st.multiselect("Supported Standards", available_standards, default=current_standards)
            
            # Save configuration
            if st.button("💾 Save Configuration", type="primary"):
                try:
                    with open(config_file, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    st.success("✅ Configuration saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")
            
            # Show current configuration
            with st.expander("📄 View Raw Configuration"):
                st.json(config_data)
                
        except Exception as e:
            st.error(f"Failed to load configuration file: {e}")
            st.info("Create a default configuration file:")
            
            if st.button("Create Default Config"):
                default_config = {
                    "subject": "Physics",
                    "language": "English",
                    "cache_directory": "cache",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4",
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "confidence_threshold": 0.7,
                    "cycle_detection_enabled": True,
                    "elective_threshold": 0.2,
                    "export_formats": ["tsv", "json", "dot", "duckdb"],
                    "supported_standards": ["MCAT", "IB_HL", "IB_SL", "A_Level", "IGCSE"]
                }
                
                try:
                    config_file.parent.mkdir(exist_ok=True)
                    with open(config_file, 'w') as f:
                        json.dump(default_config, f, indent=2)
                    st.success("✅ Default configuration created!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create configuration: {e}")
    else:
        st.warning("Configuration file not found. Create one to customize pipeline settings.")


def main():
    """Main application with enhanced 7-step visualization."""
    # Configure page settings
    st.set_page_config(
        page_title="Enhanced Curriculum AI Admin Panel",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎓 Enhanced Curriculum AI Admin Panel")
    st.markdown("*7-Step Modular Curriculum Generation System*")
    st.markdown("---")
    
    admin = EnhancedCurriculumAdmin()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Keys
        st.subheader("API Keys")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        if openai_key:
            st.success("✅ OpenAI API key found")
        else:
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            openai_key = openai_api_key
        
        if anthropic_key:
            st.success("✅ Anthropic API key found")
        
        # Discipline selection
        disciplines = admin.get_available_disciplines()
        if disciplines:
            selected_discipline = st.selectbox("Select Discipline", disciplines)
        else:
            selected_discipline = st.text_input("Enter Discipline", value="Physics")
        
        # Language selection
        selected_language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        
        # Cache management
        st.markdown("---")
        st.subheader("💾 Cache Management")
        
        # Cache statistics
        cache_stats = admin.cache_manager.get_cache_stats()
        total_cached = sum(stats['valid_entries'] for stats in cache_stats.values())
        total_expired = sum(stats['expired_entries'] for stats in cache_stats.values())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Valid Cache Entries", total_cached)
        with col2:
            st.metric("Expired Entries", total_expired)
        
        # Cache controls
        if st.button("🧹 Clean Expired Cache", help="Remove expired cache entries"):
            admin.cache_manager.cleanup_expired_cache()
            st.success("Cache cleaned!")
            st.rerun()
        
        if st.button("🗑️ Clear All Cache", help="Clear all cache for selected discipline"):
            if selected_discipline:
                admin.cache_manager.invalidate_cache(selected_discipline, selected_language)
                st.success(f"Cache cleared for {selected_discipline}/{selected_language}")
                st.rerun()
        
        # Show detailed cache stats
        with st.expander("📊 Detailed Cache Statistics"):
            for cache_type, stats in cache_stats.items():
                st.write(f"**{cache_type.replace('_', ' ').title()}:**")
                st.write(f"- Valid: {stats['valid_entries']}")
                st.write(f"- Expired: {stats['expired_entries']}")
                st.write(f"- Total files: {stats['total_files']}")
        
        # Pipeline controls
        st.markdown("---")
        st.header("🚀 7-Step Pipeline")
        
        # Pipeline options
        col1, col2, col3 = st.columns(3)
        with col1:
            force_refresh = st.checkbox("🔄 Force Refresh", value=False, 
                                      help="Re-run steps even if output already exists")
        with col2:
            disable_llm = st.checkbox("⚡ Disable LLM", value=False,
                                    help="⚠️ Disable LLM for faster execution (reduces quality significantly). Default: LLM ENABLED for best results.")
        with col3:
            openai_key = st.text_input("🔑 OpenAI API Key", type="password", 
                                     value=os.environ.get("OPENAI_API_KEY", ""),
                                     help="Required for LLM-enhanced processing")
        
        # Show LLM status
        llm_status = "🔴 DISABLED" if disable_llm else "🟢 ENABLED" 
        api_status = "🟢 SET" if (openai_key or os.environ.get("OPENAI_API_KEY")) else "🔴 MISSING"
        st.info(f"**LLM Processing:** {llm_status} | **OpenAI API Key:** {api_status}")
        
        # Step execution buttons
        steps = [
            ("step1_discovery", "📚 Book Discovery", "Discover books across all educational levels"),
            ("step2_toc", "📑 TOC Extraction", "Extract table of contents from discovered books"),
            ("step3_classification", "🎯 Core/Elective Classification", "Classify books as core curriculum vs elective domains"),
            ("step4_hierarchy", "🏗️ Six-Level Hierarchy", "Build normalized six-level hierarchical curriculum structure"),
            ("step5_prerequisites", "🔗 Prerequisites & Dependencies", "Map knowledge dependencies and prerequisite relationships"),
            ("step6_standards", "📊 Standards Mapping", "Map content to educational standards (MCAT, IB, A-Level, etc.)"),
            ("step7_export", "📤 Multi-Format Export", "Export curriculum in TSV, JSON, DOT, and DuckDB formats"),
            ("master_orchestrator", "🚀 Master Pipeline", "Run the complete curriculum generation pipeline (Steps 3-7)")
        ]
        
        # Individual step buttons
        for step_id, step_name, step_desc in steps:
            if st.button(step_name, help=step_desc, use_container_width=True):
                result = admin.run_modular_step(
                    step_id, 
                    selected_discipline, 
                    selected_language,
                    force_refresh=force_refresh,
                    disable_llm=disable_llm,
                    openai_api_key=openai_key
                )
                
                if result["success"]:
                    st.success(f"✅ {step_name} completed!")
                    if result.get("stdout"):
                        with st.expander("Output"):
                            st.code(result["stdout"])
                    st.rerun()
                else:
                    st.error(f"❌ {step_name} failed!")
                    if result.get("error"):
                        st.error(result["error"])
                    if result.get("stderr"):
                        st.error(result["stderr"])
        
        # Smart pipeline execution with caching and resume capability
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run All Steps", type="primary", use_container_width=True):
                st.session_state["run_pipeline"] = True
                st.session_state["force_refresh_pipeline"] = True  # Force re-run all steps
                st.session_state["resume_mode"] = False  # Ensure we start from step 1
                
        with col2:
            if st.button("🔄 Resume Pipeline", use_container_width=True):
                st.session_state["run_pipeline"] = True
                st.session_state["resume_mode"] = True
        
        # Pipeline execution logic
        if st.session_state.get("run_pipeline", False):
            st.session_state["run_pipeline"] = False
            resume_mode = st.session_state.get("resume_mode", False)
            force_refresh_pipeline = st.session_state.get("force_refresh_pipeline", False)
            if force_refresh_pipeline:
                st.session_state["force_refresh_pipeline"] = False  # Reset flag
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Determine starting step
            start_index = 0
            if resume_mode:
                st.info("🔄 **Resume Mode**: Checking for completed steps...")
                # Find the first incomplete step
                for i, (step_id, step_name, _) in enumerate(steps):
                    result_data = admin.load_step_results(step_id, selected_discipline, selected_language)
                    if result_data is None:
                        start_index = i
                        st.info(f"📍 Found first incomplete step: {i+1}. {step_name}")
                        break
                    else:
                        st.success(f"✅ Step {i+1} already completed: {step_name}")
                else:
                    # All steps completed
                    st.success("🎉 All steps already completed!")
                    st.session_state["resume_mode"] = False
                    st.rerun()
                    start_index = len(steps)  # Skip execution
                
                if start_index > 0:
                    st.info(f"📋 Resuming from Step {start_index + 1}: {steps[start_index][1]}")
                    st.info(f"✅ Skipping {start_index} completed step(s)")
            else:
                if force_refresh_pipeline:
                    st.info("🚀 **Run All Steps Mode**: Starting complete pipeline from Step 1 (with force refresh)")
                else:
                    st.info("🚀 **Run All Steps Mode**: Starting complete pipeline from Step 1")
            
            # Execute steps starting from start_index
            for i in range(start_index, len(steps)):
                step_id, step_name, _ = steps[i]
                
                # Check if step is already completed (unless force_refresh is enabled)
                if not force_refresh and not force_refresh_pipeline:
                    result_data = admin.load_step_results(step_id, selected_discipline, selected_language)
                    if result_data is not None:
                        status_text.text(f"✅ {step_name} (already completed)")
                        progress_bar.progress((i + 1) / len(steps))
                        st.success(f"✅ {step_name} - using cached result")
                        continue
                
                status_text.text(f"🏃 Running {step_name}...")
                progress_bar.progress(i / len(steps))
                
                # Show execution details
                llm_mode = "LLM DISABLED" if disable_llm else "LLM ENABLED"
                st.info(f"🔧 Executing Step {i+1}/{len(steps)}: {step_name} ({llm_mode})")
                
                # Show real-time status
                with st.spinner(f"Executing {step_name}..."):
                    result = admin.run_modular_step(
                        step_id,
                        selected_discipline,
                        selected_language,
                        force_refresh=force_refresh or force_refresh_pipeline,
                        disable_llm=disable_llm,
                        openai_api_key=openai_key
                    )
                
                if not result["success"]:
                    st.error(f"❌ Pipeline failed at {step_name}")
                    
                    # Show detailed error information
                    if result.get("error"):
                        st.error(f"Error: {result['error']}")
                    
                    # Show command that was run
                    if result.get("command"):
                        st.code(f"Command: {result['command']}", language="bash")
                    
                    # Show stderr output
                    if result.get("stderr"):
                        st.error("Standard Error Output:")
                        st.code(result["stderr"], language="text")
                    
                    # Show stdout output
                    if result.get("stdout"):
                        st.info("Standard Output:")
                        st.code(result["stdout"], language="text")
                    
                    # Add resume instructions
                    st.warning("💡 **Tip**: Use the '🔄 Resume Pipeline' button to continue from where it left off after fixing any issues.")
                    
                    break
                else:
                    progress_bar.progress((i + 1) / len(steps))
                    st.success(f"✅ {step_name} completed successfully!")
                    
                    # Show intermediate results after each step completion
                    st.info(f"📄 **Step {i+1} Results Available**")
                    
                    # Load and display step results immediately
                    step_result_data = admin.load_step_results(step_id, selected_discipline, selected_language)
                    if step_result_data:
                        # Create a temporary expander to show some results
                        with st.expander(f"🔍 Preview {step_name} Results", expanded=False):
                            if isinstance(step_result_data, dict):
                                # Show basic stats about the results
                                if step_id == "step1_discovery":
                                    books_by_level = step_result_data.get("books_by_level", {})
                                    total_books = sum(len(books) for books in books_by_level.values())
                                    st.metric("📚 Books Discovered", total_books)
                                    for level, books in books_by_level.items():
                                        st.write(f"**{level.replace('_', ' ').title()}**: {len(books)} books")
                                
                                elif step_id == "step2_toc":
                                    tocs_by_level = step_result_data.get("tocs_by_level", {})
                                    total_tocs = sum(len(tocs) for tocs in tocs_by_level.values())
                                    st.metric("📑 TOCs Extracted", total_tocs)
                                
                                elif step_id in ["step3_classification", "step4_hierarchy"]:
                                    topics = step_result_data.get("topics", {})
                                    st.metric("📋 Topics Processed", len(topics))
                                
                                elif step_id == "step5_prerequisites":
                                    metadata = step_result_data.get("metadata", {})
                                    st.metric("🔗 Prerequisites Mapped", metadata.get("total_relationships", 0))
                                
                                elif step_id == "step6_standards":
                                    metadata = step_result_data.get("metadata", {})
                                    standards_mapping = metadata.get("standards_mapping", {})
                                    stats = standards_mapping.get("statistics", {})
                                    st.metric("📊 Standards Mappings", stats.get("total_mappings", 0))
                                
                                elif step_id == "step7_export":
                                    st.metric("📤 Export Complete", "✅")
                                    st.write("**Available formats:** TSV, JSON, DOT, DuckDB")
                    
                    # Refresh the Pipeline Status Overview to show this step as completed
                    st.info("🔄 **Updating interface with new results...**")
                    
                    # Force a rerun to update the Pipeline Status Overview colors
                    if i < len(steps) - 1:  # Not the last step
                        st.warning(f"📋 **Next**: Starting Step {i+2}: {steps[i+1][1]}")
                        time.sleep(1)  # Brief pause to show progress
                    else:
                        st.success("🎉 **All steps completed!** The full curriculum is now available in the tabs below.")
                        time.sleep(2)
                    
            else:
                # All steps completed successfully
                progress_bar.progress(1.0)
                status_text.text("✅ All steps completed successfully!")
                st.success("🎉 Complete pipeline finished!")
                st.balloons()
                
                # Show final completion message
                st.success("📊 **All curriculum data has been processed and saved!**")
                st.info("🔽 **Scroll down to view the complete results in the tabs below**")
            
            # Reset resume mode and refresh interface
            st.session_state["resume_mode"] = False
            st.session_state["pipeline_just_completed"] = True
            st.rerun()
    
    # Main content area
    if selected_discipline:
        # Show completion banner if pipeline just finished
        if st.session_state.get("pipeline_just_completed", False):
            st.success("🎉 **Pipeline Completed Successfully!** All results are now available below.")
            st.session_state["pipeline_just_completed"] = False  # Clear the flag
        
        # Pipeline status overview
        st.subheader("📊 Pipeline Status Overview")
        
        # Check which steps have been completed
        step_status = {}
        for step_id, step_name, _ in steps:
            result_data = admin.load_step_results(step_id, selected_discipline, selected_language)
            step_status[step_id] = {
                'name': step_name,
                'completed': result_data is not None,
                'data': result_data
            }
        
        # Visual progress bar
        progress_percentage = (sum(1 for status in step_status.values() if status['completed']) / len(steps)) * 100
        st.progress(progress_percentage / 100)
        st.write(f"**Pipeline Progress: {progress_percentage:.0f}% Complete**")
        
        # Status grid with detailed information and enhanced colors
        status_cols = st.columns(len(steps))
        for i, (step_id, step_name, _) in enumerate(steps):
            with status_cols[i]:
                status = step_status[step_id]
                if status['completed']:
                    # Use green container for completed steps
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            background-color: #d4edda; 
                            border: 2px solid #28a745; 
                            border-radius: 10px; 
                            padding: 10px; 
                            text-align: center;
                            margin: 5px 0px;
                        ">
                            <h4 style="color: #155724; margin: 0;">✅ Step {i+1}</h4>
                            <p style="color: #155724; margin: 5px 0; font-size: 12px;">{status['name']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show file size if available using standardized directories
                    file_path = admin._get_step_output_file(step_id, selected_discipline, selected_language)
                    if file_path and file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        st.caption(f"📁 {size_mb:.1f} MB")
                else:
                    # Use gray/yellow container for pending steps
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            background-color: #fff3cd; 
                            border: 2px solid #ffc107; 
                            border-radius: 10px; 
                            padding: 10px; 
                            text-align: center;
                            margin: 5px 0px;
                        ">
                            <h4 style="color: #856404; margin: 0;">⏳ Step {i+1}</h4>
                            <p style="color: #856404; margin: 5px 0; font-size: 12px;">{status['name']}</p>
                            <p style="color: #856404; margin: 5px 0; font-size: 10px;">📝 Pending</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Pipeline progress summary
        completed_steps = sum(1 for status in step_status.values() if status['completed'])
        total_steps = len(steps)
        progress_percentage = (completed_steps / total_steps) * 100
        
        st.info(f"📈 **Pipeline Progress**: {completed_steps}/{total_steps} steps completed ({progress_percentage:.0f}%)")
        
        if completed_steps > 0 and completed_steps < total_steps:
            st.info(f"💡 **Next Step**: {steps[completed_steps][1]}")
            st.info("🔄 Use the 'Resume Pipeline' button to continue from where you left off.")
        
        # Main Content Tabs
        st.markdown("---")
        st.header("📊 Curriculum Analysis Dashboard")
        
        # Create main tabs for organized content
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📚 Books & TOCs", 
            "📋 Curriculum Topics", 
            "🎯 Academic Levels", 
            "🩺 MCAT Mapping",
            "🔥 Topic Heat Map",
            "📤 Final Curriculum",
            "⚙️ Advanced"
        ])
        
        with tab1:
            st.subheader("📚 Discovered Books")
            
            # Explain book list relationships
            with st.expander("ℹ️ About Book Lists and Discovery", expanded=False):
                st.markdown("""
                **Understanding the Different Book Lists:**
                
                📋 **BookList.json/tsv** (`/Books/BookList.json`)
                - **What it is**: Complete inventory of ALL available books in the repository
                - **Content**: 3000+ books across all languages, disciplines, and levels
                - **Purpose**: Master catalog for book discovery and management
                - **Format**: Structured metadata with file paths, languages, disciplines
                
                🔍 **Physics_English_books_discovered.json** (`/Books/Physics_English_books_discovered.json`) 
                - **What it is**: Filtered results from Step 1 (Book Discovery) for Physics in English
                - **Content**: Only Physics books in English (~6-12 books)
                - **Purpose**: Specific input for the curriculum generation pipeline
                - **Format**: Enhanced metadata with quality scores, educational levels, core/elective classification
                
                **Relationship:**
                1. 📖 Step 1 reads from the master `BookList.json`
                2. 🔍 Filters for discipline="Physics" and language="English" 
                3. 📊 Adds quality scoring, educational level analysis, and core/elective classification
                4. 💾 Saves filtered results as `Physics_English_books_discovered.json`
                5. 📚 This becomes input for Step 2 (TOC Extraction)
                
                **Why separate files?**
                - ⚡ **Performance**: No need to process all 3000+ books for each curriculum
                - 🎯 **Focus**: Pipeline works with relevant books only
                - 📈 **Enhancement**: Adds curriculum-specific metadata and scoring
                """)
            
            if step_status['step1_discovery']['completed'] and step_status['step1_discovery']['data']:
                create_enhanced_books_list(step_status['step1_discovery']['data'])
            else:
                st.info("Complete Step 1 (Book Discovery) to see discovered books")
                
            st.markdown("---")
            st.subheader("📖 TOC Overlap Analysis")
            if step_status['step2_toc']['completed'] and step_status['step2_toc']['data']:
                create_toc_overlap_analysis(step_status['step2_toc']['data'])
            else:
                st.info("Complete Step 2 (TOC Extraction) to see overlap analysis")
                
            st.markdown("---")
            st.subheader("📑 Individual Book TOCs")
            if step_status['step2_toc']['completed'] and step_status['step2_toc']['data']:
                create_book_toc_tabs(step_status['step2_toc']['data'])
            else:
                st.info("Complete Step 2 (TOC Extraction) to see individual book TOCs")
        
        with tab2:
            if step_status['step4_hierarchy']['completed'] and step_status['step4_hierarchy']['data']:
                create_beautiful_six_level_display(step_status['step4_hierarchy']['data'])
            else:
                st.info("Complete Step 4 (Six-Level Hierarchy) to see the complete curriculum hierarchy")
                st.write("**The six-level hierarchy includes:**")
                st.write("- 🌍 **Domains**: Main subject areas (e.g., Core Physics, Electives)")
                st.write("- 📚 **Categories**: Major topic groups within domains")
                st.write("- 💡 **Concepts**: Fundamental ideas within categories")
                st.write("- 📝 **Topics**: Specific learning topics")
                st.write("- 🔍 **Subtopics**: Detailed breakdowns of topics")
                st.write("- ⚡ **Learning Elements**: Individual learning objectives")
        
        with tab3:
            st.subheader("🎓 Academic Level Analysis")
            if step_status['step4_hierarchy']['completed'] and step_status['step4_hierarchy']['data']:
                create_academic_levels_analysis(step_status['step4_hierarchy']['data'])
            else:
                st.info("Complete Step 4 (Six-Level Hierarchy) to see academic level analysis")
        
        with tab4:
            st.subheader("🩺 MCAT Standards Mapping")
            if step_status['step6_standards']['completed'] and step_status['step6_standards']['data']:
                create_mcat_mapping_analysis(step_status['step6_standards']['data'])
            else:
                st.info("Complete Step 6 (Standards Mapping) to see MCAT analysis")
        
        with tab5:
            st.subheader("🔥 Topic vs Educational Level Heat Map")
            if step_status['step4_hierarchy']['completed'] and step_status['step4_hierarchy']['data']:
                create_topic_level_heatmap(step_status['step4_hierarchy']['data'])
            else:
                st.info("Complete Step 4 (Six-Level Hierarchy) to see topic heat map")
        
        with tab6:
            st.subheader("📤 Final Complete Curriculum")
            if step_status['step7_export']['completed'] and step_status['step7_export']['data']:
                create_export_visualization(step_status['step7_export']['data'])
                
                st.markdown("---")
                st.subheader("🌍 Complete Universal Topic Browser")
                create_universal_topic_browser(step_status, selected_discipline, selected_language)
            else:
                st.info("Complete Step 7 (Multi-Format Export) to see the final complete curriculum")
                st.write("**The final curriculum includes:**")
                st.write("- 📊 All exported file formats (TSV, JSON, DOT, DuckDB)")
                st.write("- 🌐 Complete topic browser with all curriculum subtopics")
                st.write("- 📈 Export statistics and file information")
                st.write("- 🔗 Universal search and filtering across all topics")
        
        with tab7:
            st.subheader("⚙️ Configuration Editor")
            create_config_editor()
            
            st.markdown("---")
            st.subheader("📈 Performance Analytics")
            if any(status['completed'] for status in step_status.values()):
                create_performance_analytics(step_status)
    
    else:
        st.info("👈 Select a discipline from the sidebar to begin")

def create_prerequisite_mapping_visualization(data: Dict[str, Any]) -> None:
    """Create comprehensive visualization for prerequisite mapping results."""
    st.subheader("🔗 Prerequisite Mapping Analysis")
    
    if not data:
        st.warning("No prerequisite mapping data available.")
        return
    
    # Key metrics
    prerequisite_relations = data.get('prerequisite_relations', [])
    if not prerequisite_relations:
        st.warning("No prerequisite relations found in the data.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_topics = len(prerequisite_relations)
    topics_with_prereqs = len([r for r in prerequisite_relations if r.get('prerequisite_ids')])
    avg_prereqs = sum(len(r.get('prerequisite_ids', [])) for r in prerequisite_relations) / total_topics if total_topics > 0 else 0
    max_prereqs = max(len(r.get('prerequisite_ids', [])) for r in prerequisite_relations) if prerequisite_relations else 0
    
    with col1:
        st.metric("Total Topics", total_topics)
    with col2:
        st.metric("Topics with Prerequisites", topics_with_prereqs)
    with col3:
        st.metric("Avg Prerequisites/Topic", f"{avg_prereqs:.1f}")
    with col4:
        st.metric("Max Prerequisites", max_prereqs)
    
    # Prerequisites distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Prerequisite count distribution
        prereq_counts = [len(r.get('prerequisite_ids', [])) for r in prerequisite_relations]
        count_dist = Counter(prereq_counts)
        
        fig = px.bar(
            x=list(count_dist.keys()),
            y=list(count_dist.values()),
            title="Distribution of Prerequisite Counts",
            labels={'x': 'Number of Prerequisites', 'y': 'Number of Topics'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Domain distribution
        domains = [r.get('domain', 'unknown') for r in prerequisite_relations]
        domain_dist = Counter(domains)
        
        fig = px.pie(
            values=list(domain_dist.values()),
            names=list(domain_dist.keys()),
            title="Topics by Domain"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational level analysis
    st.subheader("📚 Educational Level Analysis")
    
    # Group by educational level
    level_data = defaultdict(list)
    for relation in prerequisite_relations:
        level = relation.get('educational_level', 'unknown')
        level_data[level].append(relation)
    
    level_cols = st.columns(len(level_data))
    for i, (level, relations) in enumerate(level_data.items()):
        with level_cols[i]:
            st.metric(f"{level.title()} Topics", len(relations))
            avg_prereqs_level = sum(len(r.get('prerequisite_ids', [])) for r in relations) / len(relations)
            st.metric(f"Avg Prerequisites", f"{avg_prereqs_level:.1f}")
    
    # Prerequisite network complexity
    st.subheader("🌐 Prerequisite Network Analysis")
    
    # Create network graph data
    nodes = []
    edges = []
    topic_dict = {r['topic_id']: r for r in prerequisite_relations}
    
    for relation in prerequisite_relations:
        topic_id = relation['topic_id']
        topic_title = relation.get('topic_title', 'Unknown')
        
        # Node size based on number of dependents
        dependents = len([r for r in prerequisite_relations if topic_id in r.get('prerequisite_ids', [])])
        
        nodes.append({
            'id': topic_id,
            'title': topic_title[:30] + '...' if len(topic_title) > 30 else topic_title,
            'domain': relation.get('domain', 'unknown'),
            'level': relation.get('educational_level', 'unknown'),
            'size': max(5, dependents * 2),
            'prereq_count': len(relation.get('prerequisite_ids', []))
        })
        
        # Add edges for prerequisites
        for prereq_id in relation.get('prerequisite_ids', []):
            if prereq_id in topic_dict:
                edges.append({
                    'source': prereq_id,
                    'target': topic_id
                })
    
    # Network statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Network Nodes", len(nodes))
    with col2:
        st.metric("Network Edges", len(edges))
    with col3:
        complexity = len(edges) / len(nodes) if len(nodes) > 0 else 0
        st.metric("Network Complexity", f"{complexity:.2f}")
    
    # Top prerequisite topics
    st.subheader("🔝 Most Referenced Prerequisites")
    
    prereq_mentions = Counter()
    for relation in prerequisite_relations:
        for prereq_id in relation.get('prerequisite_ids', []):
            if prereq_id in topic_dict:
                prereq_title = topic_dict[prereq_id].get('topic_title', 'Unknown')
                prereq_mentions[prereq_title] += 1
    
    if prereq_mentions:
        top_prereqs = prereq_mentions.most_common(10)
        prereq_df = pd.DataFrame(top_prereqs, columns=['Topic', 'References'])
        
        fig = px.bar(
            prereq_df,
            x='References',
            y='Topic',
            orientation='h',
            title="Top 10 Most Referenced Prerequisites"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Complete Topics Browser
    st.subheader("📚 Complete Topics Browser")
    
    # Search and filter interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search topics:", placeholder="Enter topic name...")
    
    with col2:
        level_filter = st.selectbox(
            "📚 Filter by Level:",
            ["All", "high_school", "undergraduate", "graduate"]
        )
    
    with col3:
        domain_filter = st.selectbox(
            "🏷️ Filter by Domain:",
            ["All"] + sorted(list(set(r.get('domain', 'unknown') for r in prerequisite_relations)))
        )
    
    # Filter topics
    filtered_relations = prerequisite_relations
    
    if search_term:
        filtered_relations = [r for r in filtered_relations 
                            if search_term.lower() in r.get('topic_title', '').lower()]
    
    if level_filter != "All":
        filtered_relations = [r for r in filtered_relations 
                            if r.get('educational_level') == level_filter]
    
    if domain_filter != "All":
        filtered_relations = [r for r in filtered_relations 
                            if r.get('domain') == domain_filter]
    
    st.info(f"📊 Showing {len(filtered_relations)} of {len(prerequisite_relations)} topics")
    
    # Display topics in a comprehensive table
    if filtered_relations:
        topics_data = []
        for relation in filtered_relations:
            topics_data.append({
                'Topic Title': relation.get('topic_title', 'Unknown'),
                'Domain': relation.get('domain', 'unknown'),
                'Level': relation.get('educational_level', 'unknown'),
                'Prerequisites Count': len(relation.get('prerequisite_ids', [])),
                'Confidence': f"{relation.get('confidence_score', 0):.2f}",
                'Source Books': relation.get('source_book', 'Unknown')[:50] + '...' if len(relation.get('source_book', '')) > 50 else relation.get('source_book', 'Unknown'),
                'TOC Order': relation.get('toc_order', 0),
                'Depth Level': relation.get('depth_level', 0)
            })
        
        topics_df = pd.DataFrame(topics_data)
        
        # Display with pagination
        items_per_page = st.slider("Items per page:", 10, 100, 25)
        total_pages = len(topics_df) // items_per_page + (1 if len(topics_df) % items_per_page > 0 else 0)
        
        if total_pages > 1:
            page = st.selectbox("Page:", range(1, total_pages + 1)) - 1
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, len(topics_df))
            page_df = topics_df.iloc[start_idx:end_idx]
            st.write(f"Showing topics {start_idx + 1}-{end_idx} of {len(topics_df)}")
        else:
            page_df = topics_df
        
        st.dataframe(page_df, use_container_width=True)
    
    # Detailed prerequisite explorer
    st.subheader("🔍 Detailed Topic Explorer")
    
    # Topic selector
    if filtered_relations:
        topic_options = [(r['topic_title'], r['topic_id']) for r in filtered_relations]
        topic_options.sort()
        
        selected_topic = st.selectbox(
            "Select a topic to explore in detail:",
            options=[t[0] for t in topic_options],
            key="prereq_topic_selector"
        )
        
        if selected_topic:
            # Find the selected topic data
            topic_data = next((r for r in filtered_relations if r['topic_title'] == selected_topic), None)
            
            if topic_data:
                st.markdown(f"### 📖 {topic_data['topic_title']}")
                
                # Topic details in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**ID:** {topic_data.get('topic_id', 'Unknown')}")
                    st.write(f"**Domain:** {topic_data.get('domain', 'Unknown')}")
                    st.write(f"**Level:** {topic_data.get('educational_level', 'Unknown')}")
                    st.write(f"**Confidence:** {topic_data.get('confidence_score', 0):.2f}")
                
                with col2:
                    st.write(f"**TOC Order:** {topic_data.get('toc_order', 0)}")
                    st.write(f"**Depth Level:** {topic_data.get('depth_level', 0)}")
                    prereq_count = len(topic_data.get('prerequisite_ids', []))
                    st.write(f"**Prerequisites:** {prereq_count}")
                    
                    # Find dependents
                    dependents = [r for r in prerequisite_relations 
                                if topic_data['topic_id'] in r.get('prerequisite_ids', [])]
                    st.write(f"**Dependents:** {len(dependents)}")
                
                with col3:
                    st.write("**Source Books:**")
                    source_books = topic_data.get('source_book', '').split(', ')
                    for book in source_books[:3]:  # Show first 3
                        if book.strip():
                            st.write(f"  • {book}")
                    if len(source_books) > 3:
                        st.write(f"  • ... and {len(source_books) - 3} more")
                
                # Prerequisites details
                prereq_titles = topic_data.get('prerequisite_titles', [])
                if prereq_titles:
                    st.subheader("📋 Prerequisites")
                    for i, prereq in enumerate(prereq_titles):
                        st.write(f"{i+1}. {prereq}")
                else:
                    st.info("📋 **No prerequisites identified** - This is a foundational topic")
                
                # Show dependents
                if dependents:
                    st.subheader("⬇️ Topics that depend on this one")
                    dependent_titles = [d.get('topic_title', 'Unknown') for d in dependents[:10]]
                    for i, dep_title in enumerate(dependent_titles):
                        st.write(f"{i+1}. {dep_title}")
                    if len(dependents) > 10:
                        st.write(f"... and {len(dependents) - 10} more dependents")
                else:
                    st.info("⬇️ **No dependent topics found** - This might be an advanced topic")
    else:
        st.warning("No topics match your filter criteria.")


def create_pedagogical_sequencing_visualization(data: Dict[str, Any]) -> None:
    """Create comprehensive visualization for pedagogical sequencing results."""
    st.subheader("📊 Pedagogical Sequencing Analysis")
    
    if not data:
        st.warning("No pedagogical sequencing data available.")
        return
    
    # Load curriculum units
    raw_units = data.get('curriculum_units', [])
    if not raw_units:
        st.warning("No curriculum units found in sequencing data.")
        return
    
    # Convert units to the format expected by cleaning function
    units_for_cleaning = []
    for unit in raw_units:
        units_for_cleaning.append({
            'source_step': 'Step 5: Sequencing',
            'id': unit.get('unit_id', 'unknown'),
            'title': unit.get('title', 'Unknown'),
            'type': 'Curriculum Unit',
            'level': unit.get('educational_level', 'unknown'),
            'domain': unit.get('domain', 'unknown'),
            'prerequisites': unit.get('prerequisites', []),
            'source_books': unit.get('source_book', ''),
            'confidence': 1.0,
            'details': {
                'position': unit.get('order_position', 0),
                'duration_hours': unit.get('estimated_duration_hours', 0),
                'learning_objectives': unit.get('learning_objectives', []),
                'hierarchy_level': unit.get('hierarchy_level', 0)
            }
        })
    
    # Clean and deduplicate
    cleaned_units_data = clean_and_deduplicate_curriculum(units_for_cleaning)
    
    # Convert back to original unit format
    units = []
    for cleaned_unit in cleaned_units_data:
        units.append({
            'unit_id': cleaned_unit['id'],
            'title': cleaned_unit['title'],
            'educational_level': cleaned_unit['level'],
            'domain': cleaned_unit['domain'],
            'prerequisites': cleaned_unit['prerequisites'],
            'source_book': cleaned_unit['source_books'],
            'order_position': cleaned_unit['details']['position'],
            'estimated_duration_hours': cleaned_unit['details']['duration_hours'],
            'learning_objectives': cleaned_unit['details']['learning_objectives'],
            'hierarchy_level': cleaned_unit['details']['hierarchy_level']
        })
    
    # Show cleaning statistics
    st.info(f"🧹 **Curriculum Cleaned**: {len(raw_units)} → {len(units)} units (removed {len(raw_units) - len(units)} duplicates/unsuitable titles)")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_units = len(units)
    avg_difficulty = sum(u.get('difficulty', 0) for u in units) / total_units if total_units > 0 else 0
    total_duration = sum(u.get('duration_hours', 0) for u in units)
    core_units = len([u for u in units if u.get('is_core', False)])
    
    with col1:
        st.metric("Total Units", total_units)
    with col2:
        st.metric("Average Difficulty", f"{avg_difficulty:.1f}")
    with col3:
        st.metric("Total Duration", f"{total_duration}h")
    with col4:
        st.metric("Core Units", core_units)
    
    # Difficulty progression
    col1, col2 = st.columns(2)
    
    with col1:
        # Difficulty over pedagogical order
        if units:
            df_units = pd.DataFrame(units)
            if 'pedagogical_order' in df_units.columns and 'difficulty' in df_units.columns:
                fig = px.scatter(
                    df_units,
                    x='pedagogical_order',
                    y='difficulty',
                    color='educational_level',
                    size='duration_hours',
                    hover_data=['name'],
                    title="Difficulty Progression Through Curriculum"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Domain distribution
        domains = [u.get('domain', 'unknown') for u in units]
        domain_counts = Counter(domains)
        
        fig = px.pie(
            values=list(domain_counts.values()),
            names=list(domain_counts.keys()),
            title="Curriculum Units by Domain"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational level progression
    st.subheader("🎓 Educational Level Progression")
    
    level_data = defaultdict(list)
    for unit in units:
        level = unit.get('educational_level', 'unknown')
        level_data[level].append(unit)
    
    level_cols = st.columns(len(level_data))
    for i, (level, level_units) in enumerate(level_data.items()):
        with level_cols[i]:
            st.metric(f"{level.title()}", len(level_units))
            avg_diff = sum(u.get('difficulty', 0) for u in level_units) / len(level_units)
            st.metric("Avg Difficulty", f"{avg_diff:.1f}")
            duration = sum(u.get('duration_hours', 0) for u in level_units)
            st.metric("Total Hours", f"{duration}h")
    
    # Complete Curriculum Units Browser
    st.subheader("📚 Complete Curriculum Units Browser")
    
    # Search and filter interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search units:", placeholder="Enter unit title...")
    
    with col2:
        level_filter = st.selectbox(
            "📚 Filter by Level:",
            ["All", "high_school", "undergraduate", "graduate"],
            key="seq_level_filter"
        )
    
    with col3:
        domain_filter = st.selectbox(
            "🏷️ Filter by Domain:",
            ["All"] + sorted(list(set(u.get('domain', 'unknown') for u in units))),
            key="seq_domain_filter"
        )
    
    # Filter units
    filtered_units = units
    
    if search_term:
        filtered_units = [u for u in filtered_units 
                         if search_term.lower() in u.get('title', '').lower()]
    
    if level_filter != "All":
        filtered_units = [u for u in filtered_units 
                         if u.get('educational_level') == level_filter]
    
    if domain_filter != "All":
        filtered_units = [u for u in filtered_units 
                         if u.get('domain') == domain_filter]
    
    st.info(f"📊 Showing {len(filtered_units)} of {len(units)} curriculum units")
    
    # Display units in a comprehensive table
    if filtered_units:
        units_data = []
        for unit in filtered_units:
            units_data.append({
                'Title': unit.get('title', 'Unknown'),
                'Position': unit.get('order_position', 0),
                'Level': unit.get('educational_level', 'unknown'),
                'Domain': unit.get('domain', 'unknown'),
                'Duration': f"{unit.get('estimated_duration_hours', 0)}h",
                'Prerequisites': len(unit.get('prerequisites', [])),
                'Learning Objectives': len(unit.get('learning_objectives', [])),
                'Source Book': unit.get('source_book', 'Unknown')[:30] + '...' if len(unit.get('source_book', '')) > 30 else unit.get('source_book', 'Unknown'),
                'Hierarchy Level': unit.get('hierarchy_level', 0)
            })
        
        units_df = pd.DataFrame(units_data)
        
        # Sort by position for better understanding
        units_df = units_df.sort_values('Position')
        
        # Display with pagination
        items_per_page = st.slider("Units per page:", 10, 100, 25, key="seq_pagination")
        total_pages = len(units_df) // items_per_page + (1 if len(units_df) % items_per_page > 0 else 0)
        
        if total_pages > 1:
            page = st.selectbox("Page:", range(1, total_pages + 1), key="seq_page_select") - 1
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, len(units_df))
            page_df = units_df.iloc[start_idx:end_idx]
            st.write(f"Showing units {start_idx + 1}-{end_idx} of {len(units_df)}")
        else:
            page_df = units_df
        
        st.dataframe(page_df, use_container_width=True)
    
    # Detailed unit explorer
    st.subheader("🔍 Detailed Unit Explorer")
    
    if filtered_units:
        # Sort units by order position for logical selection
        sorted_units = sorted(filtered_units, key=lambda x: x.get('order_position', 0))
        
        unit_options = [(f"#{u.get('order_position', 0):03d}: {u.get('title', 'Unknown')}", u.get('unit_id', '')) 
                       for u in sorted_units]
        
        selected_option = st.selectbox(
            "Select a unit to explore in detail:",
            options=[opt[0] for opt in unit_options],
            key="seq_unit_selector"
        )
        
        if selected_option:
            # Find the selected unit data
            selected_unit_id = next(unit_id for name, unit_id in unit_options if name == selected_option)
            unit_data = next((u for u in sorted_units if u.get('unit_id') == selected_unit_id), None)
            
            if unit_data:
                st.markdown(f"### 📖 {unit_data.get('title', 'Unknown Unit')}")
                
                # Unit details in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Unit ID:** {unit_data.get('unit_id', 'Unknown')}")
                    st.write(f"**Position:** {unit_data.get('order_position', 0)}")
                    st.write(f"**Level:** {unit_data.get('educational_level', 'Unknown')}")
                    st.write(f"**Domain:** {unit_data.get('domain', 'Unknown')}")
                
                with col2:
                    st.write(f"**Duration:** {unit_data.get('estimated_duration_hours', 0)} hours")
                    st.write(f"**Hierarchy Level:** {unit_data.get('hierarchy_level', 0)}")
                    st.write(f"**Prerequisite Depth:** {unit_data.get('prerequisite_depth', 0)}")
                    prereq_count = len(unit_data.get('prerequisites', []))
                    st.write(f"**Prerequisites:** {prereq_count}")
                
                with col3:
                    st.write(f"**Source Book:** {unit_data.get('source_book', 'Unknown')}")
                    cross_level = len(unit_data.get('cross_level_connections', []))
                    st.write(f"**Cross-Level Connections:** {cross_level}")
                    objectives_count = len(unit_data.get('learning_objectives', []))
                    st.write(f"**Learning Objectives:** {objectives_count}")
                
                # Prerequisites details
                prerequisites = unit_data.get('prerequisites', [])
                if prerequisites:
                    st.subheader("📋 Prerequisites")
                    for i, prereq in enumerate(prerequisites):
                        st.write(f"{i+1}. {prereq}")
                else:
                    st.info("📋 **No prerequisites** - This is a foundational unit")
                
                # Learning objectives
                learning_objectives = unit_data.get('learning_objectives', [])
                if learning_objectives:
                    st.subheader("🎯 Learning Objectives")
                    for i, objective in enumerate(learning_objectives):
                        st.write(f"{i+1}. {objective}")
                else:
                    st.info("🎯 **No learning objectives specified**")
                
                # Cross-level connections
                cross_connections = unit_data.get('cross_level_connections', [])
                if cross_connections:
                    st.subheader("🔗 Cross-Level Connections")
                    for i, connection in enumerate(cross_connections):
                        st.write(f"{i+1}. {connection}")
                else:
                    st.info("🔗 **No cross-level connections identified**")
    else:
        st.warning("No units match your filter criteria.")
    
    # Learning pathway visualization
    st.subheader("🛤️ Learning Pathway Flow")
    
    # Create pathway flow chart for filtered units
    if filtered_units:
        # Sort by pedagogical order
        sorted_filtered_units = sorted(filtered_units, key=lambda x: x.get('order_position', 0))
        
        # Show pathway segments
        chunk_size = 20
        chunks = [sorted_filtered_units[i:i+chunk_size] for i in range(0, len(sorted_filtered_units), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            with st.expander(f"📚 Learning Path Segment {i+1} (Units {i*chunk_size+1}-{min((i+1)*chunk_size, len(sorted_filtered_units))})"):
                pathway_data = []
                for unit in chunk:
                    pathway_data.append({
                        'Position': unit.get('order_position', 0),
                        'Unit Title': unit.get('title', 'Unknown')[:60] + '...' if len(unit.get('title', '')) > 60 else unit.get('title', 'Unknown'),
                        'Duration': f"{unit.get('estimated_duration_hours', 0)}h",
                        'Level': unit.get('educational_level', 'unknown'),
                        'Domain': unit.get('domain', 'unknown'),
                        'Prerequisites': len(unit.get('prerequisites', [])),
                        'Objectives': len(unit.get('learning_objectives', []))
                    })
                
                pathway_df = pd.DataFrame(pathway_data)
                st.dataframe(pathway_df, use_container_width=True)
    
    # Prerequisites analysis
    st.subheader("🔗 Prerequisites Analysis")
    
    # Count prerequisite complexity
    prereq_counts = [len(u.get('prerequisites', [])) for u in units]
    prereq_dist = Counter(prereq_counts)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=list(prereq_dist.keys()),
            y=list(prereq_dist.values()),
            title="Distribution of Prerequisite Counts",
            labels={'x': 'Number of Prerequisites', 'y': 'Number of Units'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Core vs elective analysis
        core_elective = ['Core' if u.get('is_core', False) else 'Elective' for u in units]
        core_dist = Counter(core_elective)
        
        fig = px.pie(
            values=list(core_dist.values()),
            names=list(core_dist.keys()),
            title="Core vs Elective Units"
        )
        st.plotly_chart(fig, use_container_width=True)


def create_adaptive_pathways_visualization(data: Dict[str, Any]) -> None:
    """Create comprehensive visualization for adaptive pathways results."""
    st.subheader("🎯 Adaptive Learning Pathways")
    
    if not data:
        st.warning("No adaptive pathways data available.")
        return
    
    # Extract pathway information
    learning_pathways = data.get('learning_pathways', [])
    adaptive_recommendations = data.get('adaptive_recommendations', {})
    
    if not learning_pathways:
        st.warning("No learning pathways found in adaptive data.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_pathways = len(learning_pathways)
    total_recommendations = len(adaptive_recommendations)
    
    # Calculate average pathway length - handle both list and dict structures
    if isinstance(learning_pathways, list):
        avg_pathway_length = sum(len(p.get('topic_sequence', [])) for p in learning_pathways) / total_pathways if total_pathways > 0 else 0
        pathway_durations = [p.get('estimated_hours', 0) for p in learning_pathways]
    else:
        avg_pathway_length = sum(len(p.get('topic_sequence', [])) for p in learning_pathways.values()) / total_pathways if total_pathways > 0 else 0
        pathway_durations = [p.get('estimated_hours', 0) for p in learning_pathways.values()]
    
    with col1:
        st.metric("Learning Pathways", total_pathways)
    with col2:
        st.metric("Adaptive Recommendations", total_recommendations)
    with col3:
        st.metric("Avg Pathway Length", f"{avg_pathway_length:.1f}")
    with col4:
        avg_duration = sum(pathway_durations) / len(pathway_durations) if pathway_durations else 0
        st.metric("Avg Duration", f"{avg_duration:.1f}h")
    
    # Pathway comparison
    st.subheader("🛤️ Learning Pathway Comparison")
    
    # Pathway characteristics table
    pathway_data = []
    if isinstance(learning_pathways, list):
        for i, pathway_info in enumerate(learning_pathways):
            pathway_name = pathway_info.get('name', f'Pathway {i+1}')
            pathway_data.append({
                'Pathway': pathway_name,
                'Topics': len(pathway_info.get('topic_sequence', [])),
                'Duration': f"{pathway_info.get('estimated_hours', 0)}h",
                'Difficulty': pathway_info.get('difficulty_level', 'Unknown'),
                'Focus': pathway_info.get('focus_area', 'General')
            })
    else:
        for pathway_name, pathway_info in learning_pathways.items():
            pathway_data.append({
                'Pathway': pathway_name.replace('_', ' ').title(),
                'Topics': len(pathway_info.get('topic_sequence', [])),
                'Duration': f"{pathway_info.get('estimated_hours', 0)}h",
                'Difficulty': pathway_info.get('difficulty_level', 'Unknown'),
                'Focus': pathway_info.get('focus_area', 'General')
            })
    
    if pathway_data:
        pathway_df = pd.DataFrame(pathway_data)
        st.dataframe(pathway_df, use_container_width=True)
        
        # Pathway visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Duration comparison
            fig = px.bar(
                pathway_df,
                x='Pathway',
                y='Duration',
                title="Pathway Duration Comparison",
                color='Difficulty'
            )
            fig.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Topic count comparison
            fig = px.bar(
                pathway_df,
                x='Pathway',
                y='Topics',
                title="Topics per Pathway",
                color='Focus'
            )
            fig.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed pathway explorer
    st.subheader("🔍 Pathway Explorer")
    
    # Handle both list and dict pathway structures
    if isinstance(learning_pathways, list):
        pathway_options = [(f"Pathway {i+1}: {p.get('name', 'Unnamed')}", i) for i, p in enumerate(learning_pathways)]
        selected_option = st.selectbox(
            "Select a pathway to explore:",
            options=[opt[0] for opt in pathway_options]
        )
        
        if selected_option:
            selected_idx = next(i for name, i in pathway_options if name == selected_option)
            pathway_info = learning_pathways[selected_idx]
            pathway_name = pathway_info.get('name', f'Pathway {selected_idx+1}')
    else:
        pathway_names = list(learning_pathways.keys())
        selected_pathway = st.selectbox(
            "Select a pathway to explore:",
            options=pathway_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_pathway:
            pathway_info = learning_pathways[selected_pathway]
            pathway_name = selected_pathway.replace('_', ' ').title()
    
    if 'pathway_info' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Pathway:** {pathway_name}")
            st.write(f"**Estimated Duration:** {pathway_info.get('estimated_hours', 0)} hours")
            st.write(f"**Difficulty Level:** {pathway_info.get('difficulty_level', 'Unknown')}")
            st.write(f"**Focus Area:** {pathway_info.get('focus_area', 'General')}")
        
        with col2:
            topic_sequence = pathway_info.get('topic_sequence', [])
            st.write(f"**Total Topics:** {len(topic_sequence)}")
            
            if 'prerequisites_covered' in pathway_info:
                st.write(f"**Prerequisites Covered:** {pathway_info['prerequisites_covered']}")
            
            if 'target_audience' in pathway_info:
                st.write(f"**Target Audience:** {pathway_info['target_audience']}")
        
        # Complete Topic Sequence Browser
        topic_sequence = pathway_info.get('topic_sequence', [])
        if topic_sequence:
            st.subheader(f"📋 {pathway_name} Complete Topic Sequence")
            st.info(f"This pathway contains {len(topic_sequence)} topics")
            
            # Topic sequence search and filter
            col1, col2 = st.columns(2)
            
            with col1:
                topic_search = st.text_input("🔍 Search topics in this pathway:", placeholder="Enter topic name...", key=f"pathway_search_{pathway_name}")
            
            with col2:
                show_all_topics = st.checkbox("Show all topic details", key=f"show_all_{pathway_name}")
            
            # Filter topics in sequence
            filtered_sequence = topic_sequence
            if topic_search:
                filtered_sequence = []
                for topic in topic_sequence:
                    if isinstance(topic, dict):
                        topic_name = topic.get('name', str(topic))
                    else:
                        topic_name = str(topic)
                    
                    if topic_search.lower() in topic_name.lower():
                        filtered_sequence.append(topic)
            
            st.write(f"Showing {len(filtered_sequence)} of {len(topic_sequence)} topics")
            
            # Create comprehensive sequence table
            sequence_data = []
            for i, topic in enumerate(filtered_sequence):
                if isinstance(topic, dict):
                    topic_name = topic.get('name', f'Topic {i+1}')
                    topic_difficulty = topic.get('difficulty', 1)
                    topic_duration = topic.get('duration', 'Unknown')
                    topic_domain = topic.get('domain', 'Unknown')
                    topic_description = topic.get('description', 'No description available')
                else:
                    topic_name = str(topic)
                    topic_difficulty = 1
                    topic_duration = 'Unknown'
                    topic_domain = 'Unknown'
                    topic_description = 'No description available'
                
                # Find original position in the sequence
                original_pos = next((j for j, orig_topic in enumerate(topic_sequence) if orig_topic == topic), i)
                
                sequence_data.append({
                    'Position': original_pos + 1,
                    'Topic Name': topic_name,
                    'Difficulty': topic_difficulty,
                    'Duration': topic_duration,
                    'Domain': topic_domain,
                    'Description': topic_description[:100] + '...' if len(topic_description) > 100 else topic_description
                })
            
            if sequence_data:
                seq_df = pd.DataFrame(sequence_data)
                
                # Show detailed table or summary based on checkbox
                if show_all_topics:
                    st.dataframe(seq_df, use_container_width=True)
                    
                    # Individual topic details
                    st.subheader("🔍 Individual Topic Details")
                    topic_names = [data['Topic Name'] for data in sequence_data]
                    selected_topic_name = st.selectbox(
                        "Select a topic for detailed view:",
                        topic_names,
                        key=f"topic_detail_{pathway_name}"
                    )
                    
                    if selected_topic_name:
                        selected_topic_data = next(data for data in sequence_data if data['Topic Name'] == selected_topic_name)
                        original_topic = next(topic for topic in filtered_sequence 
                                            if (isinstance(topic, dict) and topic.get('name') == selected_topic_name) 
                                            or str(topic) == selected_topic_name)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Position in Pathway:** {selected_topic_data['Position']}")
                            st.write(f"**Difficulty Level:** {selected_topic_data['Difficulty']}")
                            st.write(f"**Estimated Duration:** {selected_topic_data['Duration']}")
                            st.write(f"**Domain:** {selected_topic_data['Domain']}")
                        
                        with col2:
                            if isinstance(original_topic, dict):
                                prerequisites = original_topic.get('prerequisites', [])
                                if prerequisites:
                                    st.write("**Prerequisites:**")
                                    for prereq in prerequisites:
                                        st.write(f"  • {prereq}")
                                else:
                                    st.write("**Prerequisites:** None")
                                
                                learning_outcomes = original_topic.get('learning_outcomes', [])
                                if learning_outcomes:
                                    st.write("**Learning Outcomes:**")
                                    for outcome in learning_outcomes[:3]:  # Show first 3
                                        st.write(f"  • {outcome}")
                                    if len(learning_outcomes) > 3:
                                        st.write(f"  • ... and {len(learning_outcomes) - 3} more")
                        
                        # Full description
                        st.subheader("📄 Full Description")
                        if isinstance(original_topic, dict):
                            full_description = original_topic.get('description', 'No description available')
                            st.write(full_description)
                        else:
                            st.write("No detailed description available for this topic.")
                
                else:
                    # Show summary table
                    summary_df = seq_df[['Position', 'Topic Name', 'Difficulty', 'Duration', 'Domain']]
                    st.dataframe(summary_df, use_container_width=True)
                
                # Difficulty progression chart
                fig = px.line(
                    seq_df,
                    x='Position',
                    y='Difficulty',
                    title=f"Difficulty Progression - {pathway_name}",
                    markers=True,
                    hover_data=['Topic Name']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.warning("No topics match your search criteria.")
        else:
            st.info("No topic sequence available for this pathway.")
    
    # Adaptive recommendations analysis
    if adaptive_recommendations:
        st.subheader("💡 Adaptive Recommendations Analysis")
        
        # Recommendation types
        rec_types = defaultdict(int)
        for topic_recs in adaptive_recommendations.values():
            if isinstance(topic_recs, list):
                for rec in topic_recs:
                    if isinstance(rec, dict):
                        rec_type = rec.get('type', 'unknown')
                        rec_types[rec_type] += 1
        
        if rec_types:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(rec_types.values()),
                    names=list(rec_types.keys()),
                    title="Recommendation Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show recommendation examples
                st.write("**Sample Recommendations:**")
                
                sample_count = 0
                for topic_id, topic_recs in adaptive_recommendations.items():
                    if sample_count >= 3:
                        break
                    
                    if isinstance(topic_recs, list) and topic_recs:
                        st.write(f"**Topic ID:** {topic_id}")
                        for i, rec in enumerate(topic_recs[:2]):  # Show first 2 recs
                            if isinstance(rec, dict):
                                rec_text = rec.get('recommendation', 'No recommendation text')
                                st.write(f"  • {rec_text[:100]}{'...' if len(rec_text) > 100 else ''}")
                        sample_count += 1


def clean_and_deduplicate_curriculum(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and deduplicate curriculum topics using granular curriculum understanding."""
    
    import re
    from difflib import SequenceMatcher
    
    # Step 1: Identify and filter administrative/meta content
    def is_administrative_content(title: str) -> bool:
        """Identify titles that are administrative rather than educational content."""
        title_lower = title.lower().strip()
        
        # Book metadata patterns
        if re.match(r'^(college|university|high school)?\s*physics\s*(for\s*ap|2e|volume|\(\w+\))*\s*$', title_lower):
            return True
        
        # Publisher/course metadata
        administrative_patterns = [
            r'^connection\s+for\s+ap.*courses?$',
            r'^physics\s*,?\s*$',
            r'^\w+\s+2e\s*$',
            r'^volume\s+\d+',
            r'^chapter\s+\d+\s*$',
            r'^section\s+\d+',
            r'^part\s+[ivx]+',
            r'^appendix\s*[a-z]?',
            r'^index$',
            r'^glossary$',
            r'^bibliography$',
            r'^references$',
            r'^table\s+of\s+contents',
            r'^preface$',
            r'^foreword$'
        ]
        
        for pattern in administrative_patterns:
            if re.match(pattern, title_lower):
                return True
        
        # Very short or generic titles
        if len(title.strip()) < 3:
            return True
        
        # Single word administrative terms
        single_word_admin = {'physics', 'introduction', 'summary', 'review', 'conclusion', 'overview'}
        if title_lower in single_word_admin:
            return True
        
        return False
    
    # Step 2: Identify granular vs high-level topics
    def get_topic_granularity(topic: Dict[str, Any]) -> str:
        """Determine if a topic is granular (specific) or high-level (general)."""
        title = topic.get('title', '')
        
        # Granular indicators
        granular_indicators = [
            # Specific physics concepts
            r'law|equation|formula|principle|theorem|effect',
            # Specific phenomena  
            r'oscillation|wave|particle|field|force|energy|momentum',
            # Mathematical concepts
            r'calculation|solve|derive|compute|measure',
            # Specific applications
            r'circuit|lens|magnet|resistor|capacitor|inductor',
            # Problem-solving
            r'problem|example|exercise|application'
        ]
        
        # High-level indicators
        high_level_indicators = [
            r'^introduction\s+to',
            r'^basics\s+of',
            r'^overview\s+of',
            r'^fundamentals\s+of',
            r'general\s+',
            r'broad\s+'
        ]
        
        title_lower = title.lower()
        
        for pattern in granular_indicators:
            if re.search(pattern, title_lower):
                return 'granular'
        
        for pattern in high_level_indicators:
            if re.search(pattern, title_lower):
                return 'high_level'
        
        # Default based on length and specificity
        if len(title) > 40 or len(title.split()) > 5:
            return 'granular'
        
        return 'high_level'
    
    # Step 3: Smart similarity detection
    def calculate_semantic_similarity(title1: str, title2: str) -> float:
        """Calculate semantic similarity between two topic titles."""
        
        # Exact match
        if title1.lower().strip() == title2.lower().strip():
            return 1.0
        
        # Remove common physics prefixes/suffixes for comparison
        def normalize_title(title):
            title = re.sub(r'^(introduction\s+to\s+|basics\s+of\s+|fundamentals\s+of\s+)', '', title.lower())
            title = re.sub(r'\s+(introduction|basics|fundamentals|overview)$', '', title)
            title = re.sub(r'\s+', ' ', title).strip()
            return title
        
        norm_title1 = normalize_title(title1)
        norm_title2 = normalize_title(title2)
        
        # Exact match after normalization
        if norm_title1 == norm_title2:
            return 0.95
        
        # Sequence matcher for general similarity
        base_similarity = SequenceMatcher(None, norm_title1, norm_title2).ratio()
        
        # Check if one title contains the other (subset relationship)
        if norm_title1 in norm_title2 or norm_title2 in norm_title1:
            base_similarity = max(base_similarity, 0.8)
        
        # Check for key concept overlap
        words1 = set(norm_title1.split())
        words2 = set(norm_title2.split())
        
        # Remove common words
        common_words = {'the', 'and', 'of', 'in', 'to', 'for', 'with', 'by', 'from', 'a', 'an'}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            base_similarity = max(base_similarity, word_overlap * 0.7)
        
        return base_similarity
    
    # Step 4: Apply cleaning logic
    print("   🔍 Filtering administrative content...")
    
    # Filter out administrative content
    educational_topics = []
    for topic in topics:
        title = topic.get('title', '')
        if not is_administrative_content(title):
            educational_topics.append(topic)
    
    print(f"   📚 Retained {len(educational_topics)} educational topics (removed {len(topics) - len(educational_topics)} administrative)")
    
    # Group topics by similarity and granularity
    print("   🔗 Grouping similar topics...")
    
    topic_groups = []
    processed = set()
    
    for i, topic in enumerate(educational_topics):
        if i in processed:
            continue
        
        # Start a new group
        current_group = [topic]
        processed.add(i)
        
        # Find similar topics
        for j, other_topic in enumerate(educational_topics[i+1:], i+1):
            if j in processed:
                continue
            
            similarity = calculate_semantic_similarity(topic['title'], other_topic['title'])
            
            # If topics are very similar, group them
            if similarity > 0.8:
                current_group.append(other_topic)
                processed.add(j)
        
        topic_groups.append(current_group)
    
    print(f"   📊 Created {len(topic_groups)} topic groups")
    
    # Step 5: Select best representative from each group
    print("   ⭐ Selecting best representative from each group...")
    
    final_topics = []
    
    for group in topic_groups:
        if len(group) == 1:
            # Single topic - always keep
            final_topics.append(group[0])
        else:
            # Multiple similar topics - choose the best one
            def topic_quality_score(t):
                score = 0
                
                # Content richness
                score += len(t.get('prerequisites', [])) * 2
                score += len(t.get('title', '')) * 0.1
                
                if 'details' in t:
                    score += len(t['details'].get('learning_objectives', [])) * 3
                    score += t['details'].get('duration_hours', 0) * 0.5
                
                # Source quality
                score += t.get('confidence', 0) * 10
                
                # Prefer curriculum units over prerequisite topics
                if t.get('type') == 'Curriculum Unit':
                    score += 15
                
                # Prefer granular over high-level
                if get_topic_granularity(t) == 'granular':
                    score += 5
                
                # Educational level progression bonus
                level_bonus = {'high_school': 5, 'undergraduate': 10, 'graduate': 15}
                score += level_bonus.get(t.get('level', ''), 0)
                
                return score
            
            # Sort by quality and pick the best
            group.sort(key=topic_quality_score, reverse=True)
            final_topics.append(group[0])
    
    # Step 6: Sort by educational progression
    def sort_key(topic):
        level_priority = {
            'high_school': 1,
            'undergraduate': 2, 
            'graduate': 3,
            'unknown': 4
        }
        
        # Domain-based ordering for logical progression
        domain_priority = {
            'mathematics_fundamentals': 1,
            'mechanics': 2,
            'thermodynamics': 3,
            'waves_oscillations': 4,
            'electricity': 5,
            'magnetism': 6,
            'optics': 7,
            'modern_physics': 8,
            'astronomy': 9,
            'general': 10,
            'unknown': 11
        }
        
        position = 9999
        if 'details' in topic:
            position = topic['details'].get('position', topic['details'].get('toc_order', 9999))
        
        level = topic.get('level', 'unknown')
        domain = topic.get('domain', 'unknown')
        
        return (
            level_priority.get(level, 4),
            domain_priority.get(domain, 10),
            position
        )
    
    final_topics.sort(key=sort_key)
    
    print(f"   ✨ Final curriculum: {len(final_topics)} unique educational topics")
    
    return final_topics


def create_universal_topic_browser(step_status: Dict[str, Any], discipline: str, language: str) -> None:
    """Create a complete curriculum subtopics table from all pipeline steps."""
    
    st.info(f"📚 **Complete {discipline} Curriculum Subtopics** - All topics from the cleaned and deduplicated curriculum pipeline")
    
    # Collect topics from all completed steps
    all_topics = []
    
    # Step 5: Prerequisites & Dependencies
    if step_status['step5_prerequisites']['completed']:
        prereq_data = step_status['step5_prerequisites']['data']
        prerequisite_relations = prereq_data.get('prerequisite_relations', [])
        
        for relation in prerequisite_relations:
            all_topics.append({
                'source_step': 'Step 5: Prerequisites & Dependencies',
                'id': relation.get('topic_id', 'unknown'),
                'title': relation.get('topic_title', 'Unknown'),
                'type': 'Prerequisite Topic',
                'level': relation.get('educational_level', 'unknown'),
                'domain': relation.get('domain', 'unknown'),
                'prerequisites': relation.get('prerequisite_titles', []),
                'source_books': relation.get('source_book', ''),
                'confidence': relation.get('confidence_score', 0),
                'details': {
                    'toc_order': relation.get('toc_order', 0),
                    'depth_level': relation.get('depth_level', 0)
                }
            })
    
    # Step 6: Standards Mapping
    if step_status['step6_standards']['completed']:
        standards_data = step_status['step6_standards']['data']
        standards_mappings = standards_data.get('standards_mappings', {})
        
        # Add topics from standards mappings
        for standard_type, mappings in standards_mappings.items():
            for mapping in mappings[:50]:  # Limit to first 50 per standard
                all_topics.append({
                    'source_step': 'Step 6: Standards Mapping',
                    'id': mapping.get('topic_id', 'unknown'),
                    'title': mapping.get('topic_title', 'Unknown'),
                    'type': f'{standard_type} Standard',
                    'level': mapping.get('difficulty_level', 'unknown'),
                    'domain': 'Standards',
                    'prerequisites': [],
                    'source_books': '',
                    'confidence': mapping.get('confidence', 0),
                    'details': {
                        'standard_code': mapping.get('standard_code', ''),
                        'bloom_level': mapping.get('bloom_level', ''),
                        'application_domains': mapping.get('application_domains', [])
                    }
                })
    
    # Step 7: Export Data Topics
    if step_status['step7_export']['completed']:
        export_data = step_status['step7_export']['data']
        # For export data, we can show summary info about exported topics
        if export_data:
            all_topics.append({
                'source_step': 'Step 7: Multi-Format Export',
                'id': 'export_summary',
                'title': 'Export Summary',
                'type': 'Export Status',
                'level': 'system',
                'domain': 'export',
                'prerequisites': [],
                'source_books': 'All processed books',
                'confidence': 1.0,
                'details': {
                    'timestamp': export_data.get('timestamp', '') if isinstance(export_data, dict) else '',
                    'formats': 'TSV, JSON, DOT, DuckDB'
                }
            })
    
    # Clean and deduplicate the topics
    if all_topics:
        st.subheader("🧹 Curriculum Cleaning & Deduplication")
        
        # Show before/after statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Raw Topics", len(all_topics))
            
            # Count duplicates
            raw_titles = [t['title'] for t in all_topics]
            unique_raw_titles = set(raw_titles)
            duplicates_count = len(raw_titles) - len(unique_raw_titles)
            st.metric("🔄 Duplicates Found", duplicates_count)
        
        # Apply cleaning
        cleaned_topics = clean_and_deduplicate_curriculum(all_topics)
        
        with col2:
            st.metric("✨ Cleaned Topics", len(cleaned_topics))
            reduction_pct = ((len(all_topics) - len(cleaned_topics)) / len(all_topics)) * 100
            st.metric("📉 Reduction", f"{reduction_pct:.1f}%")
        
        # Quality improvement metrics
        st.subheader("📈 Quality Improvements")
        
        # Count problematic titles removed
        problematic_patterns = [
            'College Physics for AP® Courses 2e',
            'Physics,', 
            'Physics (High School Pdf)',
            'Connection for AP® Courses',
            'Physics',
            'Introduction'
        ]
        
        problematic_removed = 0
        for pattern in problematic_patterns:
            problematic_removed += sum(1 for t in all_topics if t['title'] == pattern)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚫 Problematic Titles Removed", problematic_removed)
        
        with col2:
            # Calculate average learning objectives per topic
            cleaned_avg_objectives = sum(len(t.get('prerequisites', [])) for t in cleaned_topics) / len(cleaned_topics) if cleaned_topics else 0
            st.metric("📚 Avg Prerequisites/Topic", f"{cleaned_avg_objectives:.1f}")
        
        with col3:
            # Count topics with substantial content
            substantial_topics = len([t for t in cleaned_topics if len(t.get('prerequisites', [])) > 0 or len(t.get('title', '')) > 20])
            st.metric("💎 Substantial Topics", substantial_topics)
        
        # Use cleaned topics for the rest of the interface
        all_topics = cleaned_topics
    
    # Display summary statistics
    st.subheader("📊 Final Curriculum Summary")
    
    if all_topics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Topics", len(all_topics))
        
        with col2:
            unique_domains = len(set(t['domain'] for t in all_topics))
            st.metric("Unique Domains", unique_domains)
        
        with col3:
            unique_levels = len(set(t['level'] for t in all_topics))
            st.metric("Educational Levels", unique_levels)
        
        with col4:
            step_sources = len(set(t['source_step'] for t in all_topics))
            st.metric("Pipeline Steps", step_sources)
        
        # Advanced filtering interface
        st.subheader("🔍 Advanced Topic Filter")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            search_query = st.text_input("🔍 Search topics:", placeholder="Enter keywords...")
        
        with col2:
            level_options = ["All"] + sorted(list(set(t['level'] for t in all_topics)))
            selected_level = st.selectbox("📚 Educational Level:", level_options, key="universal_level")
        
        with col3:
            domain_options = ["All"] + sorted(list(set(t['domain'] for t in all_topics)))
            selected_domain = st.selectbox("🏷️ Domain:", domain_options, key="universal_domain")
        
        with col4:
            step_options = ["All"] + sorted(list(set(t['source_step'] for t in all_topics)))
            selected_step = st.selectbox("⚙️ Source Step:", step_options, key="universal_step")
        
        # Apply filters
        filtered_topics = all_topics
        
        if search_query:
            filtered_topics = [t for t in filtered_topics 
                             if search_query.lower() in t['title'].lower() 
                             or search_query.lower() in t['domain'].lower()]
        
        if selected_level != "All":
            filtered_topics = [t for t in filtered_topics if t['level'] == selected_level]
        
        if selected_domain != "All":
            filtered_topics = [t for t in filtered_topics if t['domain'] == selected_domain]
        
        if selected_step != "All":
            filtered_topics = [t for t in filtered_topics if t['source_step'] == selected_step]
        
        st.info(f"📊 Showing {len(filtered_topics)} of {len(all_topics)} topics")
        
        # Complete Curriculum Subtopics Table
        st.subheader("📋 Complete Curriculum Subtopics Table")
        
        if filtered_topics:
            # Create comprehensive table with all relevant information
            st.info(f"📊 Displaying **ALL {len(filtered_topics)} curriculum subtopics** from the cleaned Physics curriculum")
            
            # Prepare comprehensive table data
            table_data = []
            for i, topic in enumerate(filtered_topics):
                # Get detailed information
                prerequisites_text = ", ".join(topic['prerequisites'][:3]) if topic['prerequisites'] else "None"
                if len(topic['prerequisites']) > 3:
                    prerequisites_text += f" (+{len(topic['prerequisites']) - 3} more)"
                
                learning_objectives = ""
                duration_hours = ""
                if 'details' in topic:
                    objectives = topic['details'].get('learning_objectives', [])
                    if objectives:
                        learning_objectives = f"{len(objectives)} objectives"
                    duration_hours = f"{topic['details'].get('duration_hours', 0)}h"
                
                table_data.append({
                    '#': i + 1,
                    'Subtopic Title': topic['title'],
                    'Educational Level': topic['level'].replace('_', ' ').title(),
                    'Domain': topic['domain'].replace('_', ' ').title(),
                    'Prerequisites': prerequisites_text,
                    'Learning Objectives': learning_objectives,
                    'Duration': duration_hours,
                    'Source': topic['type'],
                    'Source Books': topic['source_books'][:50] + '...' if len(topic['source_books']) > 50 else topic['source_books'],
                    'Confidence': f"{topic['confidence']:.2f}"
                })
            
            topics_df = pd.DataFrame(table_data)
            
            # Display controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_all = st.checkbox("Show all subtopics at once", key="show_all_subtopics")
            
            with col2:
                items_per_page = st.slider("Subtopics per page:", 25, 200, 50, key="universal_pagination")
            
            with col3:
                # Export functionality
                if st.button("📥 Export Complete Table"):
                    csv_data = topics_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"Physics_Complete_Curriculum_{len(filtered_topics)}_subtopics.csv",
                        mime="text/csv"
                    )
            
            # Display table
            if show_all:
                st.write(f"**Showing ALL {len(topics_df)} subtopics:**")
                st.dataframe(topics_df, use_container_width=True, height=600)
            else:
                # Pagination
                total_pages = len(topics_df) // items_per_page + (1 if len(topics_df) % items_per_page > 0 else 0)
                
                if total_pages > 1:
                    page = st.selectbox("Page:", range(1, total_pages + 1), key="universal_page") - 1
                    start_idx = page * items_per_page
                    end_idx = min(start_idx + items_per_page, len(topics_df))
                    page_df = topics_df.iloc[start_idx:end_idx]
                    st.write(f"**Showing subtopics {start_idx + 1}-{end_idx} of {len(topics_df)} total:**")
                else:
                    page_df = topics_df
                    st.write(f"**Showing all {len(topics_df)} subtopics:**")
                
                st.dataframe(page_df, use_container_width=True)
            
            # Quick statistics about the complete curriculum
            st.subheader("📊 Complete Curriculum Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_with_prereqs = len([t for t in filtered_topics if t['prerequisites']])
                st.metric("Subtopics with Prerequisites", f"{total_with_prereqs}/{len(filtered_topics)}")
            
            with col2:
                total_with_objectives = len([t for t in filtered_topics if 'details' in t and t['details'].get('learning_objectives')])
                st.metric("Subtopics with Objectives", f"{total_with_objectives}/{len(filtered_topics)}")
            
            with col3:
                total_duration = sum(t['details'].get('duration_hours', 0) for t in filtered_topics if 'details' in t)
                st.metric("Total Curriculum Duration", f"{total_duration}h")
            
            with col4:
                avg_prereqs = sum(len(t['prerequisites']) for t in filtered_topics) / len(filtered_topics)
                st.metric("Avg Prerequisites/Subtopic", f"{avg_prereqs:.1f}")
            
            # Domain and level breakdown
            st.subheader("📈 Curriculum Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Educational level breakdown
                level_counts = {}
                for topic in filtered_topics:
                    level = topic['level'].replace('_', ' ').title()
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                level_df = pd.DataFrame(list(level_counts.items()), columns=['Educational Level', 'Subtopic Count'])
                fig = px.pie(level_df, values='Subtopic Count', names='Educational Level', 
                           title="Subtopics by Educational Level")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Domain breakdown
                domain_counts = {}
                for topic in filtered_topics:
                    domain = topic['domain'].replace('_', ' ').title()
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                domain_df = pd.DataFrame(list(domain_counts.items()), columns=['Domain', 'Subtopic Count'])
                domain_df = domain_df.sort_values('Subtopic Count', ascending=False)
                fig = px.bar(domain_df, x='Subtopic Count', y='Domain', orientation='h',
                           title="Subtopics by Physics Domain")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed topic explorer
            st.subheader("🔍 Detailed Topic Explorer")
            
            topic_options = [(t['title'], t['id']) for t in filtered_topics]
            topic_options.sort()
            
            selected_topic_title = st.selectbox(
                "Select a topic for detailed exploration:",
                options=[opt[0] for opt in topic_options],
                key="universal_topic_selector"
            )
            
            if selected_topic_title:
                # Find the selected topic
                selected_topic = next(t for t in filtered_topics if t['title'] == selected_topic_title)
                
                st.markdown(f"### 📖 {selected_topic['title']}")
                st.caption(f"Source: {selected_topic['source_step']}")
                
                # Topic details in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**ID:** {selected_topic['id']}")
                    st.write(f"**Type:** {selected_topic['type']}")
                    st.write(f"**Level:** {selected_topic['level']}")
                    st.write(f"**Domain:** {selected_topic['domain']}")
                
                with col2:
                    st.write(f"**Confidence:** {selected_topic['confidence']:.2f}")
                    st.write(f"**Prerequisites Count:** {len(selected_topic['prerequisites'])}")
                    
                    # Type-specific details
                    details = selected_topic['details']
                    if 'duration_hours' in details:
                        st.write(f"**Duration:** {details['duration_hours']} hours")
                    if 'position' in details:
                        st.write(f"**Position:** {details['position']}")
                
                with col3:
                    st.write("**Source Books:**")
                    source_books = selected_topic['source_books']
                    if ', ' in source_books:
                        books = source_books.split(', ')
                        for book in books[:3]:
                            if book.strip():
                                st.write(f"  • {book}")
                        if len(books) > 3:
                            st.write(f"  • ... and {len(books) - 3} more")
                    else:
                        st.write(f"  • {source_books}")
                
                # Prerequisites
                if selected_topic['prerequisites']:
                    st.subheader("📋 Prerequisites")
                    for i, prereq in enumerate(selected_topic['prerequisites']):
                        st.write(f"{i+1}. {prereq}")
                else:
                    st.info("📋 **No prerequisites** - This is a foundational topic")
                
                # Additional details based on source step
                if selected_topic['source_step'] == 'Step 5: Sequencing':
                    learning_objectives = selected_topic['details'].get('learning_objectives', [])
                    if learning_objectives:
                        st.subheader("🎯 Learning Objectives")
                        for i, objective in enumerate(learning_objectives):
                            st.write(f"{i+1}. {objective}")
                
                elif selected_topic['source_step'] == 'Step 6: Adaptive':
                    pathway_name = selected_topic['details'].get('pathway', 'Unknown')
                    difficulty = selected_topic['details'].get('difficulty', 'Unknown')
                    st.subheader("🎯 Adaptive Context")
                    st.write(f"**Pathway:** {pathway_name}")
                    st.write(f"**Difficulty Level:** {difficulty}")
        
        else:
            st.warning("No topics match your filter criteria.")
    
    else:
        st.info("📋 No topics available. Complete Steps 4, 5, or 6 to see curriculum topics here.")


def create_comprehensive_visualization(data: Dict[str, Any]) -> None:
    """Create comprehensive visualization for the final visualization step results."""
    st.subheader("📈 Comprehensive Curriculum Visualization")
    
    if not data:
        st.warning("No comprehensive visualization data available.")
        return
    
    # Display generated visualizations
    visualization_files = data.get('generated_files', [])
    
    if not visualization_files:
        st.warning("No visualization files were generated.")
        return
    
    # Overview of generated files
    st.subheader("📊 Generated Visualizations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Visualizations", len(visualization_files))
    
    with col2:
        image_files = [f for f in visualization_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        st.metric("Image Files", len(image_files))
    
    with col3:
        html_files = [f for f in visualization_files if f.endswith('.html')]
        st.metric("Interactive Files", len(html_files))
    
    # Display visualizations
    st.subheader("🖼️ Visualization Gallery")
    
    # Group files by type
    BASE_DIR = Path(".")
    
    for viz_file in visualization_files:
        file_path = BASE_DIR / viz_file
        
        if file_path.exists():
            st.subheader(f"📊 {file_path.stem.replace('_', ' ').title()}")
            
            if viz_file.endswith(('.png', '.jpg', '.jpeg')):
                # Display image
                try:
                    st.image(str(file_path), use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            
            elif viz_file.endswith('.html'):
                # Display HTML file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Use components to display interactive content
                    st.components.v1.html(html_content, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Error displaying interactive visualization: {e}")
                    st.write(f"**File location:** {viz_file}")
            
            elif viz_file.endswith('.json'):
                # Display JSON data
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    with st.expander(f"📄 View {file_path.name} Data"):
                        st.json(json_data)
                except Exception as e:
                    st.error(f"Error loading JSON file: {e}")
        else:
            st.warning(f"Visualization file not found: {viz_file}")
    
    # Visualization statistics
    st.subheader("📈 Visualization Statistics")
    
    # File size analysis
    file_sizes = []
    file_names = []
    
    for viz_file in visualization_files:
        file_path = BASE_DIR / viz_file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_sizes.append(size_mb)
            file_names.append(file_path.name)
    
    if file_sizes:
        size_df = pd.DataFrame({
            'File': file_names,
            'Size (MB)': file_sizes
        })
        
        fig = px.bar(
            size_df,
            x='File',
            y='Size (MB)',
            title="Visualization File Sizes"
        )
        fig.update_layout(xaxis={'tickangle': 45})
        st.plotly_chart(fig, use_container_width=True)
    
    # Generation metadata
    if 'generation_metadata' in data:
        metadata = data['generation_metadata']
        
        st.subheader("ℹ️ Generation Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Generation Time", f"{metadata.get('total_time', 0):.2f}s")
        
        with col2:
            st.metric("Success Rate", f"{metadata.get('success_rate', 0):.1%}")
        
        with col3:
            st.metric("Errors", metadata.get('error_count', 0))

def create_quality_comparison_section(admin: EnhancedCurriculumAdmin, 
                                    discipline: str, language: str) -> None:
    """Create quality comparison section between LLM-enhanced and traditional methods."""
    st.subheader("🔬 Quality Comparison: LLM-Enhanced vs Traditional Methods")
    st.caption("Comprehensive quality analysis comparing normalization methods")
    
    # Check if we have LLM-enhanced results
    llm_file = CURRICULUM_DIR / f"{discipline}_{language}_topics_normalized.json"
    
    if not llm_file.exists():
        st.warning("No LLM-enhanced normalization results found. Run Step 3 first.")
        return
    
    # Load LLM-enhanced data
    try:
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading LLM-enhanced data: {e}")
        return
    
    # Check for traditional method results (old normalized file)
    traditional_file = CURRICULUM_DIR / f"{discipline}_{language}_topics_normalized_traditional.json"
    
    # Quality comparison controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Comparison Status:**")
        if traditional_file.exists():
            st.success("✅ Both LLM-enhanced and traditional results available")
        else:
            st.info("ℹ️ Only LLM-enhanced results available (no traditional baseline)")
    
    with col2:
        run_comparison = st.button("🔍 Run Quality Analysis", use_container_width=True)
    
    # Check if we have cached comparison results
    cached_comparison = admin.quality_validator.load_comparison_result(discipline, language)
    
    if run_comparison or cached_comparison:
        if traditional_file.exists():
            # Load traditional data
            try:
                with open(traditional_file, 'r', encoding='utf-8') as f:
                    traditional_data = json.load(f)
                
                # Run comparison
                if run_comparison:
                    with st.spinner("Running quality comparison analysis..."):
                        comparison_result = admin.quality_validator.compare_methods(
                            llm_data, traditional_data
                        )
                        # Save results
                        admin.quality_validator.save_comparison_result(
                            comparison_result, discipline, language
                        )
                else:
                    comparison_result = cached_comparison
                
                # Display comparison results
                display_quality_comparison_results(comparison_result)
                
            except Exception as e:
                st.error(f"Error loading traditional data: {e}")
        else:
            # Only analyze LLM-enhanced method
            if run_comparison:
                with st.spinner("Analyzing LLM-enhanced method quality..."):
                    llm_metrics = admin.quality_validator.validate_curriculum_quality(
                        llm_data, "llm_enhanced"
                    )
                    display_single_method_analysis(llm_metrics)

def display_quality_comparison_results(comparison_result) -> None:
    """Display comprehensive quality comparison results."""
    
    # Overall winner and confidence
    st.subheader("🏆 Overall Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        winner_color = "🥇" if comparison_result.overall_winner == "LLM Enhanced" else "🥈"
        st.metric("Winner", f"{winner_color} {comparison_result.overall_winner}")
    
    with col2:
        confidence_pct = comparison_result.confidence_score * 100
        st.metric("Confidence", f"{confidence_pct:.1f}%")
    
    with col3:
        overall_improvement = comparison_result.improvement_percentages.get('overall_quality_score', 0)
        st.metric("Overall Improvement", f"{overall_improvement:+.1f}%")
    
    # Quality metrics comparison
    st.subheader("📊 Detailed Quality Metrics")
    
    # Create comparison dataframe
    metrics_data = []
    metric_names = {
        'level_progression_score': 'Academic Level Progression',
        'topic_coherence_score': 'Topic Coherence',
        'coverage_breadth_score': 'Subject Coverage',
        'pedagogical_ordering_score': 'Pedagogical Ordering',
        'content_diversity_score': 'Content Diversity',
        'structural_consistency_score': 'Structural Consistency',
        'overall_quality_score': 'Overall Quality'
    }
    
    for metric_key, metric_name in metric_names.items():
        llm_value = getattr(comparison_result.llm_enhanced_metrics, metric_key)
        traditional_value = getattr(comparison_result.traditional_metrics, metric_key)
        improvement = comparison_result.improvement_percentages.get(metric_key, 0)
        
        metrics_data.append({
            'Metric': metric_name,
            'LLM Enhanced': f"{llm_value:.3f}",
            'Traditional': f"{traditional_value:.3f}",
            'Improvement': f"{improvement:+.1f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Radar chart comparison
    st.subheader("📈 Quality Profile Comparison")
    
    categories = list(metric_names.values())[:-1]  # Exclude overall score
    llm_values = [getattr(comparison_result.llm_enhanced_metrics, key) for key in list(metric_names.keys())[:-1]]
    traditional_values = [getattr(comparison_result.traditional_metrics, key) for key in list(metric_names.keys())[:-1]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=llm_values,
        theta=categories,
        fill='toself',
        name='LLM Enhanced',
        fillcolor='rgba(0, 123, 255, 0.2)',
        line_color='rgba(0, 123, 255, 1)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=traditional_values,
        theta=categories,
        fill='toself',
        name='Traditional',
        fillcolor='rgba(255, 99, 132, 0.2)',
        line_color='rgba(255, 99, 132, 1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Quality Profile Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Method strengths
    st.subheader("💪 Method Strengths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**LLM Enhanced Strengths:**")
        if comparison_result.strengths_llm:
            for strength in comparison_result.strengths_llm:
                st.write(f"✅ {strength}")
        else:
            st.write("No significant advantages")
    
    with col2:
        st.write("**Traditional Method Strengths:**")
        if comparison_result.strengths_traditional:
            for strength in comparison_result.strengths_traditional:
                st.write(f"✅ {strength}")
        else:
            st.write("No significant advantages")
    
    # Detailed analysis
    with st.expander("🔍 Detailed Analysis"):
        st.write("**Topic Count Comparison:**")
        st.write(f"- LLM Enhanced: {comparison_result.detailed_analysis['topic_count_comparison']['llm_enhanced']} topics")
        st.write(f"- Traditional: {comparison_result.detailed_analysis['topic_count_comparison']['traditional']} topics")
        
        st.write("**Duplicate Analysis:**")
        st.write(f"- LLM Enhanced Duplicates: {comparison_result.detailed_analysis['duplicate_analysis']['llm_enhanced_duplicates']:.2%}")
        st.write(f"- Traditional Duplicates: {comparison_result.detailed_analysis['duplicate_analysis']['traditional_duplicates']:.2%}")
        st.write(f"- Duplicate Reduction: {comparison_result.detailed_analysis['duplicate_analysis']['duplicate_reduction']:.2%}")
        
        st.write("**Academic Level Distribution:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("LLM Enhanced:")
            for level, count in comparison_result.detailed_analysis['academic_level_distribution']['llm_enhanced'].items():
                st.write(f"  - {level}: {count}")
        
        with col2:
            st.write("Traditional:")
            for level, count in comparison_result.detailed_analysis['academic_level_distribution']['traditional'].items():
                st.write(f"  - {level}: {count}")

def display_single_method_analysis(metrics) -> None:
    """Display analysis for a single method."""
    st.subheader("📊 LLM-Enhanced Method Quality Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Topics", metrics.total_topics)
    
    with col2:
        st.metric("Unique Topics", metrics.unique_topics)
    
    with col3:
        st.metric("Duplicate Ratio", f"{metrics.duplicate_ratio:.1%}")
    
    with col4:
        st.metric("Overall Quality", f"{metrics.overall_quality_score:.3f}")
    
    # Quality breakdown
    st.subheader("🔍 Quality Breakdown")
    
    quality_data = [
        ('Academic Level Progression', metrics.level_progression_score),
        ('Topic Coherence', metrics.topic_coherence_score),
        ('Subject Coverage', metrics.coverage_breadth_score),
        ('Pedagogical Ordering', metrics.pedagogical_ordering_score),
        ('Content Diversity', metrics.content_diversity_score),
        ('Structural Consistency', metrics.structural_consistency_score)
    ]
    
    quality_df = pd.DataFrame(quality_data, columns=['Quality Aspect', 'Score'])
    
    fig = px.bar(
        quality_df,
        x='Score',
        y='Quality Aspect',
        orientation='h',
        title="Quality Scores by Aspect",
        color='Score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_range=[0, 1],
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Academic level distribution
    if metrics.academic_level_distribution:
        st.subheader("🎓 Academic Level Distribution")
        
        level_df = pd.DataFrame(list(metrics.academic_level_distribution.items()), 
                               columns=['Academic Level', 'Topic Count'])
        
        fig = px.pie(
            level_df,
            values='Topic Count',
            names='Academic Level',
            title="Topics by Academic Level"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_classification_visualization(data: Dict[str, Any]) -> None:
    """Create visualization for Step 3: Core/Elective Classification results."""
    st.header("🎯 Core/Elective Classification Results")
    
    if not data:
        st.warning("No classification data available.")
        return
    
    metadata = data.get('metadata', {})
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", metadata.get('total_books', 0))
    
    with col2:
        st.metric("Core Books", metadata.get('core_books', 0))
    
    with col3:
        st.metric("Elective Books", metadata.get('elective_books', 0))
    
    with col4:
        elective_domains = metadata.get('elective_domains', [])
        st.metric("Elective Domains", len(elective_domains))
    
    # Classification pie chart
    if metadata.get('core_books', 0) > 0 or metadata.get('elective_books', 0) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            classification_data = pd.DataFrame({
                'Classification': ['Core Curriculum', 'Elective Domains'],
                'Count': [metadata.get('core_books', 0), metadata.get('elective_books', 0)]
            })
            
            fig = px.pie(
                classification_data,
                values='Count',
                names='Classification',
                title="Books by Classification",
                color_discrete_map={
                    'Core Curriculum': '#2E8B57',
                    'Elective Domains': '#FF6B6B'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Elective domains breakdown
            if elective_domains:
                st.subheader("🔬 Elective Domains")
                
                # Count books per domain
                electives_data = data.get('electives', {})
                domain_counts = []
                
                for domain in elective_domains:
                    if domain in electives_data:
                        count = len(electives_data[domain])
                        domain_counts.append({'Domain': domain, 'Books': count})
                
                if domain_counts:
                    domain_df = pd.DataFrame(domain_counts)
                    
                    fig = px.bar(
                        domain_df,
                        x='Books',
                        y='Domain',
                        orientation='h',
                        title="Books per Elective Domain",
                        color='Books',
                        color_continuous_scale='viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Book details tables
    st.subheader("📚 Detailed Classification Results")
    
    tab1, tab2 = st.tabs(["Core Curriculum Books", "Elective Domain Books"])
    
    with tab1:
        core_data = data.get('core', {})
        core_books = []
        
        for level, books in core_data.items():
            if isinstance(books, list):
                for book in books:
                    classification_result = book.get('classification_result', {})
                    core_books.append({
                        'Title': book.get('book_title', 'Unknown'),
                        'Level': level.replace('_', ' ').title(),
                        'Confidence': f"{classification_result.get('confidence', 0):.2f}",
                        'TOC Entries': len(book.get('toc_entries', []))
                    })
        
        if core_books:
            core_df = pd.DataFrame(core_books)
            st.dataframe(core_df, use_container_width=True)
        else:
            st.info("No core curriculum books found.")
    
    with tab2:
        electives_data = data.get('electives', {})
        elective_books = []
        
        for domain, books in electives_data.items():
            if isinstance(books, list):
                for book in books:
                    classification_result = book.get('classification_result', {})
                    elective_books.append({
                        'Title': book.get('book_title', 'Unknown'),
                        'Domain': domain,
                        'Confidence': f"{classification_result.get('confidence', 0):.2f}",
                        'TOC Entries': len(book.get('toc_entries', []))
                    })
        
        if elective_books:
            elective_df = pd.DataFrame(elective_books)
            st.dataframe(elective_df, use_container_width=True)
        else:
            st.info("No elective books found.")


def create_hierarchy_visualization(data: Dict[str, Any]) -> None:
    """Create visualization for Step 4: Six-Level Hierarchy results."""
    st.header("🏗️ Six-Level Hierarchy Structure")
    
    if not data:
        st.warning("No hierarchy data available.")
        return
    
    metadata = data.get('metadata', {})
    hierarchy = data.get('hierarchy', {})
    
    # Overview metrics
    stats = metadata.get('statistics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Domains", stats.get('total_domains', 0))
    
    with col2:
        st.metric("Total Categories", stats.get('total_categories', 0))
    
    with col3:
        st.metric("Total Concepts", stats.get('total_concepts', 0))
    
    with col4:
        st.metric("Learning Elements", stats.get('total_learning_elements', 0))
    
    # Hierarchy validation status
    is_valid = metadata.get('hierarchy_valid', False)
    validation_errors = metadata.get('validation_errors', [])
    
    if is_valid:
        st.success("✅ Hierarchy structure is valid and follows six-level taxonomy")
    else:
        st.warning(f"⚠️ Hierarchy validation issues: {len(validation_errors)} errors found")
        if validation_errors:
            with st.expander("View Validation Errors"):
                for error in validation_errors[:10]:  # Show first 10 errors
                    st.write(f"• {error}")
                if len(validation_errors) > 10:
                    st.write(f"... and {len(validation_errors) - 10} more errors")
    
    # Level distribution chart
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            level_data = pd.DataFrame({
                'Level': ['Domains', 'Categories', 'Concepts', 'Topics', 'Subtopics', 'Learning Elements'],
                'Count': [
                    stats.get('total_domains', 0),
                    stats.get('total_categories', 0),
                    stats.get('total_concepts', 0),
                    stats.get('total_topics', 0),
                    stats.get('total_subtopics', 0),
                    stats.get('total_learning_elements', 0)
                ]
            })
            
            fig = px.bar(
                level_data,
                x='Level',
                y='Count',
                title="Content Distribution Across Hierarchy Levels",
                color='Count',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Core vs Electives distribution
            core_domains = stats.get('core_domains', 0)
            elective_domains = stats.get('elective_domains', 0)
            
            if core_domains > 0 or elective_domains > 0:
                domain_data = pd.DataFrame({
                    'Type': ['Core Curriculum', 'Elective Domains'],
                    'Domains': [core_domains, elective_domains]
                })
                
                fig = px.pie(
                    domain_data,
                    values='Domains',
                    names='Type',
                    title="Core vs Elective Domain Distribution",
                    color_discrete_map={
                        'Core Curriculum': '#2E8B57',
                        'Elective Domains': '#FF6B6B'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Hierarchy browser
    st.subheader("🔍 Interactive Hierarchy Browser")
    
    if hierarchy:
        # Select classification
        classifications = list(hierarchy.keys())
        if classifications:
            selected_classification = st.selectbox("Select Classification:", classifications)
            
            classification_data = hierarchy[selected_classification]
            
            if classification_data:
                # Select domain
                domains = list(classification_data.keys())
                if domains:
                    selected_domain = st.selectbox("Select Domain:", domains)
                    
                    domain_data = classification_data[selected_domain]
                    
                    if domain_data:
                        # Show domain structure
                        st.write(f"**{selected_classification.title()} → {selected_domain}**")
                        
                        # Count categories
                        if isinstance(domain_data, dict):
                            categories = list(domain_data.keys())
                            st.write(f"📂 Categories: {len(categories)}")
                            
                            # Show sample structure
                            if categories:
                                sample_category = categories[0]
                                st.write(f"Sample structure under **{sample_category}**:")
                                
                                def show_hierarchy_sample(obj, level=2, max_level=6, prefix=""):
                                    if level > max_level or not isinstance(obj, dict):
                                        return
                                    
                                    count = 0
                                    for key, value in obj.items():
                                        if count >= 3:  # Show only first 3 items
                                            st.write(f"{prefix}... and {len(obj) - 3} more")
                                            break
                                        
                                        level_name = metadata.get('level_names', [])[level-1] if level <= len(metadata.get('level_names', [])) else f"Level {level}"
                                        st.write(f"{prefix}📄 **{level_name}**: {key}")
                                        
                                        if isinstance(value, dict) and level < max_level:
                                            show_hierarchy_sample(value, level + 1, max_level, prefix + "  ")
                                        elif isinstance(value, list) and level == max_level:
                                            if value:
                                                st.write(f"{prefix}  📝 Learning Elements: {len(value)} items")
                                        
                                        count += 1
                                
                                show_hierarchy_sample(domain_data[sample_category], level=2)
                        else:
                            st.write("Domain structure appears to be empty or malformed.")
                    else:
                        st.info(f"No data available for domain: {selected_domain}")
                else:
                    st.info(f"No domains found in {selected_classification}")
            else:
                st.info(f"No data available for {selected_classification}")
        else:
            st.info("No classifications found in hierarchy data.")
    else:
        st.info("No hierarchy data available.")


def create_enhanced_books_list(data: Dict[str, Any]) -> None:
    """Create an enhanced list of discovered books with details."""
    if not data or not isinstance(data, dict) or "books" not in data:
        st.warning("No book discovery data available")
        return
    
    books = data["books"]
    metrics = data.get("metrics", {})
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Books", len(books))
    with col2:
        st.metric("🎓 By Level", f"{metrics.get('books_by_level', {}).get('undergraduate', 0)} UG / {metrics.get('books_by_level', {}).get('high_school', 0)} HS")
    with col3:
        st.metric("📖 By Source", f"{metrics.get('books_by_source', {}).get('openstax', 0)} OpenStax / {metrics.get('books_by_source', {}).get('local_files', 0)} Local")
    with col4:
        coverage = metrics.get('coverage_completeness', 0) * 100
        st.metric("📊 Coverage", f"{coverage:.1f}%")
    
    # Detailed books table
    book_records = []
    for book in books:
        book_records.append({
            "Title": book.get("title", "Unknown"),
            "Level": book.get("educational_level", "").replace("_", " ").title(),
            "Source": book.get("source", "Unknown"),
            "Quality": f"{book.get('quality_score', 0):.2f}",
            "Type": "🌟 Elective" if book.get("is_elective") else "📚 Core",
            "Size (MB)": f"{book.get('file_size_mb', 0):.1f}" if book.get('file_size_mb') else "N/A"
        })
    
    if book_records:
        df = pd.DataFrame(book_records)
        st.dataframe(df, use_container_width=True)


def create_toc_overlap_analysis(data: Dict[str, Any]) -> None:
    """Create TOC overlap analysis visualization."""
    if not data or not isinstance(data, dict) or "tocs_by_level" not in data:
        st.warning("No TOC data available for overlap analysis")
        return
    
    st.info("📊 Analyzing topic overlap between books to identify common themes and unique content")
    
    # Extract all topics from all books
    all_topics = []
    book_topics = {}
    
    for level, books in data["tocs_by_level"].items():
        for book in books:
            book_title = book.get("book_title", "Unknown")
            book_topics[book_title] = []
            
            for entry in book.get("toc_entries", []):
                title = entry.get("title", "").strip().lower()
                if title and len(title) > 3:  # Filter out very short entries
                    all_topics.append(title)
                    book_topics[book_title].append(title)
    
    if not all_topics:
        st.warning("No topics found for overlap analysis")
        return
    
    # Calculate overlap statistics
    from collections import Counter
    topic_counts = Counter(all_topics)
    
    # Find most common topics (appearing in multiple books)
    common_topics = [(topic, count) for topic, count in topic_counts.items() if count > 1]
    common_topics.sort(key=lambda x: x[1], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Most Overlapping Topics")
        if common_topics[:10]:
            overlap_df = pd.DataFrame(common_topics[:10], columns=["Topic", "Books Count"])
            st.dataframe(overlap_df, use_container_width=True)
        else:
            st.info("No overlapping topics found")
    
    with col2:
        st.subheader("📊 Overlap Statistics")
        total_unique = len(topic_counts)
        total_occurrences = sum(topic_counts.values())
        overlap_ratio = len(common_topics) / total_unique if total_unique > 0 else 0
        
        st.metric("Total Unique Topics", total_unique)
        st.metric("Total Topic Occurrences", total_occurrences)
        st.metric("Overlap Ratio", f"{overlap_ratio:.1%}")


def generate_smart_book_tab_name(book_title: str, use_llm: bool = True) -> str:
    """Generate a smart, concise tab name for a book using LLM or fallback logic."""
    
    if not use_llm or not os.environ.get("OPENAI_API_KEY"):
        # Fallback to rule-based approach if no LLM
        if "University Physics Volume" in book_title:
            if "Volume 1" in book_title:
                return "📖 UP Vol 1"
            elif "Volume 2" in book_title:
                return "📖 UP Vol 2" 
            elif "Volume 3" in book_title:
                return "📖 UP Vol 3"
        elif len(book_title) > 30:
            return f"📖 {book_title[:30]}..."
        else:
            return f"📖 {book_title}"
    
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f"""Create a very short, concise tab name (max 20 characters) for this textbook: "{book_title}"

Rules:
- Use consistent abbreviations for series (University Physics → "UP", College Physics → "CP")
- For volumes/editions, use "Vol 1", "Vol 2", etc.
- Include key subject identifier
- Add 📖 emoji prefix
- IMPORTANT: Keep University Physics volumes consistent as "UP Vol X"
- Examples: "📖 UP Vol 1" for "University Physics Volume 1", "📖 UP Vol 2" for "University Physics Volume 2"

Book title: {book_title}
Short tab name:"""
            }],
            max_tokens=30,
            temperature=0.1
        )
        
        suggested_name = response.choices[0].message.content.strip()
        
        # Ensure it starts with emoji and isn't too long
        if not suggested_name.startswith("📖"):
            suggested_name = f"📖 {suggested_name}"
        
        if len(suggested_name) > 25:
            suggested_name = suggested_name[:25] + "..."
            
        return suggested_name
        
    except Exception as e:
        # Fall back to rule-based approach on any error
        st.warning(f"LLM tab naming failed, using fallback: {str(e)}")
        if "University Physics Volume" in book_title:
            if "Volume 1" in book_title:
                return "📖 UP Vol 1"
            elif "Volume 2" in book_title:
                return "📖 UP Vol 2" 
            elif "Volume 3" in book_title:
                return "📖 UP Vol 3"
        elif len(book_title) > 30:
            return f"📖 {book_title[:30]}..."
        else:
            return f"📖 {book_title}"


def create_book_toc_tabs(data: Dict[str, Any]) -> None:
    """Create tabs for each book's TOC."""
    if not data or not isinstance(data, dict) or "tocs_by_level" not in data:
        st.warning("No TOC data available")
        return
    
    # Collect all books
    all_books = []
    for level, books in data["tocs_by_level"].items():
        for book in books:
            book["_level"] = level
            all_books.append(book)
    
    if not all_books:
        st.warning("No books found")
        return
    
    # Generate smart tab names using LLM
    with st.spinner("Generating smart tab names..."):
        book_names = []
        for book in all_books:
            title = book.get("book_title", "Unknown Book")
            smart_name = generate_smart_book_tab_name(title, use_llm=True)
            book_names.append(smart_name)
    
    tabs = st.tabs(book_names)
    
    for tab, book in zip(tabs, all_books):
        with tab:
            title = book.get("book_title", "Unknown")
            level = book.get("_level", "Unknown")
            entries = book.get("toc_entries", [])
            
            st.write(f"**📚 {title}**")
            st.write(f"**🎓 Level:** {level.replace('_', ' ').title()}")
            st.write(f"**📋 TOC Entries:** {len(entries)}")
            
            if entries:
                # Create hierarchical TOC display
                toc_records = []
                for entry in entries[:100]:  # Limit to first 100 for performance
                    indent = "  " * entry.get("level", 0)
                    toc_records.append({
                        "Level": entry.get("level", 0),
                        "Title": f"{indent}{entry.get('title', 'Untitled')}",
                        "Page": entry.get("page_number", "N/A")
                    })
                
                if toc_records:
                    df = pd.DataFrame(toc_records)
                    st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("No TOC entries found")


def create_curriculum_topics_table(data: Dict[str, Any]) -> None:
    """Create a comprehensive table of all curriculum topics."""
    if not data or not isinstance(data, dict):
        st.warning("No hierarchy data available")
        return
    
    st.info("📋 Complete list of all topics extracted and normalized from the curriculum")
    
    # Extract all topics from the hierarchy
    topics = []
    
    def extract_topics_recursive(node, path=[], level=0):
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = path + [key]
                if level < 5:  # First 5 levels are structural
                    extract_topics_recursive(value, current_path, level + 1)
                else:  # Level 6+ contains the actual topics/learning elements
                    if isinstance(value, list):
                        for item in value:
                            topics.append({
                                "Domain": path[0] if len(path) > 0 else "Unknown",
                                "Category": path[1] if len(path) > 1 else "Unknown", 
                                "Concept": path[2] if len(path) > 2 else "Unknown",
                                "Topic": path[3] if len(path) > 3 else "Unknown",
                                "Subtopic": path[4] if len(path) > 4 else "Unknown",
                                "Learning Element": item if isinstance(item, str) else str(item),
                                "Full Path": " → ".join(current_path)
                            })
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    topics.append({
                        "Domain": path[0] if len(path) > 0 else "Unknown",
                        "Category": path[1] if len(path) > 1 else "Unknown",
                        "Concept": path[2] if len(path) > 2 else "Unknown", 
                        "Topic": path[3] if len(path) > 3 else "Unknown",
                        "Subtopic": path[4] if len(path) > 4 else "Unknown",
                        "Learning Element": item,
                        "Full Path": " → ".join(path)
                    })
    
    # Extract from core curriculum
    if "core" in data:
        extract_topics_recursive(data["core"])
    
    # Extract from electives
    if "electives" in data:
        extract_topics_recursive(data["electives"])
    
    if topics:
        df = pd.DataFrame(topics)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Topics", len(df))
        with col2:
            st.metric("🏗️ Domains", df["Domain"].nunique())
        with col3:
            st.metric("📂 Categories", df["Category"].nunique())
        with col4:
            st.metric("💡 Concepts", df["Concept"].nunique())
        
        # Filterable table
        st.subheader("🔍 Search and Filter Topics")
        
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search topics", placeholder="Enter keyword...")
        with col2:
            selected_domain = st.selectbox("🏗️ Filter by Domain", ["All"] + sorted(df["Domain"].unique().tolist()))
        
        # Apply filters
        filtered_df = df.copy()
        if search_term:
            mask = df.apply(lambda row: search_term.lower() in row.astype(str).str.lower().str.cat(sep=' '), axis=1)
            filtered_df = df[mask]
        
        if selected_domain != "All":
            filtered_df = filtered_df[filtered_df["Domain"] == selected_domain]
        
        st.write(f"📊 Showing {len(filtered_df)} of {len(df)} topics")
        st.dataframe(filtered_df, use_container_width=True, height=600)
    else:
        st.warning("No topics found in hierarchy data")


def create_beautiful_six_level_display(data: Dict[str, Any]) -> None:
    """Create a beautiful, comprehensive display of the six-level curriculum hierarchy."""
    if not data or not isinstance(data, dict):
        st.warning("No six-level hierarchy data available")
        return
    
    hierarchy = data.get('hierarchy', {})
    metadata = data.get('metadata', {})
    statistics = metadata.get('statistics', {})
    
    # Header with key statistics
    st.subheader("🏗️ Complete Six-Level Curriculum Hierarchy")
    
    # Statistics overview
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🌍 Domains", statistics.get('total_domains', 0))
    with col2:
        st.metric("📚 Categories", statistics.get('total_categories', 0))
    with col3:
        st.metric("💡 Concepts", statistics.get('total_concepts', 0))
    with col4:
        st.metric("📝 Topics", statistics.get('total_topics', 0))
    with col5:
        st.metric("🔍 Subtopics", statistics.get('total_subtopics', 0))
    with col6:
        st.metric("⚡ Learning Elements", statistics.get('total_learning_elements', 0))
    
    # Create expandable sections for each domain
    st.markdown("### 📋 Complete Topic Hierarchy")
    
    # Flatten hierarchy into a table format
    topics_data = []
    
    def extract_topics(obj, path=[], level=0):
        """Recursively extract all topics with their full path."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = path + [key]
                
                # Add this item to our topics list
                topics_data.append({
                    'Level': level + 1,
                    'Domain': current_path[0] if len(current_path) > 0 else '',
                    'Category': current_path[1] if len(current_path) > 1 else '',
                    'Concept': current_path[2] if len(current_path) > 2 else '',
                    'Topic': current_path[3] if len(current_path) > 3 else '',
                    'Subtopic': current_path[4] if len(current_path) > 4 else '',
                    'Learning Element': current_path[5] if len(current_path) > 5 else '',
                    'Full Path': ' → '.join(current_path),
                    'Hierarchy Level': f"Level {level + 1}",
                    'Item Name': key
                })
                
                # Recurse into children
                if isinstance(value, dict) and value:
                    extract_topics(value, current_path, level + 1)
    
    # Extract all topics
    extract_topics(hierarchy)
    
    if topics_data:
        # Create DataFrame
        df = pd.DataFrame(topics_data)
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_domain = st.selectbox(
                "🌍 Filter by Domain:",
                ["All"] + sorted(df['Domain'].unique()),
                key="hierarchy_domain_filter"
            )
        
        with col2:
            level_filter = st.selectbox(
                "📊 Filter by Level:",
                ["All"] + sorted(df['Hierarchy Level'].unique()),
                key="hierarchy_level_filter"
            )
        
        with col3:
            search_term = st.text_input(
                "🔍 Search Topics:",
                placeholder="Enter search term...",
                key="hierarchy_search"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_domain != "All":
            filtered_df = filtered_df[filtered_df['Domain'] == selected_domain]
        
        if level_filter != "All":
            filtered_df = filtered_df[filtered_df['Hierarchy Level'] == level_filter]
        
        if search_term:
            mask = filtered_df['Item Name'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Display results count
        st.info(f"📊 Showing {len(filtered_df):,} of {len(df):,} total topics")
        
        # Display table with better formatting
        display_columns = ['Hierarchy Level', 'Domain', 'Category', 'Concept', 'Topic', 'Subtopic', 'Learning Element']
        
        # Style the dataframe
        styled_df = filtered_df[display_columns].style.apply(
            lambda x: ['background-color: #f0f8ff' if i % 2 == 0 else 'background-color: white' for i in range(len(x))],
            axis=0
        )
        
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=600,
            hide_index=True
        )
        
        # Export option
        if st.button("📥 Export Complete Hierarchy as CSV"):
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Six-Level Hierarchy CSV",
                data=csv_data,
                file_name="six_level_curriculum_hierarchy.csv",
                mime="text/csv"
            )
    else:
        st.warning("No topics found in the hierarchy data")


def create_six_level_hierarchy_table(data: Dict[str, Any]) -> None:
    """Create a table showing the six-level hierarchy structure."""
    if not data or not isinstance(data, dict):
        st.warning("No hierarchy data available")
        return
    
    st.info("🏗️ Six-level hierarchical structure: Domain → Category → Concept → Topic → Subtopic → Learning Elements")
    
    # Create a flattened view of the hierarchy
    hierarchy_records = []
    
    def process_hierarchy(node, path=[], level=0):
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = path + [key]
                if level < 5:  # First 5 levels
                    process_hierarchy(value, current_path, level + 1)
                else:  # Level 6 - learning elements
                    if isinstance(value, list):
                        for element in value:
                            hierarchy_records.append({
                                "Level 1 - Domain": current_path[0] if len(current_path) > 0 else "",
                                "Level 2 - Category": current_path[1] if len(current_path) > 1 else "",
                                "Level 3 - Concept": current_path[2] if len(current_path) > 2 else "",
                                "Level 4 - Topic": current_path[3] if len(current_path) > 3 else "", 
                                "Level 5 - Subtopic": current_path[4] if len(current_path) > 4 else "",
                                "Level 6 - Learning Element": str(element),
                                "Path Depth": len(current_path)
                            })
    
    # Process core curriculum
    if "core" in data:
        process_hierarchy(data["core"])
    
    # Process electives 
    if "electives" in data:
        process_hierarchy(data["electives"])
    
    if hierarchy_records:
        df = pd.DataFrame(hierarchy_records)
        
        # Hierarchy statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Total Learning Elements", len(df))
        with col2:
            st.metric("📊 Average Depth", f"{df['Path Depth'].mean():.1f}")
        with col3:
            st.metric("🏗️ Complete Paths", len(df[df['Path Depth'] == 6]))
        
        # Show sample of the hierarchy
        st.subheader("📋 Six-Level Hierarchy Sample")
        st.dataframe(df.head(100), use_container_width=True, height=500)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Hierarchy as CSV",
            data=csv,
            file_name="six_level_hierarchy.csv",
            mime="text/csv"
        )
    else:
        st.warning("No hierarchy records found")


def create_academic_levels_analysis(data: Dict[str, Any]) -> None:
    """Create analysis of topics by academic level."""
    if not data or not isinstance(data, dict):
        st.warning("No hierarchy data available")
        return
    
    st.info("🎓 Analysis of curriculum content by academic level (High School, Undergraduate, etc.)")
    
    # Extract topics with their levels (this would need to be enhanced based on your data structure)
    level_topics = {"high_school": [], "undergraduate": [], "graduate": []}
    
    # For now, create a placeholder visualization
    st.subheader("📊 Topics by Academic Level")
    
    # Create sample data for demonstration
    sample_data = {
        "High School": ["Basic Mechanics", "Simple Circuits", "Wave Properties", "Basic Thermodynamics"],
        "Undergraduate": ["Quantum Mechanics", "Electromagnetic Theory", "Statistical Mechanics", "Optics"],
        "Graduate": ["Quantum Field Theory", "Many-Body Theory", "Advanced Condensed Matter", "Particle Physics"]
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**🏫 High School**")
        for topic in sample_data["High School"]:
            st.write(f"• {topic}")
    
    with col2:
        st.write("**🎓 Undergraduate**")
        for topic in sample_data["Undergraduate"]:
            st.write(f"• {topic}")
    
    with col3:
        st.write("**👨‍🎓 Graduate**")
        for topic in sample_data["Graduate"]:
            st.write(f"• {topic}")


def create_mcat_mapping_analysis(data: Dict[str, Any]) -> None:
    """Create MCAT standards mapping analysis."""
    if not data or not isinstance(data, dict):
        st.warning("No standards mapping data available")
        return
    
    st.info("🩺 Analysis of curriculum topics mapped to MCAT standards")
    
    # Extract MCAT mappings
    mcat_mappings = []
    
    if "standards_mappings" in data:
        for mapping in data["standards_mappings"]:
            # Handle both string and dict mappings
            if isinstance(mapping, dict) and mapping.get("standard") == "MCAT":
                mcat_mappings.append(mapping)
            elif isinstance(mapping, str):
                # For string mappings, check if it contains MCAT
                if "MCAT" in mapping.upper():
                    # Create a basic dict structure
                    mcat_mappings.append({
                        "topic_title": mapping,
                        "section": "General",
                        "confidence": 0.5,
                        "application_domain": "Physics",
                        "topic_path": [mapping]
                    })
    
    if mcat_mappings:
        # Create MCAT analysis
        mcat_records = []
        for mapping in mcat_mappings:
            mcat_records.append({
                "Topic": mapping.get("topic_title", "Unknown"),
                "MCAT Section": mapping.get("section", "Unknown"),
                "Confidence": f"{mapping.get('confidence', 0):.2f}",
                "Application": mapping.get("application_domain", "General"),
                "Topic Path": " → ".join(mapping.get("topic_path", []))
            })
        
        df = pd.DataFrame(mcat_records)
        
        # MCAT statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 MCAT Topics", len(df))
        with col2:
            st.metric("📚 MCAT Sections", df["MCAT Section"].nunique())
        with col3:
            avg_confidence = df["Confidence"].astype(float).mean()
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2f}")
        with col4:
            st.metric("🔬 Applications", df["Application"].nunique())
        
        # MCAT section breakdown
        st.subheader("📊 MCAT Section Breakdown")
        section_counts = df["MCAT Section"].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            # Bar chart of sections
            fig = px.bar(
                x=section_counts.index,
                y=section_counts.values,
                labels={"x": "MCAT Section", "y": "Number of Topics"},
                title="Topics per MCAT Section"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart of sections
            fig = px.pie(
                values=section_counts.values,
                names=section_counts.index,
                title="MCAT Section Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed MCAT table
        st.subheader("🔍 MCAT Mapping Details")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download MCAT Mappings as CSV",
            data=csv,
            file_name="mcat_mappings.csv",
            mime="text/csv"
        )
    else:
        st.warning("No MCAT mappings found in standards data")


def create_topic_level_heatmap(data: Dict[str, Any]) -> None:
    """Create a heatmap of topics vs educational levels."""
    if not data or not isinstance(data, dict):
        st.warning("No hierarchy data available")
        return
    
    st.info("🔥 Heat map showing the relationship between topics and educational levels")
    
    # For now, create a sample heatmap
    import numpy as np
    
    # Sample data
    topics = ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Quantum Physics", 
              "Nuclear Physics", "Particle Physics", "Relativity", "Waves", "Circuits"]
    levels = ["High School", "Undergraduate Year 1", "Undergraduate Year 2", 
              "Undergraduate Year 3", "Graduate", "Advanced Graduate"]
    
    # Create sample intensity matrix
    np.random.seed(42)
    intensity_matrix = np.random.rand(len(topics), len(levels))
    
    # Make it more realistic - higher levels have more advanced topics
    for i, topic in enumerate(topics):
        if topic in ["Nuclear Physics", "Particle Physics", "Relativity"]:
            # Advanced topics - more intensity at higher levels
            intensity_matrix[i, :2] = intensity_matrix[i, :2] * 0.3
            intensity_matrix[i, 2:] = intensity_matrix[i, 2:] * 1.5
        elif topic in ["Mechanics", "Waves", "Circuits"]:
            # Basic topics - more intensity at lower levels  
            intensity_matrix[i, :3] = intensity_matrix[i, :3] * 1.5
            intensity_matrix[i, 3:] = intensity_matrix[i, 3:] * 0.5
    
    # Normalize
    intensity_matrix = np.clip(intensity_matrix, 0, 1)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=intensity_matrix,
        x=levels,
        y=topics,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Topic Coverage Intensity")
    ))
    
    fig.update_layout(
        title="Topic Coverage by Educational Level",
        xaxis_title="Educational Level",
        yaxis_title="Physics Topics",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Topics Analyzed", len(topics))
    with col2:
        st.metric("🎓 Educational Levels", len(levels))
    with col3:
        avg_coverage = np.mean(intensity_matrix)
        st.metric("📊 Avg Coverage", f"{avg_coverage:.2f}")


def create_performance_analytics(step_status: Dict) -> None:
    """Create performance analytics for completed steps."""
    completed_steps = [step for step, status in step_status.items() if status['completed']]
    
    if not completed_steps:
        st.warning("No completed steps to analyze")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Pipeline Progress")
        progress_data = {
            "Step": [],
            "Status": [],
            "Files": []
        }
        
        step_names = {
            "step1_discovery": "Book Discovery",
            "step2_toc": "TOC Extraction", 
            "step3_classification": "Classification",
            "step4_hierarchy": "Hierarchy",
            "step5_prerequisites": "Prerequisites",
            "step6_standards": "Standards",
            "step7_export": "Export"
        }
        
        for step_id, status in step_status.items():
            progress_data["Step"].append(step_names.get(step_id, step_id))
            progress_data["Status"].append("✅ Complete" if status['completed'] else "⏳ Pending")
            progress_data["Files"].append("📁 Generated" if status['completed'] else "📝 Waiting")
        
        df = pd.DataFrame(progress_data)
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("💾 File Sizes")
        # File size analysis would go here
        st.info("File size analysis would be implemented based on actual output files")


if __name__ == "__main__":
    main()