"""
OpenBooks Orchestrator - Core orchestration module for managing the complete acquisition workflow.

This module contains the main orchestration logic previously embedded in GetOpenBooks.py,
providing a clean separation between command-line interface and core functionality.
All orchestration logic is properly tested and uses only core modules.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .config import OpenBooksConfig
from .terminal_ui import TerminalUI
from .enhanced_logging import OperationLogger
from .book_discoverer import BookDiscoverer
from .repository_manager import RepositoryManager
from .content_processor import ContentProcessor
from .parallel_processor import ParallelProcessor, ProcessingTask, create_parallel_task_batches
from .pdf_downloader import PDFDownloader
from .search_indexer import SearchIndexer
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class OpenBooksOrchestrator:
    """
    Core orchestrator for the OpenBooks acquisition and management system.
    
    This class manages the complete workflow for discovering, acquiring, organizing,
    and indexing open educational textbooks. It uses only core functionality modules
    and is designed to be fully unit testable.
    """
    
    def __init__(self, config: OpenBooksConfig, terminal_ui: TerminalUI = None):
        """Initialize the orchestrator with configuration and UI."""
        self.config = config
        self.ui = terminal_ui or TerminalUI()
        self.logger = OperationLogger('orchestrator')
        
        # Initialize core components
        self.discoverer = BookDiscoverer(config)
        self.repo_manager = RepositoryManager(config)
        self.content_processor = ContentProcessor(config)
        self.search_indexer = SearchIndexer(config)
        self.language_detector = LanguageDetector()
        self.parallel_processor = ParallelProcessor(config)
        self.pdf_downloader = PDFDownloader(config)
        
        logger.info(f"OpenBooksOrchestrator initialized with {config.max_workers} workers")
    
    def run_complete_workflow(self, 
                            update_existing: bool = False,
                            dry_run: bool = False,
                            subjects: Optional[List[str]] = None,
                            language_filter: Optional[str] = None,
                            openstax_only: bool = True,
                            check_updates: bool = True,
                            git_only: bool = False) -> Dict[str, Any]:
        """
        Execute the complete OpenBooks workflow.
        
        Args:
            update_existing: Whether to update existing repositories
            dry_run: Preview mode without making changes
            subjects: List of subjects to focus on
            language_filter: Language filter for discovery
            openstax_only: Restrict to OpenStax repositories only
            check_updates: Check existing repositories for updates
            
        Returns:
            Dictionary with workflow results and statistics
        """
        start_time = time.time()
        workflow_results = {
            'phases_completed': [],
            'total_books': 0,
            'errors': 0,
            'warnings': 0,
            'duration': 0
        }
        
        try:
            self.ui.print_header("OpenBooks Acquisition System", 
                               "Comprehensive open textbook discovery and management")
            
            # Display configuration
            self._display_configuration(dry_run, openstax_only, check_updates)
            
            if dry_run:
                self.ui.print_info("DRY RUN MODE - No changes will be made")
            
            # Phase 1: Catalog existing collection
            self._run_phase("catalog_existing_collection", 1, 
                          lambda: self.catalog_existing_collection(openstax_only, language_filter),
                          workflow_results)
            
            # Phase 2: Repository updates (if enabled)
            if check_updates:
                self._run_phase("update_existing_repositories", 2,
                              lambda: self.update_existing_repositories(dry_run),
                              workflow_results)
            
            # Phase 3: Auto-discover available books
            self._run_phase("discover_books", 3,
                          lambda: self.discover_books(subjects, language_filter, openstax_only, git_only),
                          workflow_results)
            
            # Phase 4: Acquire books
            self._run_phase("acquire_books", 4,
                          lambda: self.acquire_books(dry_run, openstax_only),
                          workflow_results)
            
            # Phase 5: Update existing books (if enabled)
            if update_existing:
                self._run_phase("update_books", 5,
                              lambda: self.update_existing_books(dry_run),
                              workflow_results)
            
            # Phase 6: Generate catalog
            self._run_phase("generate_catalog", 6,
                          lambda: self.generate_catalog(),
                          workflow_results)
            
            # Phase 7: Build search index
            self._run_phase("build_search_index", 7,
                          lambda: self.build_search_index(),
                          workflow_results)
            
            # Final statistics
            workflow_results['duration'] = time.time() - start_time
            self._display_final_statistics(workflow_results)
            
        except Exception as e:
            self.logger.log_error(e, "workflow execution", "orchestrator", "error")
            workflow_results['errors'] += 1
        finally:
            self.parallel_processor.stop()
        
        return workflow_results
    
    def _load_statistics_from_booklist(self, books_path: Path) -> Optional[Dict[str, Any]]:
        """Load statistics from existing BookList.json if available."""
        booklist_path = books_path / "BookList.json"
        if not booklist_path.exists():
            return None
        
        try:
            import json
            with open(booklist_path, 'r', encoding='utf-8') as f:
                books = json.load(f)
            
            # Calculate statistics from BookList.json
            stats = {
                "pdfs": [],
                "git_repos": [],
                "other_formats": [],
                "total_size_mb": 0,
                "languages": {},
                "levels": {},
                "subjects": {},
                "organizations": {}
            }
            
            for book in books:
                # Categorize by format
                if book['format'] == 'pdf':
                    stats["pdfs"].append(book)
                elif book['format'] == 'git_repository':
                    stats["git_repos"].append(book)
                else:
                    stats["other_formats"].append(book)
                
                # Count statistics
                language = book.get('language', 'unknown')
                level = book.get('level', 'unknown')
                discipline = book.get('discipline', 'unknown')
                
                stats["languages"][language] = stats["languages"].get(language, 0) + 1
                stats["levels"][level] = stats["levels"].get(level, 0) + 1
                stats["subjects"][discipline] = stats["subjects"].get(discipline, 0) + 1
                stats["organizations"]["OpenStax"] = stats["organizations"].get("OpenStax", 0) + 1
                
                # Calculate size if available (for PDFs)
                if book['format'] == 'pdf' and book.get('pdf_file'):
                    try:
                        size_mb = Path(book['pdf_file']).stat().st_size / (1024 * 1024)
                        stats["total_size_mb"] += size_mb
                    except:
                        pass
            
            # Also find any standalone/media PDFs not tracked in BookList
            # This includes media files within repositories and any standalone PDFs
            existing_pdfs = self.pdf_downloader.get_existing_pdfs()
            
            # Add PDFs that aren't already tracked as main books
            for pdf_info in existing_pdfs:
                # Check if this PDF is already counted as a main book
                pdf_path = pdf_info.get('path', '')
                already_tracked = any(
                    book.get('pdf_file') == pdf_path for book in books
                )
                
                if not already_tracked:
                    stats["pdfs"].append(pdf_info)
                    stats["total_size_mb"] += pdf_info.get("size_mb", 0)
            
            return stats
            
        except Exception as e:
            self.logger.log_parsing_error(str(booklist_path), "BookList.json", str(e))
            return None

    def catalog_existing_collection(self, openstax_only: bool = False, 
                                  language_filter: Optional[str] = None) -> Dict[str, Any]:
        """Catalog existing books in the Books directory."""
        op = self.logger.start_operation("catalog_existing_collection", 
                                       openstax_only=openstax_only, 
                                       language_filter=language_filter)
        
        self.ui.print_phase(1, "Cataloging Existing Collection", 
                           "Scanning Books directory for PDFs and repositories")
        
        if openstax_only:
            self.ui.print_info("Filtering to OpenStax-only repositories")
        if language_filter:
            self.ui.print_info(f"Language filter: {language_filter}")
        
        books_path = Path(self.config.books_path)
        if not books_path.exists():
            error = FileNotFoundError(f"Books directory does not exist: {books_path}")
            self.logger.log_file_error(str(books_path), "catalog", error)
            self.logger.end_operation(False)
            return {
                "pdfs": [], "git_repos": [], "other_formats": [],
                "total_size_mb": 0, "languages": {}, "levels": {}, 
                "subjects": {}, "organizations": {}
            }
        
        # Try to load from existing BookList.json first
        existing_books = self._load_statistics_from_booklist(books_path)
        
        if existing_books is not None:
            self.ui.print_info("Using existing BookList.json for statistics")
        else:
            # Fall back to directory scanning
            self.ui.print_info("BookList.json not found, scanning directories")
            existing_books = {
                "pdfs": [],
                "git_repos": [],
                "other_formats": [],
                "total_size_mb": 0,
                "languages": {},
                "levels": {},
                "subjects": {},
                "organizations": {}
            }
            
            # Find PDFs in hierarchical structure (Language/Discipline/Level/*.pdf)
            existing_pdfs = self.pdf_downloader.get_existing_pdfs()
            for pdf_info in existing_pdfs:
                existing_books["pdfs"].append(pdf_info)
                existing_books["total_size_mb"] += pdf_info.get("size_mb", 0)
                
                # Update statistics
                language = pdf_info.get("language", "unknown")
                level = pdf_info.get("level", "unknown") 
                discipline = pdf_info.get("discipline", "unknown")
                
                existing_books["languages"][language] = existing_books["languages"].get(language, 0) + 1
                existing_books["levels"][level] = existing_books["levels"].get(level, 0) + 1
                existing_books["subjects"][discipline] = existing_books["subjects"].get(discipline, 0) + 1
            
            # First collect all repository paths for progress tracking
            all_repo_paths = []
            self._collect_repository_paths(books_path, all_repo_paths, openstax_only, language_filter)
            
            # Process repositories with progress bar
            if all_repo_paths:
                for i, repo_path in enumerate(all_repo_paths, 1):
                    self.ui.print_progress(i, len(all_repo_paths), "Scanning", details=repo_path.name)
                    self._process_repository(repo_path, existing_books, openstax_only, language_filter)
                
                # Clear progress line
                self.ui.clear_line()
        
        # Display comprehensive results using terminal UI
        stats = {
            "PDFs": len(existing_books['pdfs']),
            "Git Repositories": len(existing_books['git_repos']),
            "Other Formats": len(existing_books['other_formats']),
            "Total Size (MB)": f"{existing_books['total_size_mb']:.1f}",
            "Breakdown by Level": existing_books['levels'],
            "Breakdown by Subject": existing_books['subjects'],
            "Breakdown by Organization": existing_books['organizations']
        }
        
        self.ui.print_statistics(stats)
        
        # Language distribution
        if existing_books['languages']:
            self.ui.print_language_summary(existing_books['languages'])
        
        self.logger.end_operation(True, pdfs=len(existing_books['pdfs']), 
                                 repos=len(existing_books['git_repos']), 
                                 size_mb=existing_books['total_size_mb'])
        
        return existing_books
    
    def discover_books(self, subjects: Optional[List[str]] = None,
                      language_filter: Optional[str] = None,
                      openstax_only: bool = True,
                      git_only: bool = False) -> Dict[str, Any]:
        """
        Discover available books using the discovery engine.
        
        Args:
            subjects: List of subjects to focus on
            language_filter: Language filter for discovery
            openstax_only: Restrict to OpenStax repositories only
            
        Returns:
            Dictionary with discovery results
        """
        self.ui.print_info("🔍 Discovering OpenStax and CNX repositories...")
        discovered_books = self.discoverer.discover_openstax_books(openstax_only=openstax_only, git_only=git_only)
        
        # Store discovered books for later acquisition
        self.discovered_books = discovered_books
        
        self.ui.print_success(f"📚 Discovered {len(discovered_books)} books from GitHub repositories")
        
        return {
            'books_discovered': len(discovered_books),
            'discovery_successful': True,
            'books': discovered_books
        }
    
    def acquire_books(self, dry_run: bool = False, openstax_only: bool = True) -> Dict[str, Any]:
        """
        Acquire discovered books using parallel processing.
        
        Args:
            dry_run: Preview mode without making changes
            openstax_only: Restrict to OpenStax repositories only
            
        Returns:
            Dictionary with acquisition results
        """
        # Get discovered books from previous discovery phase
        discovered_books = getattr(self, 'discovered_books', [])
        
        if not discovered_books:
            self.ui.print_warning("No books discovered for acquisition")
            return {'books_acquired': 0, 'success_count': 0, 'error_count': 0}
        
        # Separate PDF books from Git repositories
        pdf_books = [book for book in discovered_books if book.get('format') == 'pdf']
        git_books = [book for book in discovered_books if book.get('format') != 'pdf']
        
        # Create parallel tasks for Git repository acquisition
        discovery_tasks, clone_tasks, processing_tasks = create_parallel_task_batches(
            git_books, self.config, openstax_only=openstax_only
        )
        
        if dry_run:
            self.ui.print_info(f"📥 [DRY RUN] Would acquire {len(git_books)} Git repositories and {len(pdf_books)} PDFs")
            self._preview_acquisition(discovered_books)
            return {'books_acquired': len(discovered_books), 'success_count': 0, 'error_count': 0}
        
        # Execute parallel Git repository acquisition
        git_results = {'success_count': 0, 'error_count': 0}
        if git_books:
            self.ui.print_info(f"📥 Acquiring {len(git_books)} Git repositories using {self.config.max_workers} workers...")
            results = self.parallel_processor.process_batch_parallel(
                discovery_tasks=discovery_tasks,
                clone_tasks=clone_tasks,
                processing_tasks=processing_tasks
            )
            git_results = self._process_acquisition_results(results)
        
        # Download PDF books
        pdf_results = {'successful': 0, 'failed': 0, 'already_exists': 0}
        if pdf_books:
            self.ui.print_info(f"📥 Downloading {len(pdf_books)} PDF books...")
            pdf_results = self.pdf_downloader.download_pdfs_batch(pdf_books, dry_run=False)
            self.ui.print_info(f"📄 PDF downloads: {pdf_results['successful']} successful, {pdf_results['failed']} failed, {pdf_results['already_exists']} already existed")
        
        # Combine results
        total_success = git_results['success_count'] + pdf_results['successful'] + pdf_results['already_exists']
        total_error = git_results['error_count'] + pdf_results['failed']
        
        self.ui.print_success(f"✅ Acquisition completed: {total_success} successful, {total_error} errors")
        
        combined_results = {
            'books_acquired': len(discovered_books),
            'success_count': total_success,
            'error_count': total_error,
            'git_results': git_results,
            'pdf_results': pdf_results
        }
        
        return combined_results
    
    def update_existing_repositories(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Update all existing repositories.
        
        Args:
            dry_run: Preview mode without making changes
            
        Returns:
            Dictionary with update results
        """
        books_path = Path(self.config.books_path)
        update_results = {'updated_count': 0, 'error_count': 0, 'repositories': []}
        
        if not books_path.exists():
            return update_results
        
        # Find all git repositories
        git_repos = []
        for git_dir in books_path.rglob(".git"):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                git_repos.append(repo_path)
        
        self.ui.print_info(f"Found {len(git_repos)} repositories to update")
        
        for i, repo_path in enumerate(git_repos):
            self.ui.print_progress(i + 1, len(git_repos), "Updating repositories", 
                                 details=repo_path.name)
            
            if dry_run:
                self.ui.print_info(f"[DRY RUN] Would check and update: {repo_path}")
                continue
            
            try:
                success = self.repo_manager.update_repository(str(repo_path))
                if success:
                    update_results['updated_count'] += 1
                else:
                    update_results['error_count'] += 1
                
                update_results['repositories'].append({
                    'path': str(repo_path),
                    'name': repo_path.name,
                    'success': success
                })
                
            except Exception as e:
                self.logger.log_error(e, f"updating repository {repo_path.name}", 
                                    "repository_manager", "warning")
                update_results['error_count'] += 1
        
        return update_results
    
    def update_existing_books(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Update existing books (alias for update_existing_repositories).
        
        Args:
            dry_run: Preview mode without making changes
            
        Returns:
            Dictionary with update results
        """
        return self.update_existing_repositories(dry_run)
    
    def generate_catalog(self) -> Dict[str, Any]:
        """
        Generate comprehensive catalog of the collection.
        
        Returns:
            Dictionary with catalog generation results
        """
        try:
            # Get existing collection info
            existing_collection = self.catalog_existing_collection()
            
            # Generate markdown catalog
            markdown_catalog = self.content_processor.generate_catalog_markdown(existing_collection)
            
            # Generate JSON catalog  
            json_catalog = self.content_processor.generate_json_catalog(existing_collection)
            
            # Save catalog
            self.content_processor.save_catalog(existing_collection)
            
            return {
                'markdown_generated': True,
                'json_generated': True,
                'catalog_saved': True,
                'total_books': len(existing_collection.get('git_repos', [])),
                'markdown_length': len(markdown_catalog),
                'json_entries': len(json_catalog.get('books', []))
            }
            
        except Exception as e:
            self.logger.log_error(e, "generating catalog", "content_processor", "error")
            return {
                'markdown_generated': False,
                'json_generated': False,
                'catalog_saved': False,
                'error': str(e)
            }
    
    def build_search_index(self) -> Dict[str, Any]:
        """
        Build search index for the collection.
        
        Returns:
            Dictionary with search index results
        """
        books_path = Path(self.config.books_path)
        
        if not books_path.exists():
            return {'books_indexed': 0, 'terms_indexed': 0, 'size_mb': 0.0}
        
        # Find all repositories to index
        repo_paths = []
        for git_dir in books_path.rglob(".git"):
            if git_dir.is_dir():
                repo_paths.append(git_dir.parent)
        
        # Index repositories
        indexed_count = 0
        for repo_path in repo_paths:
            try:
                if self.index_book(repo_path):
                    indexed_count += 1
            except Exception as e:
                self.logger.log_error(e, f"indexing {repo_path.name}", "search_indexer", "warning")
        
        # Get final index statistics
        stats = self.search_indexer.get_index_stats()
        
        return {
            'books_indexed': indexed_count,
            'terms_indexed': stats.unique_terms,
            'size_mb': stats.index_size_mb
        }
    
    def index_book(self, repo_path: Path) -> bool:
        """Index a single book repository for search."""
        try:
            # Extract metadata from the repository
            metadata = self.content_processor.extract_book_metadata(str(repo_path))
            
            if metadata:
                # For now, just log successful metadata extraction
                # Full content indexing would require implementing ExtractedContent integration
                self.logger.logger.info(f"Extracted metadata for {repo_path.name}")
                return True
            else:
                self.logger.log_warning(f"No metadata extracted from {repo_path}")
                return False
                
        except Exception as e:
            self.logger.log_error(e, f"indexing book {repo_path.name}", "search_indexer", "warning")
            return False
    
    def _run_phase(self, phase_name: str, phase_number: int, phase_func, workflow_results: Dict[str, Any]):
        """Execute a workflow phase with timing and error handling."""
        start_time = time.time()
        phase_display_name = phase_name.replace('_', ' ').title()
        
        self.ui.print_phase(phase_number, phase_display_name)
        
        try:
            result = phase_func()
            duration = time.time() - start_time
            
            workflow_results['phases_completed'].append(phase_name)
            self.logger.logger.info(f"✅ Completed: {phase_name} ({duration:.2f}s) - {result}")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, f"phase {phase_name}", "orchestrator", "error")
            workflow_results['errors'] += 1
            
            self.ui.print_error(f"Phase {phase_number} failed: {e}")
            raise
    
    def _display_configuration(self, dry_run: bool, openstax_only: bool, check_updates: bool):
        """Display current configuration settings."""
        config_info = [
            f"Parallel Processing: {'✓' if self.config.enable_parallel_processing else '✗'}",
            f"Search Indexing: {'✓' if self.config.enable_search_indexing else '✗'}",
            f"Max Workers: {self.config.max_workers}",
            f"OpenStax Only: {'✓' if openstax_only else '✗'}",
            f"Check Updates: {'✓' if check_updates else '✗'}",
            f"Dry Run: {'✓' if dry_run else '✗'}"
        ]
        
        self.ui.print_info("Configuration:")
        for info in config_info:
            self.ui.print_info(f"  {info}")
        print()
    
    def _preview_acquisition(self, books: List[Dict[str, Any]]):
        """Preview what books would be acquired in dry run mode."""
        self.ui.print_info("Books that would be acquired:")
        for book in books:
            repo_name = book.get('repo', book.get('name', 'Unknown'))
            self.ui.print_info(f"  - {repo_name}")
    
    def _process_acquisition_results(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Process and summarize acquisition results."""
        total_success = sum(len(result_list) for result_list in results.values())
        total_tasks = total_success + self.parallel_processor.stats.get('tasks_failed', 0)
        
        return {
            'total_tasks': total_tasks,
            'success_count': total_success,
            'error_count': self.parallel_processor.stats.get('tasks_failed', 0),
            'discovery_results': len(results.get('discovery', [])),
            'clone_results': len(results.get('clone', [])),
            'processing_results': len(results.get('processing', []))
        }
    
    def _display_final_statistics(self, workflow_results: Dict[str, Any]):
        """Display final workflow statistics."""
        phases_completed = len(workflow_results['phases_completed'])
        total_duration = workflow_results['duration']
        
        self.ui.print_success(f"Workflow completed successfully!")
        self.ui.print_info(f"Phases completed: {phases_completed}")
        self.ui.print_info(f"Total duration: {total_duration:.2f} seconds")
        
        if workflow_results['errors'] > 0 or workflow_results['warnings'] > 0:
            self.ui.print_warning(f"Errors: {workflow_results['errors']}, "
                                f"Warnings: {workflow_results['warnings']}")
    
    # ===== ALL ORIGINAL METHODS FROM GetOpenBooks CLASS =====
    
    def _collect_repository_paths(self, books_path: Path, repo_paths: List[Path], openstax_only: bool = False, language_filter: Optional[str] = None) -> None:
        """Recursively collect all repository paths for progress tracking."""
        try:
            # Check for language-based structure first
            language_dirs = ['english', 'spanish', 'french', 'polish', 'german']
            has_language_structure = any((books_path / lang).exists() and (books_path / lang).is_dir() 
                                       for lang in language_dirs)
            
            if has_language_structure:
                for lang_dir in books_path.iterdir():
                    if lang_dir.is_dir() and lang_dir.name in language_dirs:
                        if language_filter and lang_dir.name != language_filter:
                            continue
                        self._collect_fields_in_language(lang_dir, repo_paths, openstax_only)
            else:
                # Try OpenAlex structure or fallback to recursive search
                try:
                    from .openalex_disciplines import get_hierarchy
                    hierarchy = get_hierarchy()
                    
                    # Check for field-based organization
                    field_like_dirs = [d for d in books_path.iterdir() 
                                     if d.is_dir() and any(char.isupper() for char in d.name)]
                    
                    if field_like_dirs:
                        for field_dir in field_like_dirs:
                            self._collect_openalex_structure_single_field(field_dir, repo_paths, openstax_only)
                    else:
                        # Fallback to recursive directory search
                        self._collect_directory_repos(books_path, repo_paths, recursive=True, openstax_only=openstax_only)
                        
                except ImportError:
                    # No OpenAlex available, do recursive search
                    self._collect_directory_repos(books_path, repo_paths, recursive=True, openstax_only=openstax_only)
                
        except Exception as e:
            self.logger.log_error(e, f"collecting repository paths from {books_path}", "file_system", "warning")
    
    def _collect_fields_in_language(self, lang_path: Path, repo_paths: List[Path], openstax_only: bool = False) -> None:
        """Collect repositories from fields within a language directory."""
        try:
            from .openalex_disciplines import get_hierarchy
            hierarchy = get_hierarchy()
            
            for field_dir in lang_path.iterdir():
                if not field_dir.is_dir():
                    continue
                
                field_name = field_dir.name
                
                # Check for level subdirectories or repositories directly
                level_dirs = ["HighSchool", "University", "Graduate"]
                has_levels = any((field_dir / level).exists() and (field_dir / level).is_dir() 
                               for level in level_dirs)
                
                if has_levels:
                    for level_dir in field_dir.iterdir():
                        if level_dir.is_dir() and level_dir.name in level_dirs:
                            self._collect_directory_repos(level_dir, repo_paths, recursive=False, openstax_only=openstax_only)
                else:
                    # Repositories directly under field or non-standard structure
                    self._collect_directory_repos(field_dir, repo_paths, recursive=True, openstax_only=openstax_only)
                    
        except Exception as e:
            self.logger.log_error(e, f"collecting fields in {lang_path}", "file_system", "warning")
    
    def _collect_openalex_structure_single_field(self, field_path: Path, repo_paths: List[Path], openstax_only: bool = False) -> None:
        """Collect repositories from a single OpenAlex field directory."""
        try:
            field_name = field_path.name
            
            # Check for level subdirectories
            level_dirs = ["HighSchool", "University", "Graduate"]
            has_levels = any((field_path / level).exists() and (field_path / level).is_dir() 
                           for level in level_dirs)
            
            if has_levels:
                for level_dir in field_path.iterdir():
                    if level_dir.is_dir() and level_dir.name in level_dirs:
                        self._collect_directory_repos(level_dir, repo_paths, recursive=False, openstax_only=openstax_only)
            else:
                # Repository directory directly under field or non-standard structure
                self._collect_directory_repos(field_path, repo_paths, recursive=True, openstax_only=openstax_only)
                
        except Exception as e:
            self.logger.log_error(e, f"collecting from field {field_path}", "file_system", "warning")
    
    def _collect_directory_repos(self, directory: Path, repo_paths: List[Path], recursive: bool = False, openstax_only: bool = False) -> None:
        """Collect repository paths from a directory."""
        try:
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                
                # Check if this is a git repository
                if (item / '.git').exists():
                    # Apply OpenStax filter if requested
                    if openstax_only and not self._is_openstax_repository_path(item):
                        continue
                    repo_paths.append(item)
                elif recursive:
                    # Recurse into subdirectories
                    self._collect_directory_repos(item, repo_paths, recursive=True, openstax_only=openstax_only)
                    
        except Exception as e:
            self.logger.log_error(e, f"collecting repos from {directory}", "file_system", "warning")
    
    def _process_repository(self, repo_path: Path, existing_books: Dict[str, Any], openstax_only: bool = False, language_filter: Optional[str] = None) -> None:
        """Process a single repository and add to existing_books."""
        try:
            if not repo_path.exists() or not repo_path.is_dir():
                return
            
            # Check if it's a git repository
            if not (repo_path / '.git').exists():
                return
            
            # Apply OpenStax filter
            if openstax_only and not self._is_openstax_repository_path(repo_path):
                return
            
            # Get repository information
            repo_name = repo_path.name
            size_mb = self._get_directory_size(repo_path) / (1024 * 1024)
            
            # Detect language
            detected_language = self.language_detector.detect_language(repo_path, repo_name)
            
            # Apply language filter
            if language_filter and detected_language != language_filter:
                return
            
            # Detect subject, level, and organization
            subject = self._detect_subject_from_path(repo_path)
            level = self._detect_level_from_path(repo_path)
            organization = self._detect_organization_from_path(repo_path)
            
            # Create repository entry
            repo_info = {
                "name": repo_name,
                "path": str(repo_path),
                "size_mb": round(size_mb, 2),
                "language": detected_language,
                "level": level,
                "subject": subject,
                "organization": organization
            }
            
            existing_books["git_repos"].append(repo_info)
            existing_books["total_size_mb"] += size_mb
            
            # Update counters
            existing_books["languages"][detected_language] = existing_books["languages"].get(detected_language, 0) + 1
            existing_books["levels"][level] = existing_books["levels"].get(level, 0) + 1
            existing_books["subjects"][subject] = existing_books["subjects"].get(subject, 0) + 1
            existing_books["organizations"][organization] = existing_books["organizations"].get(organization, 0) + 1
            
            # Don't print individual status during cataloging to avoid interfering with progress bar
            # The progress bar already shows which repo is being processed
            
        except Exception as e:
            self.logger.log_error(e, f"processing repository {repo_path}", "repository", "warning")
    
    def _detect_subject_from_path(self, path: Path) -> str:
        """Detect subject from repository path."""
        path_parts = [p.lower() for p in path.parts]
        
        # Common subject keywords in path
        subject_map = {
            'physics': 'Physics', 'biology': 'Biology', 'chemistry': 'Chemistry',
            'mathematics': 'Mathematics', 'math': 'Mathematics', 'calculus': 'Mathematics',
            'algebra': 'Mathematics', 'statistics': 'Statistics', 'psychology': 'Psychology',
            'sociology': 'Sociology', 'business': 'Business', 'economics': 'Economics',
            'computer': 'Computer Science', 'engineering': 'Engineering',
            'history': 'History', 'philosophy': 'Philosophy', 'art': 'Art'
        }
        
        for part in path_parts:
            for keyword, subject in subject_map.items():
                if keyword in part:
                    return subject
        
        # Try extracting from repository name
        return self._extract_subject_from_repo_name(path.name)
    
    def _extract_subject_from_repo_name(self, repo_name: str) -> str:
        """Extract subject from repository name."""
        name_lower = repo_name.lower()
        
        subject_keywords = {
            'physics': ['physics', 'fisica'], 'biology': ['biology', 'biologia'], 
            'chemistry': ['chemistry', 'quimica'], 'mathematics': ['math', 'calculus', 'algebra', 'matematicas'],
            'psychology': ['psychology', 'psicologia'], 'business': ['business', 'negocios'],
            'economics': ['economics', 'economia'], 'sociology': ['sociology', 'sociologia']
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return subject.title()
        
        return "Other"
    
    def _detect_organization_from_path(self, path: Path) -> str:
        """Detect organization from repository path or name."""
        repo_name = path.name.lower()
        
        if 'osbooks-' in repo_name or 'openstax' in str(path).lower():
            return "OpenStax"
        elif 'cnxbook-' in repo_name or 'cnx' in str(path).lower():
            return "CNX"
        else:
            return "Other"
    
    def _detect_level_from_path(self, path: Path) -> str:
        """Detect educational level from repository path."""
        path_str = str(path).lower()
        repo_name = path.name.lower()
        
        # Check path components
        if 'highschool' in path_str or 'high-school' in path_str:
            return "HighSchool"
        elif 'graduate' in path_str:
            return "Graduate"
        elif 'university' in path_str:
            return "University"
        
        # Check repository name for level indicators
        high_school_indicators = ['prealgebra', 'pre-algebra', 'ap-', 'high-school']
        graduate_indicators = ['graduate', 'phd', 'advanced', 'research']
        
        if any(indicator in repo_name for indicator in high_school_indicators):
            return "HighSchool"
        elif any(indicator in repo_name for indicator in graduate_indicators):
            return "Graduate"
        else:
            return "University"
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            self.logger.log_error(e, f"calculating size of {path}", "file_system", "warning")
        return total_size
    
    def _has_content(self, directory: Path) -> bool:
        """Check if directory has substantial content."""
        try:
            file_count = sum(1 for _ in directory.rglob('*') if _.is_file())
            return file_count > 5  # Arbitrary threshold for "substantial" content
        except Exception:
            return False
    
    def _is_openstax_repository_path(self, repo_path: Path) -> bool:
        """Check if repository path indicates an OpenStax repository."""
        repo_name = repo_path.name.lower()
        path_str = str(repo_path).lower()
        
        openstax_indicators = [
            'osbooks-', 'openstax', 'derived-from-osbooks-'
        ]
        
        return any(indicator in repo_name or indicator in path_str 
                  for indicator in openstax_indicators)
    
    def cleanup_non_openstax_repositories(self, dry_run: bool = False) -> Dict[str, Any]:
        """Remove non-OpenStax repositories from the collection."""
        op = self.logger.start_operation("cleanup_non_openstax", dry_run=dry_run)
        
        books_path = Path(self.config.books_path)
        results = {
            "repositories_checked": 0,
            "non_openstax_found": 0,
            "removed": 0,
            "errors": 0,
            "removed_repositories": [],
            "error_repositories": []
        }
        
        if not books_path.exists():
            self.logger.end_operation(False, error="Books directory does not exist")
            return results
        
        # Find all git repositories
        for git_dir in books_path.rglob(".git"):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                results["repositories_checked"] += 1
                
                if not self._is_openstax_repository_path(repo_path):
                    results["non_openstax_found"] += 1
                    
                    if self._remove_repository(repo_path, dry_run):
                        results["removed"] += 1
                        results["removed_repositories"].append(str(repo_path))
                    else:
                        results["errors"] += 1
                        results["error_repositories"].append(str(repo_path))
        
        self.logger.end_operation(True, **results)
        return results
    
    def _remove_repository(self, repo_path: Path, dry_run: bool = True) -> bool:
        """Remove a repository directory."""
        try:
            if dry_run:
                self.ui.print_info(f"[DRY RUN] Would remove: {repo_path}")
                return True
            else:
                import shutil
                shutil.rmtree(repo_path)
                self.ui.print_success(f"Removed: {repo_path}")
                return True
        except Exception as e:
            self.logger.log_error(e, f"removing repository {repo_path}", "file_system", "error")
            return False
    
    def check_for_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate repositories in the collection."""
        op = self.logger.start_operation("check_duplicates")
        
        books_path = Path(self.config.books_path)
        results = {
            "total_repositories": 0,
            "duplicates_found": 0,
            "duplicate_groups": [],
            "unique_repositories": 0
        }
        
        if not books_path.exists():
            self.logger.end_operation(False, error="Books directory does not exist")
            return results
        
        # Collect all repositories with their characteristics
        repositories = {}
        for git_dir in books_path.rglob(".git"):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                repo_name = repo_path.name
                
                # Create a key for duplicate detection
                key = repo_name.lower().replace('-', '').replace('_', '')
                
                if key not in repositories:
                    repositories[key] = []
                repositories[key].append(str(repo_path))
                results["total_repositories"] += 1
        
        # Find duplicates
        for key, paths in repositories.items():
            if len(paths) > 1:
                results["duplicates_found"] += len(paths) - 1
                results["duplicate_groups"].append({
                    "key": key,
                    "paths": paths,
                    "count": len(paths)
                })
            else:
                results["unique_repositories"] += 1
        
        self.logger.end_operation(True, **results)
        return results