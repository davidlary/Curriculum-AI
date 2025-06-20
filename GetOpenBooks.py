#!/usr/bin/env python3
"""
GetOpenBooks.py - Main script for acquiring OpenStax and other open textbooks

This script provides a comprehensive system for discovering, acquiring, and managing
open textbooks with emphasis on OpenStax collections. Default behavior runs with
high-performance settings optimized for OpenStax repository acquisition.

Usage:
    python GetOpenBooks.py [options]
    
Default Behavior (equivalent to --workers 20 --openstax-only --check-updates --process-pdfs):
    - Uses 20 parallel workers for optimal performance
    - Downloads both Git repositories AND PDFs
    - Restricts to OpenStax books only
    - Checks existing repositories for updates
    - Processes PDFs using Claude API for text extraction (requires ANTHROPIC_API_KEY)
    
Override Options:
    --git-only              Download only Git repositories (no PDFs)
    --no-openstax-only      Include non-OpenStax repositories  
    --no-check-updates      Skip checking for repository updates
    --no-process-pdfs       Skip PDF processing with Claude API
    --force-pdf-reprocess   Force reprocessing of PDFs (ignore cache)
    --workers N             Change number of parallel workers (default: 20)
    --verbose              Enable detailed progress display
    
Examples:
    python GetOpenBooks.py                                    # Default: OpenStax Git repos, PDFs, and PDF processing
    python GetOpenBooks.py --verbose                          # Same with detailed progress display
    python GetOpenBooks.py --git-only                         # Only Git repositories (no PDFs or processing)
    python GetOpenBooks.py --no-process-pdfs                  # Download PDFs but skip Claude API processing
    python GetOpenBooks.py --force-pdf-reprocess --verbose    # Reprocess all PDFs with detailed output
    python GetOpenBooks.py --no-openstax-only --subjects Physics,Biology  # Include all Physics/Biology books
    python GetOpenBooks.py --dry-run --verbose                # Preview what would be done

PDF Processing:
    Requires ANTHROPIC_API_KEY environment variable for Claude API access.
    Processes PDFs to extract high-quality text content for search and curriculum generation.
    Uses intelligent caching to avoid reprocessing the same content.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import OpenBooksConfig
from core.terminal_ui import TerminalUI
from core.orchestrator import OpenBooksOrchestrator
from core.enhanced_logging import OperationLogger
from core.data_config import get_data_config

# PDF processing imports (optional - only used if API key available)
try:
    from core.pdf_integration import PDFContentManager, check_pdf_processing_status
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    # Get data-driven configuration
    data_config = get_data_config()
    
    parser = argparse.ArgumentParser(
        description="Acquire and manage OpenStax and other open textbooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    # Workflow control
    parser.add_argument("--update-existing", action="store_true", 
                       help="Update already acquired books")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without executing")
    
    # Content filtering
    parser.add_argument("--subjects", type=str,
                       help="Comma-separated list of subjects to focus on")
    # Get language choices from data configuration (data-driven)
    supported_languages = data_config.get_supported_languages()
    default_language = data_config.get_default_language()
    
    parser.add_argument("--language", type=str, default=default_language,
                       choices=supported_languages,
                       help=f"Language filter for discovery (default: {default_language})")
    
    # Get application defaults from data configuration (data-driven)
    app_defaults = data_config.get_application_defaults()
    
    # Format preferences with data-driven defaults
    parser.add_argument("--git-only", action="store_true", 
                       default=app_defaults.get('git_only', False),
                       help="Focus only on git repository books (no PDFs)")
    parser.add_argument("--no-git-only", action="store_true",
                       help="Include PDFs and other formats (legacy compatibility)")
    
    # Repository filtering with data-driven defaults
    parser.add_argument("--openstax-only", action="store_true", 
                       default=app_defaults.get('openstax_only', True),
                       help=f"Restrict discovery to OpenStax books only (default: {app_defaults.get('openstax_only', True)})")
    parser.add_argument("--no-openstax-only", action="store_true",
                       help="Include non-OpenStax repositories (overrides --openstax-only default)")
    
    # Update checking with data-driven defaults
    parser.add_argument("--check-updates", action="store_true", 
                       default=app_defaults.get('check_updates', True),
                       help=f"Check existing repositories for updates and sync if needed (default: {app_defaults.get('check_updates', True)})")
    parser.add_argument("--no-check-updates", action="store_true",
                       help="Skip checking for repository updates (overrides --check-updates default)")
    
    # Performance settings with data-driven defaults
    parser.add_argument("--workers", type=int, 
                       default=app_defaults.get('workers', 20),
                       help=f"Number of parallel workers (default: {app_defaults.get('workers', 20)})")
    parser.add_argument("--parallel", action="store_true", 
                       default=app_defaults.get('parallel', True),
                       help=f"Enable parallel processing (default: {app_defaults.get('parallel', True)})")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    
    # Search indexing with data-driven defaults
    parser.add_argument("--index", action="store_true", 
                       default=app_defaults.get('index', True),
                       help=f"Build search index (default: {app_defaults.get('index', True)})")
    parser.add_argument("--no-index", action="store_true",
                       help="Skip search index building")
    
    # PDF processing with data-driven defaults
    pdf_defaults = data_config.get_pdf_processing_defaults()
    parser.add_argument("--process-pdfs", action="store_true", 
                       default=pdf_defaults.get('auto_process', True),
                       help=f"Process PDFs using Claude API for text extraction (default: {pdf_defaults.get('auto_process', True)})")
    parser.add_argument("--no-process-pdfs", action="store_true",
                       help="Skip PDF processing (overrides --process-pdfs default)")
    parser.add_argument("--force-pdf-reprocess", action="store_true",
                       help="Force reprocessing of PDFs (ignore cache)")
    
    # Output control
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    # Maintenance operations
    parser.add_argument("--check-duplicates", action="store_true",
                       help="Check for and report duplicate repositories")
    parser.add_argument("--cleanup-non-openstax", action="store_true",
                       help="Remove non-OpenStax repositories from collection")
    
    return parser


def configure_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('openbooks.log'),
            logging.StreamHandler() if verbose else logging.NullHandler()
        ]
    )


def resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve conflicting arguments and apply defaults."""
    # Resolve git-only setting (now defaults to False, meaning PDFs are included by default)
    # no_git_only is now for legacy compatibility
    if args.no_git_only:
        args.git_only = False
    
    # Resolve openstax-only setting
    if args.no_openstax_only:
        args.openstax_only = False
    
    # Resolve check-updates setting
    if args.no_check_updates:
        args.check_updates = False
    
    # Resolve parallel processing setting
    if args.no_parallel:
        args.parallel = False
    
    # Resolve search indexing setting
    if args.no_index:
        args.index = False
    
    # Resolve PDF processing setting
    if args.no_process_pdfs:
        args.process_pdfs = False
    
    # Parse subjects list
    if args.subjects:
        args.subjects = [s.strip() for s in args.subjects.split(',')]
    else:
        args.subjects = None
    
    return args


def main():
    """Main entry point for the script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    args = resolve_arguments(args)
    
    # Configure logging
    configure_logging(args.verbose)
    
    try:
        # Initialize configuration
        config = OpenBooksConfig(
            max_workers=args.workers,
            enable_parallel_processing=args.parallel,
            enable_search_indexing=args.index
        )
        
        # Initialize UI and orchestrator
        ui = TerminalUI()
        orchestrator = OpenBooksOrchestrator(config, ui)
        
        # Handle maintenance operations
        if args.check_duplicates:
            ui.print_info("Checking for duplicate repositories...")
            results = orchestrator.check_for_duplicates()
            if results['duplicates_found'] > 0:
                ui.print_warning(f"Found {results['duplicates_found']} duplicate repositories")
                for group in results['duplicate_groups']:
                    ui.print_info(f"Duplicate group: {group['key']} ({group['count']} copies)")
                    for path in group['paths']:
                        ui.print_info(f"  - {path}")
            else:
                ui.print_success("No duplicate repositories found")
            return
        
        if args.cleanup_non_openstax:
            ui.print_info("Cleaning up non-OpenStax repositories...")
            results = orchestrator.cleanup_non_openstax_repositories(dry_run=False)
            ui.print_success(f"Cleanup completed: {results['removed']} repositories removed, {results['errors']} errors")
            return
        
        # Run the main workflow
        results = orchestrator.run_complete_workflow(
            update_existing=args.update_existing,
            dry_run=args.dry_run,
            subjects=args.subjects,
            language_filter=args.language if args.language != "all" else None,
            openstax_only=args.openstax_only,
            check_updates=args.check_updates,
            git_only=args.git_only
        )
        
        # PDF Processing Phase (after main workflow)
        if args.process_pdfs and not args.dry_run and not args.git_only:
            ui.print_info("📚 Starting PDF processing with Claude API...")
            
            if not PDF_PROCESSING_AVAILABLE:
                ui.print_warning("PDF processing not available - missing dependencies or API key")
                ui.print_info("To enable PDF processing:")
                ui.print_info("  1. pip install PyPDF2 anthropic rich")
                ui.print_info("  2. export ANTHROPIC_API_KEY='your-api-key-here'")
            else:
                try:
                    # Initialize PDF content manager
                    pdf_manager = PDFContentManager()
                    
                    if not pdf_manager.is_pdf_processing_available():
                        ui.print_warning("PDF processing not available - missing ANTHROPIC_API_KEY")
                        ui.print_info("Set environment variable: export ANTHROPIC_API_KEY='your-api-key-here'")
                    else:
                        ui.print_info("🔍 Scanning for PDF files to process...")
                        
                        # Get Books directory path
                        books_path = Path(config.books_path)
                        
                        # Check current PDF processing status
                        status = check_pdf_processing_status(books_path)
                        
                        ui.print_info(f"📊 PDF Status: {status['total_pdfs']} total, "
                                    f"{status['processed_pdfs']} processed, "
                                    f"{status['unprocessed_pdfs']} unprocessed")
                        
                        if status['unprocessed_pdfs'] > 0 or args.force_pdf_reprocess:
                            # Determine which languages to process
                            languages_to_process = None
                            if args.language and args.language != "all":
                                languages_to_process = [args.language]
                            
                            ui.print_info(f"🤖 Processing PDFs using Claude API...")
                            if args.force_pdf_reprocess:
                                ui.print_info("🔄 Force reprocessing enabled - ignoring cache")
                            
                            # Process PDFs in batches by language
                            pdf_results = pdf_manager.batch_process_books(
                                books_path, 
                                languages=languages_to_process
                            )
                            
                            ui.print_success(f"✅ PDF Processing Complete: "
                                           f"{pdf_results['processed']} processed, "
                                           f"{pdf_results['failed']} failed, "
                                           f"{pdf_results['skipped']} skipped")
                            
                            if pdf_results['failed'] > 0:
                                ui.print_warning(f"⚠️  {pdf_results['failed']} PDFs failed to process - check logs")
                        else:
                            ui.print_success("✅ All PDFs already processed (use --force-pdf-reprocess to reprocess)")
                            
                except Exception as e:
                    ui.print_error(f"❌ PDF processing failed: {e}")
                    logging.error(f"PDF processing error: {e}", exc_info=True)
        elif args.process_pdfs and args.git_only:
            ui.print_info("📝 Skipping PDF processing (--git-only mode)")
        elif args.process_pdfs and args.dry_run:
            ui.print_info("📝 PDF processing would be performed (skipped in dry-run mode)")
        
        # Generate BookList after successful completion
        if not args.dry_run:
            ui.print_info("📋 Generating BookList.tsv and BookList.json...")
            try:
                from core.book_list_generator import generate_book_list
                books_path = Path(config.books_path)
                generate_book_list(books_path)
                ui.print_success("✅ BookList files generated successfully")
            except Exception as e:
                ui.print_warning(f"⚠️ Failed to generate BookList: {e}")
                logging.warning(f"BookList generation failed: {e}")
        
        # Display session summary
        logger = OperationLogger('main')
        counts = logger.get_session_counts()
        error_count, warning_count = counts['errors'], counts['warnings']
        
        if error_count > 0 or warning_count > 0:
            ui.print_error_warning_summary(error_count, warning_count)
        else:
            ui.print_success("🎉 Process completed successfully with no errors or warnings!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("📝 Check openbooks.log for detailed information")
        sys.exit(1)


if __name__ == "__main__":
    main()