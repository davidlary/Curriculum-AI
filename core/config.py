"""
Configuration management for OpenBooks project.

Based on patterns from Pedigree project with enhancements for textbook acquisition.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
from .data_config import get_data_config

logger = logging.getLogger(__name__)


@dataclass
class OpenBooksConfig:
    """Configuration class for OpenBooks project."""
    
    # Project directories - Enhanced structure (auto-detected)
    project_root: str = field(default_factory=lambda: OpenBooksConfig._detect_project_root())
    books_dir: str = "Books"
    cache_dir: str = "cache"
    metadata_dir: str = "metadata"
    docs_dir: str = "docs"
    tests_dir: str = "tests"
    scripts_dir: str = "scripts"
    config_dir: str = "config"
    
    # Enhanced Books subdirectories - New educational structure
    textbooks_dir: str = "Books/Textbooks"
    professional_dir: str = "Books/Professional" 
    specialized_dir: str = "Books/Specialized"
    extracted_content_dir: str = "Books/Specialized/Archives/extracted"
    search_index_dir: str = "metadata/search_index"
    
    # Educational levels (data-driven)
    educational_levels: List[str] = field(default_factory=lambda: 
        get_data_config().get_educational_levels())
    
    # Academic disciplines - Legacy list (deprecated in favor of OpenAlex, data-driven)
    disciplines: List[str] = field(default_factory=lambda: 
        get_data_config().get_legacy_disciplines())
    
    # OpenAlex discipline configuration
    use_openalex_disciplines: bool = True  # Use OpenAlex hierarchy for organization
    openalex_hierarchy_file: Optional[str] = None  # Optional external hierarchy file
    enable_openalex_api: bool = False  # Enable real-time OpenAlex API classification
    openalex_api_email: Optional[str] = None  # Required for OpenAlex API access
    
    # Directory structure mode
    directory_structure_mode: str = "openalex"  # "openalex" or "legacy"
    
    # Source priorities (1=highest priority)
    source_priority: Dict[str, int] = field(default_factory=lambda: {
        "existing_collection": 1,
        "git_repositories": 2,
        "structured_formats": 3,
        "pdf_downloads": 4
    })
    
    # OpenStax configuration
    openstax_base_url: str = "https://openstax.org"
    github_api_base_url: str = "https://api.github.com"  # For API calls
    github_base_url: str = "https://github.com"  # For git clone operations
    github_ssh_base_url: str = "git@github.com"  # For SSH git clone operations
    use_ssh_for_git: bool = True  # Use SSH URLs instead of HTTPS for git operations
    cnx_user_books_org: str = "cnx-user-books"
    
    # Rate limiting and delays
    request_delay_seconds: float = 2.0
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Processing configuration - Optimized for 24-core machine
    batch_size: int = 20
    max_workers: int = 20  # Use 20 cores by default (leaving 4 for system)
    max_discovery_workers: int = 8  # Parallel discovery workers
    max_clone_workers: int = 6  # Parallel git clone workers
    max_processing_workers: int = 12  # Content processing workers
    enable_pdf_processing: bool = True
    enable_git_cloning: bool = True
    enable_parallel_processing: bool = True
    enable_text_extraction: bool = True
    enable_search_indexing: bool = True
    
    # Content validation
    min_book_size_mb: float = 1.0
    max_book_size_mb: float = 500.0
    
    # Git configuration
    git_lfs_enabled: bool = True
    clone_depth: Optional[int] = None  # None for full clone
    
    @staticmethod
    def _detect_project_root() -> str:
        """Auto-detect project root directory."""
        # Start from this file's location
        current_file = Path(__file__).resolve()
        
        # Look for project indicators going up the directory tree
        indicators = ['GetOpenBooks.py', 'config', 'Books', '.git']
        
        for parent in [current_file.parent.parent, current_file.parent]:
            if any((parent / indicator).exists() for indicator in indicators):
                return str(parent)
        
        # Fallback to current working directory
        cwd = Path.cwd()
        logger.warning(f"Could not auto-detect project root, using: {cwd}")
        return str(cwd)

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Convert relative paths to absolute
        if not os.path.isabs(self.project_root):
            self.project_root = os.path.abspath(self.project_root)
            
        # Validate directories exist or can be created
        for dir_attr in ['books_dir', 'cache_dir', 'metadata_dir']:
            dir_path = self.get_absolute_path(getattr(self, dir_attr))
            os.makedirs(dir_path, exist_ok=True)
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path within project."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.project_root, relative_path)
    
    @property
    def books_path(self) -> str:
        """Get absolute path to books directory."""
        return self.get_absolute_path(self.books_dir)
    
    @property
    def cache_path(self) -> str:
        """Get absolute path to cache directory."""
        return self.get_absolute_path(self.cache_dir)
    
    @property
    def metadata_path(self) -> str:
        """Get absolute path to metadata directory."""
        return self.get_absolute_path(self.metadata_dir)
    
    @property
    def textbooks_path(self) -> str:
        """Get absolute path to textbooks directory."""
        return self.get_absolute_path(self.textbooks_dir)
    
    def get_discipline_path(self, discipline: str, educational_level: str) -> str:
        """Get path for a specific discipline and educational level."""
        if self.use_openalex_disciplines and self.directory_structure_mode == "openalex":
            # Use OpenAlex hierarchy for path generation
            try:
                from .openalex_disciplines import get_hierarchy
                hierarchy = get_hierarchy()
                field = hierarchy.classify_subject(discipline)
                if field:
                    return os.path.join(self.books_path, hierarchy.get_directory_path(field.id, educational_level))
                else:
                    # Fallback to legacy structure
                    return os.path.join(self.textbooks_path, discipline, educational_level)
            except ImportError:
                logger.warning("OpenAlex disciplines module not available, using legacy structure")
                return os.path.join(self.textbooks_path, discipline, educational_level)
        else:
            # Legacy structure
            return os.path.join(self.textbooks_path, discipline, educational_level)
    
    def get_all_textbook_paths(self) -> List[str]:
        """Get all textbook directory paths."""
        paths = []
        
        if self.use_openalex_disciplines and self.directory_structure_mode == "openalex":
            # Get paths based on OpenAlex hierarchy
            try:
                from .openalex_disciplines import get_hierarchy
                hierarchy = get_hierarchy()
                for domain in hierarchy.DOMAINS.values():
                    for field in hierarchy.get_fields_by_domain(domain.id):
                        for level in self.educational_levels:
                            path_parts = hierarchy.get_directory_path(field.id, level)
                            path = os.path.join(self.books_path, path_parts)
                            if os.path.exists(path):
                                paths.append(path)
            except ImportError:
                logger.warning("OpenAlex disciplines module not available, using legacy structure")
                # Fall back to legacy method
                for discipline in self.disciplines:
                    for level in self.educational_levels:
                        path = self.get_discipline_path(discipline, level)
                        if os.path.exists(path):
                            paths.append(path)
        else:
            # Legacy structure
            for discipline in self.disciplines:
                for level in self.educational_levels:
                    path = self.get_discipline_path(discipline, level)
                    if os.path.exists(path):
                        paths.append(path)
        return paths
    
    def get_openalex_config(self) -> Dict[str, Any]:
        """Get OpenAlex-specific configuration."""
        return {
            "use_openalex_disciplines": self.use_openalex_disciplines,
            "directory_structure_mode": self.directory_structure_mode,
            "openalex_hierarchy_file": self.openalex_hierarchy_file,
            "enable_openalex_api": self.enable_openalex_api,
            "openalex_api_email": self.openalex_api_email
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_data = self.__dict__.copy()
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'OpenBooksConfig':
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                logger.warning(f"Empty configuration file {config_path}, using defaults")
                return cls()
            
            return cls(**config_data)
        
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def get_default_config_path(cls) -> str:
        """Get default configuration file path."""
        project_root = cls._detect_project_root()
        return os.path.join(project_root, "config", "openbooks_config.yaml")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate paths
        if not os.path.exists(self.project_root):
            issues.append(f"Project root does not exist: {self.project_root}")
        
        # Validate rate limiting
        if self.request_delay_seconds < 0.5:
            issues.append("Request delay should be at least 0.5 seconds for respectful scraping")
        
        # Validate worker limits - be more permissive for high-end systems
        cpu_count = os.cpu_count() or 4
        if self.max_workers > cpu_count:
            issues.append(f"Max workers ({self.max_workers}) exceeds CPU count ({cpu_count})")
        elif self.max_workers > cpu_count - 2:
            issues.append(f"Consider leaving 2+ cores for system (CPU count: {cpu_count}, workers: {self.max_workers})")
        
        # Validate size limits
        if self.min_book_size_mb >= self.max_book_size_mb:
            issues.append("min_book_size_mb must be less than max_book_size_mb")
        
        return issues
    
    def set_git_mode(self, use_ssh: bool = True) -> None:
        """
        Set the Git clone mode to use SSH or HTTPS.
        
        Args:
            use_ssh: If True, use SSH URLs (git@github.com:org/repo.git)
                    If False, use HTTPS URLs (https://github.com/org/repo.git)
        """
        self.use_ssh_for_git = use_ssh
        logger.info(f"Git clone mode set to: {'SSH' if use_ssh else 'HTTPS'}")
        
        if use_ssh:
            logger.info("SSH mode requires SSH key authentication with GitHub")
        else:
            logger.warning("HTTPS mode may require GitHub token or cause authentication failures")
    
    def get_known_repositories(self) -> List[Dict[str, str]]:
        """Get list of known OpenStax repositories (fallback only - prefer auto-discovery)."""
        # These are kept as fallback seeds, but the system now uses auto-discovery
        return [
            {
                "name": "University Physics Volume 1",
                "repo": "cnxbook-university-physics-volume-1",
                "org": "cnx-user-books",
                "subject": "Physics"
            },
            {
                "name": "University Physics Volume 2", 
                "repo": "cnxbook-university-physics-volume-2",
                "org": "cnx-user-books",
                "subject": "Physics"
            },
            {
                "name": "University Physics Volume 3",
                "repo": "cnxbook-university-physics-volume-3", 
                "org": "cnx-user-books",
                "subject": "Physics"
            },
            {
                "name": "Anatomy & Physiology",
                "repo": "osbooks-anatomy-physiology",
                "org": "cnx-user-books",
                "subject": "Biology"
            },
            {
                "name": "Biology Bundle",
                "repo": "osbooks-biology-bundle",
                "org": "cnx-user-books", 
                "subject": "Biology"
            },
            {
                "name": "Introduction to Sociology",
                "repo": "osbooks-introduction-sociology",
                "org": "cnx-user-books",
                "subject": "Sociology"
            },
            {
                "name": "College Physics Bundle", 
                "repo": "osbooks-college-physics-bundle",
                "org": "cnx-user-books",
                "subject": "Physics"
            },
            {
                "name": "Principles of Economics Bundle",
                "repo": "osbooks-principles-economics-bundle", 
                "org": "cnx-user-books",
                "subject": "Economics"
            },
            {
                "name": "Prealgebra Bundle",
                "repo": "osbooks-prealgebra-bundle",
                "org": "cnx-user-books",
                "subject": "Mathematics"
            },
            {
                "name": "Introductory Statistics Bundle",
                "repo": "osbooks-introductory-statistics-bundle",
                "org": "cnx-user-books", 
                "subject": "Statistics"
            },
            {
                "name": "Chemistry Bundle",
                "repo": "osbooks-chemistry-bundle",
                "org": "cnx-user-books",
                "subject": "Chemistry"
            },
            {
                "name": "Microbiology",
                "repo": "osbooks-microbiology",
                "org": "cnx-user-books",
                "subject": "Biology"
            },
            {
                "name": "Organic Chemistry",
                "repo": "osbooks-organic-chemistry",
                "org": "cnx-user-books",
                "subject": "Chemistry"
            },
            {
                "name": "Psychology",
                "repo": "osbooks-psychology",
                "org": "cnx-user-books",
                "subject": "Psychology"
            }
        ]