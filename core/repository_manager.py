"""
Repository management module for cloning and updating Git repositories.

This module handles Git operations for OpenStax textbook repositories
with proper error handling and Git LFS support.
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil

from .config import OpenBooksConfig
from .repository_tracker import RepositoryTracker

logger = logging.getLogger(__name__)


class RepositoryManager:
    """Manages Git repository operations for textbook acquisition."""
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize with configuration."""
        self.config = config
        self.books_path = Path(config.books_path)
        self.books_path.mkdir(parents=True, exist_ok=True)
        self.tracker = RepositoryTracker(config)
    
    def clone_repository(self, book_info: Dict[str, Any], openstax_only: bool = True) -> bool:
        """
        Clone a Git repository for a textbook.
        
        Args:
            book_info: Dictionary containing repository information
            
        Returns:
            True if cloning successful, False otherwise
        """
        repo_name = book_info.get('repo')
        if not repo_name:
            logger.error("No repository name provided")
            return False
        
        org = book_info.get('org', self.config.cnx_user_books_org)
        clone_url = book_info.get('clone_url')
        
        if not clone_url:
            # Construct clone URL - use SSH if configured, otherwise HTTPS
            if self.config.use_ssh_for_git:
                clone_url = f"{self.config.github_ssh_base_url}:{org}/{repo_name}.git"
            else:
                clone_url = f"{self.config.github_base_url}/{org}/{repo_name}.git"
        else:
            # Convert existing clone_url to the preferred format
            clone_url = self._convert_clone_url_format(clone_url)
        
        # Determine target directory based on repository type and subject
        target_path = self._get_target_path(book_info)
        
        # Validate repository content before cloning
        if not self._validate_repository_content(book_info, target_path, openstax_only=openstax_only):
            logger.error(f"Repository content validation failed for {repo_name}")
            return False
        
        # Check if repository is already tracked (prevents duplicates)
        if self.tracker.is_repository_tracked(clone_url, str(target_path)):
            logger.info(f"Repository already tracked, skipping clone: {repo_name}")
            return True
        
        # Check if repository already exists on disk
        if target_path.exists():
            logger.info(f"Repository already exists: {target_path}")
            # Add to tracker if not already tracked
            if not self.tracker.is_repository_tracked(clone_url, str(target_path)):
                size_mb = self._get_directory_size(target_path) / (1024 * 1024)
                self.tracker.add_repository(book_info, str(target_path), size_mb)
            return self.update_repository(str(target_path))
        
        logger.info(f"Cloning repository: {clone_url}")
        
        try:
            # Prepare git clone command with quiet flags
            cmd = ['git', 'clone', '--quiet', '--no-progress']
            
            # Add depth option if specified
            if self.config.clone_depth:
                cmd.extend(['--depth', str(self.config.clone_depth)])
            
            # Add LFS support if enabled
            if self.config.git_lfs_enabled:
                # Check if git-lfs is available
                if self._check_git_lfs():
                    cmd.append('--filter=blob:none')
                else:
                    logger.warning("Git LFS not available, cloning without LFS support")
            
            cmd.extend([clone_url, str(target_path)])
            
            # Set environment to prevent interactive prompts
            env = self._get_git_env()
            
            # Execute clone command with comprehensive output capture
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout for large repositories
                env=env,
                stdin=subprocess.DEVNULL  # Prevent stdin input requests
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully cloned: {repo_name}")
                
                # Initialize LFS if enabled and available
                if self.config.git_lfs_enabled and self._check_git_lfs():
                    self._setup_git_lfs(target_path)
                
                # Add to repository tracker
                size_mb = self._get_directory_size(target_path) / (1024 * 1024)
                self.tracker.add_repository(book_info, str(target_path), size_mb)
                
                # Log repository information
                self._log_repository_info(target_path, book_info)
                
                return True
            else:
                logger.error(f"Failed to clone {repo_name}: {result.stderr}")
                
                # Mark as failed in tracker
                self.tracker.update_repository_status(clone_url, str(target_path), 'failed')
                
                # Clean up partial clone
                if target_path.exists():
                    shutil.rmtree(target_path)
                
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Clone timeout for {repo_name}")
            if target_path.exists():
                shutil.rmtree(target_path)
            return False
            
        except Exception as e:
            logger.error(f"Error cloning {repo_name}: {e}")
            if target_path.exists():
                shutil.rmtree(target_path)
            return False
    
    def update_repository(self, repo_path: str) -> bool:
        """
        Update an existing Git repository.
        
        Args:
            repo_path: Path to the repository directory
            
        Returns:
            True if update successful, False otherwise
        """
        repo_path = Path(repo_path)
        
        if not repo_path.exists() or not (repo_path / '.git').exists():
            logger.error(f"Not a git repository: {repo_path}")
            return False
        
        logger.info(f"Updating repository: {repo_path.name}")
        
        try:
            # Change to repository directory
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            
            # Get environment for all git operations
            env = self._get_git_env()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            
            if result.stdout.strip():
                logger.warning(f"Repository has uncommitted changes: {repo_path.name}")
                # For now, skip update if there are uncommitted changes
                return False
            
            # Fetch latest changes
            result = subprocess.run(
                ['git', 'fetch', 'origin'],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to fetch: {result.stderr}")
                return False
            
            # Check if updates are available
            result = subprocess.run(
                ['git', 'rev-list', 'HEAD..origin/main', '--count'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode != 0:
                # Try master branch
                result = subprocess.run(
                    ['git', 'rev-list', 'HEAD..origin/master', '--count'],
                    capture_output=True,
                    text=True,
                    env=env,
                    stdin=subprocess.DEVNULL
                )
            
            if result.returncode == 0:
                update_count = int(result.stdout.strip())
                if update_count == 0:
                    logger.info(f"Repository is up to date: {repo_path.name}")
                    return True
                else:
                    logger.info(f"Found {update_count} updates for {repo_path.name}")
            
            # Pull latest changes
            result = subprocess.run(
                ['git', 'pull', 'origin'],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully updated: {repo_path.name}")
                
                # Update LFS files if enabled
                if self.config.git_lfs_enabled and self._check_git_lfs():
                    self._update_git_lfs(repo_path)
                
                return True
            else:
                logger.error(f"Failed to pull updates: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Update timeout for {repo_path.name}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating {repo_path.name}: {e}")
            return False
            
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def _get_git_env(self) -> Dict[str, str]:
        """Get environment variables for git operations that prevent interactive prompts."""
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'  # Disable terminal prompts
        env['GIT_ASKPASS'] = 'echo'       # Return empty string for password prompts
        env['GIT_QUIET'] = '1'            # Suppress git output
        env['GIT_PROGRESS'] = '0'         # Disable progress output  
        env['GIT_LFS_SKIP_SMUDGE'] = '1'  # Skip LFS file downloads during clone
        return env
    
    def _check_git_lfs(self) -> bool:
        """Check if Git LFS is available."""
        try:
            result = subprocess.run(
                ['git', 'lfs', 'version'],
                capture_output=True,
                text=True,
                env=self._get_git_env(),
                stdin=subprocess.DEVNULL
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _setup_git_lfs(self, repo_path: Path) -> None:
        """Setup Git LFS for a repository."""
        try:
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            env = self._get_git_env()
            
            # Initialize LFS (capture output to prevent terminal spam)
            result = subprocess.run(
                ['git', 'lfs', 'install'], 
                capture_output=True, 
                text=True, 
                env=env, 
                stdin=subprocess.DEVNULL
            )
            if result.returncode != 0:
                logger.warning(f"Git LFS install failed for {repo_path.name}: {result.stderr}")
                return
            
            # Pull LFS files (capture output to prevent terminal spam)
            result = subprocess.run(
                ['git', 'lfs', 'pull'], 
                capture_output=True, 
                text=True, 
                env=env, 
                stdin=subprocess.DEVNULL
            )
            if result.returncode != 0:
                logger.warning(f"Git LFS pull failed for {repo_path.name}: {result.stderr}")
                return
            
            logger.info(f"Git LFS setup complete for {repo_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git LFS setup failed for {repo_path.name}: {e}")
        except Exception as e:
            logger.warning(f"Error setting up Git LFS: {e}")
        finally:
            os.chdir(original_cwd)
    
    def _update_git_lfs(self, repo_path: Path) -> None:
        """Update Git LFS files for a repository."""
        try:
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            env = self._get_git_env()
            
            # Pull latest LFS files
            subprocess.run(['git', 'lfs', 'pull'], check=True, env=env, stdin=subprocess.DEVNULL)
            
            logger.debug(f"Git LFS files updated for {repo_path.name}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git LFS update failed for {repo_path.name}: {e}")
        except Exception as e:
            logger.warning(f"Error updating Git LFS: {e}")
        finally:
            os.chdir(original_cwd)
    
    def _log_repository_info(self, repo_path: Path, book_info: Dict[str, Any]) -> None:
        """Log information about the cloned repository."""
        try:
            # Get repository size
            size_mb = self._get_directory_size(repo_path) / (1024 * 1024)
            
            # Count files
            file_count = sum(1 for _ in repo_path.rglob('*') if _.is_file())
            
            logger.info(f"Repository info for {repo_path.name}:")
            logger.info(f"  Size: {size_mb:.1f} MB")
            logger.info(f"  Files: {file_count}")
            logger.info(f"  Subject: {book_info.get('subject', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"Error logging repository info: {e}")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size
    
    def get_repository_status(self, repo_path: str) -> Dict[str, Any]:
        """
        Get status information for a repository.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary with repository status information
        """
        repo_path = Path(repo_path)
        status = {
            'exists': False,
            'is_git_repo': False,
            'has_changes': False,
            'behind_remote': False,
            'ahead_remote': False,
            'last_commit': None,
            'size_mb': 0,
            'file_count': 0
        }
        
        if not repo_path.exists():
            return status
        
        status['exists'] = True
        
        if not (repo_path / '.git').exists():
            return status
        
        status['is_git_repo'] = True
        
        try:
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            env = self._get_git_env()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            status['has_changes'] = bool(result.stdout.strip())
            
            # Get last commit info
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%H|%s|%ai'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            if result.returncode == 0 and result.stdout.strip():
                commit_info = result.stdout.strip().split('|')
                if len(commit_info) >= 3:
                    status['last_commit'] = {
                        'hash': commit_info[0][:8],
                        'message': commit_info[1],
                        'date': commit_info[2]
                    }
            
            # Check if behind/ahead of remote
            subprocess.run(['git', 'fetch'], capture_output=True, env=env, stdin=subprocess.DEVNULL)
            
            # Check behind
            result = subprocess.run(
                ['git', 'rev-list', 'HEAD..origin/main', '--count'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            if result.returncode == 0:
                status['behind_remote'] = int(result.stdout.strip()) > 0
            
            # Check ahead
            result = subprocess.run(
                ['git', 'rev-list', 'origin/main..HEAD', '--count'],
                capture_output=True,
                text=True,
                env=env,
                stdin=subprocess.DEVNULL
            )
            if result.returncode == 0:
                status['ahead_remote'] = int(result.stdout.strip()) > 0
            
        except Exception as e:
            logger.warning(f"Error getting repository status: {e}")
        finally:
            os.chdir(original_cwd)
        
        # Get size and file count
        status['size_mb'] = self._get_directory_size(repo_path) / (1024 * 1024)
        status['file_count'] = sum(1 for _ in repo_path.rglob('*') if _.is_file())
        
        return status
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """
        List all Git repositories in the books directory.
        
        Returns:
            List of repository information dictionaries
        """
        repositories = []
        
        for item in self.books_path.iterdir():
            if item.is_dir() and (item / '.git').exists():
                status = self.get_repository_status(str(item))
                repo_info = {
                    'name': item.name,
                    'path': str(item),
                    **status
                }
                repositories.append(repo_info)
        
        return repositories
    
    def _get_target_path(self, book_info: Dict[str, Any]) -> Path:
        """
        Determine the target path for a repository using Language/Discipline/Level hierarchy.
        Uses structure: Books/Language/Discipline/Level/Repository
        
        Args:
            book_info: Dictionary containing book information
            
        Returns:
            Path object for the target directory
        """
        repo_name = book_info.get('repo', book_info.get('name', 'unknown'))
        
        # Check if we should use OpenAlex structure
        if self.config.use_openalex_disciplines and self.config.directory_structure_mode == "openalex":
            return self._get_openalex_target_path(book_info)
        else:
            return self._get_legacy_target_path(book_info)
    
    def _get_openalex_target_path(self, book_info: Dict[str, Any]) -> Path:
        """
        Get target path using OpenAlex Level-0 concept hierarchy.
        
        Uses structure: Books/Language/Concept/Level/Repository
        
        Args:
            book_info: Dictionary containing book information
            
        Returns:
            Path object for the target directory
        """
        repo_name = book_info.get('repo', book_info.get('name', 'unknown'))
        educational_level = self._determine_educational_level(book_info)
        
        # Use existing dynamic language detection functionality
        detected_language = self._get_repository_language(book_info, repo_name)
        
        try:
            from .openalex_disciplines import get_hierarchy
            hierarchy = get_hierarchy()
            
            # Determine subject for classification
            subject = book_info.get('subject', '')
            if not subject:
                # Extract subject from repository name if not provided
                subject = self._extract_subject_from_repo_name(repo_name)
            
            # Get Level-0 concept classification
            concept_name = hierarchy.classify_subject(subject)
            if concept_name:
                # Create Language/Concept/Level structure
                target_path = self.books_path / detected_language / concept_name / educational_level / repo_name
            else:
                # Uncategorized fallback with language
                target_path = self.books_path / detected_language / "Uncategorized" / educational_level / repo_name
            
            # Create directory structure if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            return target_path
            
        except ImportError:
            logger.warning("OpenAlex disciplines module not available, using legacy structure")
            return self._get_legacy_target_path(book_info)
        except Exception as e:
            logger.warning(f"Error using OpenAlex classification, falling back to legacy: {e}")
            return self._get_legacy_target_path(book_info)
    
    def _get_legacy_target_path(self, book_info: Dict[str, Any]) -> Path:
        """
        Get target path using legacy discipline structure with Language/Discipline/Level format.
        
        Args:
            book_info: Dictionary containing book information
            
        Returns:
            Path object for the target directory
        """
        repo_name = book_info.get('repo', book_info.get('name', 'unknown'))
        subject = book_info.get('subject', 'Other')
        
        # Use existing dynamic language detection functionality
        detected_language = self._get_repository_language(book_info, repo_name)
        
        # Determine educational level and discipline
        discipline = self._map_subject_to_discipline(subject)
        educational_level = self._determine_educational_level(book_info)
        
        # Handle special categories (non-textbook content)
        if self._is_personal_or_experimental(book_info):
            # Personal/experimental work goes to Language/Specialized/Personal
            target_path = self.books_path / detected_language / "Specialized" / "Personal" / repo_name
        elif self._is_professional_development(book_info):
            # Professional development goes to Language/Professional/Level
            target_path = self.books_path / detected_language / "Professional" / educational_level / repo_name
        else:
            # Regular textbooks go to Language/Discipline/Level structure
            target_path = self.books_path / detected_language / discipline / educational_level / repo_name
        
        # Create directory structure if it doesn't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        return target_path
    
    def _extract_subject_from_repo_name(self, repo_name: str) -> str:
        """
        Extract subject from repository name.
        
        Args:
            repo_name: Repository name
            
        Returns:
            Extracted subject string
        """
        # Remove common prefixes
        name = repo_name.lower()
        for prefix in ['osbooks-', 'cnxbook-']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Subject keywords for classification
        subject_keywords = {
            'physics': 'physics',
            'college-physics': 'physics',
            'university-physics': 'physics',
            'astronomy': 'astronomy',
            'biology': 'biology',
            'anatomy-physiology': 'anatomy',
            'microbiology': 'microbiology',
            'chemistry': 'chemistry',
            'organic-chemistry': 'chemistry',
            'psychology': 'psychology',
            'sociology': 'sociology',
            'business': 'business',
            'entrepreneurship': 'business',
            'finance': 'finance',
            'marketing': 'marketing',
            'economics': 'economics',
            'mathematics': 'mathematics',
            'statistics': 'statistics',
            'prealgebra': 'mathematics',
            'government': 'political-science',
            'political-science': 'political-science',
            'history': 'history',
            'writing': 'english',
            'philosophy': 'philosophy',
            'anthropology': 'anthropology',
            'college-success': 'study-skills'
        }
        
        # Find best match
        for keyword, subject in subject_keywords.items():
            if keyword in name:
                return subject
        
        # Return first word as fallback
        words = name.replace('-', ' ').split()
        return words[0] if words else 'unknown'
    
    def _is_openstax_repository(self, book_info: Dict[str, Any]) -> bool:
        """Check if the repository is an OpenStax repository."""
        repo_name = book_info.get('repo', '').lower()
        org = book_info.get('org', '').lower()
        source = book_info.get('source', '').lower()
        
        # Check for OpenStax indicators
        openstax_indicators = [
            'osbooks-',
            'openstax',
            'derived-from-osbooks-'
        ]
        
        return (
            any(indicator in repo_name for indicator in openstax_indicators) or
            'openstax' in org or
            'openstax' in source
        )
    
    def _is_cnx_repository(self, book_info: Dict[str, Any]) -> bool:
        """Check if the repository is a CNX repository."""
        repo_name = book_info.get('repo', '').lower()
        org = book_info.get('org', '').lower()
        
        # Check for CNX indicators
        cnx_indicators = [
            'cnxbook-',
            'cnx-'
        ]
        
        return (
            any(indicator in repo_name for indicator in cnx_indicators) or
            'cnx' in org
        )
    
    def _map_subject_to_discipline(self, subject: str) -> str:
        """Map subject names to standardized discipline names."""
        subject_lower = subject.lower()
        
        # Map common subject variations to standard disciplines
        discipline_mapping = {
            'physics': 'Physics',
            'biology': 'Biology', 
            'chemistry': 'Chemistry',
            'mathematics': 'Mathematics',
            'math': 'Mathematics',
            'calculus': 'Mathematics',
            'algebra': 'Mathematics',
            'statistics': 'Statistics',
            'economics': 'Economics',
            'business': 'Business',
            'psychology': 'Psychology',
            'sociology': 'Sociology',
            'engineering': 'Engineering',
            'music': 'Music',
            'computer science': 'Engineering',
            'cs': 'Engineering'
        }
        
        # Check for exact matches first
        for key, discipline in discipline_mapping.items():
            if key in subject_lower:
                return discipline
                
        # If no match found, capitalize the subject
        return subject.title() if subject else 'Other'
    
    def _determine_educational_level(self, book_info: Dict[str, Any]) -> str:
        """
        Determine educational level from book information using comprehensive analysis.
        
        Checks repository name, title, description, and if needed, repository content
        to accurately classify educational level and prevent misplacement.
        """
        repo_name = book_info.get('repo', book_info.get('name', '')).lower()
        title = book_info.get('title', '').lower() 
        description = book_info.get('description', '').lower()
        
        # Check if discovery provided a level hint
        level_hint = book_info.get('level_hint')
        if level_hint:
            return level_hint
        
        # Enhanced high school indicators
        high_school_indicators = [
            'high-school', 'high_school', 'highschool', 'hs-', 'ap-', '-ap',
            'prealgebra', 'pre-algebra', 'basic-music-theory',
            'high school', 'ap physics', 'ap chemistry', 'ap biology',
            'secondary', 'prep', 'preparatory'
        ]
        
        # Enhanced graduate indicators  
        graduate_indicators = [
            'graduate', 'phd', 'doctoral', 'advanced research', 'research methods',
            'signal-and-information-processing', 'compressive-sensing',
            'seismic-imaging', 'machine-learning', 'masters', 'thesis'
        ]
        
        # University-specific indicators (helps distinguish from HS)
        university_indicators = [
            'university', 'college', 'undergraduate', 'calculus-based',
            'advanced placement', 'honors', 'survey course'
        ]
        
        # Check for high school level first (most specific)
        for indicator in high_school_indicators:
            if indicator in repo_name or indicator in title or indicator in description:
                return 'HighSchool'
        
        # Special case: Check for specific repository content patterns
        if self._check_repository_content_for_level(book_info, repo_name) == 'HighSchool':
            return 'HighSchool'
                
        # Check for graduate level
        for indicator in graduate_indicators:
            if indicator in repo_name or indicator in title or indicator in description:
                return 'Graduate'
        
        # Check for explicit university indicators
        for indicator in university_indicators:
            if indicator in repo_name or indicator in title or indicator in description:
                return 'University'
                
        # Subject-specific level analysis
        level_from_subject = self._determine_level_from_subject_analysis(repo_name, title)
        if level_from_subject:
            return level_from_subject
                
        # Default to University level (most common for OpenStax)
        return 'University'
    
    def _check_repository_content_for_level(self, book_info: Dict[str, Any], repo_name: str) -> Optional[str]:
        """
        Check repository content for educational level indicators.
        
        This function looks at actual repository content to detect level,
        particularly useful for cases like osbooks-physics which has CNX_HSPhysics
        media files indicating high school level.
        """
        # If repository exists locally, check for content-based indicators
        potential_paths = [
            self.books_path / "english" / "Physics" / "University" / repo_name,
            self.books_path / "english" / "Physics" / "HighSchool" / repo_name,
            self.books_path / "english" / "Uncategorized" / "University" / repo_name
        ]
        
        for repo_path in potential_paths:
            if repo_path.exists():
                # Check media files for level indicators
                media_path = repo_path / "media"
                if media_path.exists():
                    try:
                        # Look for high school indicators in media file names
                        for media_file in media_path.iterdir():
                            if media_file.is_file():
                                filename = media_file.name.lower()
                                if 'cnx_hsphysics' in filename or 'hs_physics' in filename or 'highschool' in filename:
                                    logger.info(f"Found HS content indicator in {repo_name}: {filename}")
                                    return 'HighSchool'
                                elif 'cnx_upsphysics' in filename or 'university_physics' in filename:
                                    return 'University'
                    except Exception as e:
                        logger.debug(f"Could not check media files for {repo_name}: {e}")
                        
                # Check collection.xml for level indicators
                collections_path = repo_path / "collections"
                if collections_path.exists():
                    try:
                        for xml_file in collections_path.glob("*.xml"):
                            content = xml_file.read_text()
                            if 'high school' in content.lower() or 'secondary' in content.lower():
                                return 'HighSchool'
                    except Exception as e:
                        logger.debug(f"Could not check collections for {repo_name}: {e}")
        
        return None
    
    def _determine_level_from_subject_analysis(self, repo_name: str, title: str) -> Optional[str]:
        """
        Determine educational level based on subject-specific analysis.
        
        Different subjects have different naming conventions and level indicators.
        """
        # Physics-specific analysis
        if 'physics' in repo_name or 'physics' in title:
            if 'college-physics' in repo_name:
                return 'University'  # College Physics is introductory university
            elif 'university-physics' in repo_name:
                return 'University'  # University Physics is advanced university
            elif repo_name == 'osbooks-physics' or 'basic physics' in title:
                # The generic "physics" book is actually high school level
                return 'HighSchool'
        
        # Mathematics-specific analysis
        if any(term in repo_name for term in ['algebra', 'precalculus', 'prealgebra']):
            if 'pre' in repo_name or 'basic' in repo_name:
                return 'HighSchool'
            elif 'college' in repo_name:
                return 'University'
        
        # Chemistry-specific analysis
        if 'chemistry' in repo_name:
            if 'general' in repo_name or 'intro' in repo_name:
                return 'University'
            elif 'organic' in repo_name or 'physical' in repo_name:
                return 'University'  # These are typically university-level
        
        return None
    
    def _is_personal_or_experimental(self, book_info: Dict[str, Any]) -> bool:
        """Check if the repository is personal or experimental work."""
        repo_name = book_info.get('repo', book_info.get('name', '')).lower()
        
        personal_indicators = [
            'art-of-the-pfug', 'jsea-effective-practices-personal',
            'mark-xiornik-rozen-pettinelli', 'my-final-analysis'
        ]
        
        return any(indicator in repo_name for indicator in personal_indicators)
    
    def _is_professional_development(self, book_info: Dict[str, Any]) -> bool:
        """Check if the repository is for professional development."""
        repo_name = book_info.get('repo', book_info.get('name', '')).lower()
        
        professional_indicators = [
            'performance-assessment-in-educational-leadership',
            'teacher-training', 'professional-development'
        ]
        
        return any(indicator in repo_name for indicator in professional_indicators)
    
    def _run_git_command(self, cmd: List[str], repo_path: Path, capture_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run a git command in the specified repository.
        
        Args:
            cmd: Git command as list of strings (without 'git' prefix)
            repo_path: Path to the repository
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            CompletedProcess result
        """
        try:
            # Prepend 'git' to the command
            full_cmd = ['git'] + cmd
            
            # Set environment to prevent interactive prompts
            env = self._get_git_env()
            
            result = subprocess.run(
                full_cmd,
                cwd=repo_path,
                capture_output=capture_output,
                text=True,
                timeout=30,
                env=env,
                stdin=subprocess.DEVNULL  # Prevent stdin input requests
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timeout: {' '.join(full_cmd)} in {repo_path}")
            raise
        except Exception as e:
            logger.error(f"Git command failed: {' '.join(full_cmd)} in {repo_path}: {e}")
            raise
    

    def _validate_repository_content(self, book_info: Dict[str, Any], target_path: Path, openstax_only: bool = False) -> bool:
        """
        Validate that repository is an educational textbook and not infrastructure/utility code.
        
        This prevents non-textbook repositories from contaminating the Books directory.
        """
        clone_url = book_info.get('clone_url', '')
        repo_name = book_info.get('repo', '').lower()
        description = book_info.get('description', '').lower()
        
        # CRITICAL: Check for non-textbook repository patterns
        non_textbook_patterns = [
            'cms', 'salesforce', 'utilities', 'utility', 'util', 'research-kg', 
            'research', 'tagging-legend', 'legend', 'min-book', 'setup-',
            'machine', 'github.io', 'onthego', 'setup', 'work-management',
            'management', 'dev', 'development', 'build', 'tools', 'devops',
            'automation', 'platform', 'service', 'services', 'core', 'lib',
            'library', 'shared', 'common', 'base', 'api', 'backend', 'frontend',
            'admin', 'dashboard', 'app', 'cli', 'test', 'demo', 'template',
            'infrastructure', 'deployment', 'pipeline'
        ]
        
        # Reject repositories with non-textbook patterns
        for pattern in non_textbook_patterns:
            if pattern in repo_name or pattern in description:
                logger.warning(f"Non-textbook repository rejected: {repo_name} (contains '{pattern}')")
                return False
        
        # When openstax_only=True, be more restrictive - ONLY allow actual OpenStax repositories
        if openstax_only:
            is_openstax = False
            
            # Check for OpenStax organization (most reliable)
            if 'openstax' in clone_url.lower():
                is_openstax = True
            
            # Check for osbooks- prefix (OpenStax book naming convention)
            elif repo_name.startswith('osbooks-'):
                is_openstax = True
            
            # Do NOT allow cnxbook- repositories when openstax_only=True
            # cnxbook repositories are user-created content, not official OpenStax
            
            if not is_openstax:
                logger.warning(f"Non-OpenStax repository rejected (openstax_only=True): {repo_name} ({clone_url})")
                return False
        else:
            # When openstax_only=False, allow broader educational content
            is_educational = False
            
            # Check for OpenStax organization
            if 'openstax' in clone_url.lower():
                is_educational = True
            
            # Check for osbooks- prefix (OpenStax book naming convention)
            elif repo_name.startswith('osbooks-'):
                is_educational = True
            
            # Check for cnxbook- prefix (CNX user books - allowed when not openstax_only)
            elif repo_name.startswith('cnxbook-'):
                is_educational = True
            
            if not is_educational:
                logger.warning(f"Non-educational repository rejected: {repo_name} ({clone_url})")
                return False
        
        # Additional textbook validation for OpenStax repositories
        textbook_indicators = [
            'osbooks-', 'cnxbook-', 'physics', 'biology', 'chemistry', 'mathematics',
            'calculus', 'statistics', 'psychology', 'sociology', 'economics', 'business',
            'anatomy', 'physiology', 'microbiology', 'organic', 'college', 'prealgebra',
            'algebra', 'geometry', 'trigonometry', 'precalculus', 'accounting', 'finance',
            'entrepreneurship', 'philosophy', 'history', 'anthropology', 'government',
            'astronomy', 'principles', 'introduction', 'university'
        ]
        
        # Check if repository has textbook indicators
        has_textbook_indicators = any(indicator in repo_name or indicator in description 
                                    for indicator in textbook_indicators)
        
        if not has_textbook_indicators:
            logger.warning(f"Repository lacks textbook indicators: {repo_name}")
            return False
        
        # Additional validation for textbook directories
        if 'Textbooks' in str(target_path):
            # Validate subject alignment for OpenStax books
            expected_subject = self._extract_subject_from_url(clone_url)
            path_subject = self._extract_subject_from_path(target_path)
            
            if expected_subject and path_subject and expected_subject != path_subject:
                # Allow astronomy in physics (common practice)
                if not (expected_subject == 'astronomy' and path_subject == 'physics'):
                    logger.warning(f"Subject mismatch for OpenStax book: URL suggests {expected_subject}, path suggests {path_subject}")
                    # Don't fail validation, just warn for OpenStax books
        
        logger.debug(f"OpenStax repository validated: {repo_name}")
        return True
    
    def _extract_subject_from_url(self, clone_url: str) -> Optional[str]:
        """Extract subject from repository URL/name."""
        url_lower = clone_url.lower()
        
        subjects = {
            'physics': ['physics', 'college-physics', 'university-physics'],
            'astronomy': ['astronomy'],
            'biology': ['biology', 'microbiology'],
            'chemistry': ['chemistry', 'organic-chemistry'],
            'mathematics': ['mathematics', 'math', 'statistics', 'calculus', 'prealgebra'],
            'psychology': ['psychology'],
            'sociology': ['sociology'],
            'business': ['business', 'entrepreneurship', 'finance', 'marketing']
        }
        
        for subject, keywords in subjects.items():
            if any(keyword in url_lower for keyword in keywords):
                return subject
        
        return None
    
    def _extract_subject_from_path(self, target_path: Path) -> Optional[str]:
        """Extract subject from file system path."""
        parts = [p.lower() for p in target_path.parts]
        
        subject_indicators = ['physics', 'biology', 'chemistry', 'mathematics', 'astronomy', 
                            'psychology', 'sociology', 'business', 'statistics']
        
        for part in parts:
            if part in subject_indicators:
                return part
        
        return None
    
    def _get_repository_language(self, book_info: Dict[str, Any], repo_name: str) -> str:
        """
        Get the language of a repository using existing dynamic language detection.
        
        Args:
            book_info: Dictionary containing book information
            repo_name: Repository name
            
        Returns:
            Detected language (defaults to 'english')
        """
        try:
            # Try to get language from book_info first
            if 'language' in book_info and book_info['language']:
                return book_info['language'].lower()
            
            # Initialize language detector if needed
            if not hasattr(self, '_language_detector'):
                from .language_detector import LanguageDetector
                self._language_detector = LanguageDetector()
            
            # Use the existing dynamic language detection with repository name analysis
            # Since the repository isn't cloned yet, we'll use name-based detection
            detected_language = self._language_detector._analyze_repository_name(repo_name)
            
            if detected_language:
                return detected_language
            
            # Check known repositories as fallback
            known_language = self._language_detector._check_known_repositories(repo_name)
            if known_language:
                return known_language
            
            # Default to English for OpenStax repositories
            return 'english'
            
        except Exception as e:
            logger.warning(f"Error detecting language for {repo_name}: {e}")
            return 'english'
    
    def _convert_clone_url_format(self, clone_url: str) -> str:
        """
        Convert clone URL between SSH and HTTPS formats based on configuration.
        
        Args:
            clone_url: The original clone URL
            
        Returns:
            Converted clone URL in the preferred format
        """
        if self.config.use_ssh_for_git:
            # Convert HTTPS to SSH format
            if clone_url.startswith('https://github.com/'):
                # Extract org/repo.git from https://github.com/org/repo.git
                path = clone_url.replace('https://github.com/', '')
                return f"git@github.com:{path}"
        else:
            # Convert SSH to HTTPS format
            if clone_url.startswith('git@github.com:'):
                # Extract org/repo.git from git@github.com:org/repo.git
                path = clone_url.replace('git@github.com:', '')
                return f"https://github.com/{path}"
        
        # Return unchanged if already in correct format or unrecognized format
        return clone_url