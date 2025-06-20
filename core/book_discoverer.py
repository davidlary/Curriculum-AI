"""
Book discovery module for finding OpenStax and other open textbooks.

This module implements responsible web scraping and API-based discovery
with appropriate rate limiting and respectful request patterns.
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import json
import re

from .config import OpenBooksConfig
from .data_config import get_data_config

logger = logging.getLogger(__name__)


class BookDiscoverer:
    """Discovers available open textbooks from various sources."""
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize with configuration."""
        self.config = config
        self.data_config = get_data_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenBooks/1.0 (Educational Research; davidlary@me.com)'
        })
    
    def discover_openstax_books(self, openstax_only: bool = False, git_only: bool = False) -> List[Dict[str, Any]]:
        """
        Discover OpenStax books through multiple strategies with graceful fallback.
        
        Returns:
            List of book dictionaries with metadata
        """
        logger.info("Starting OpenStax book discovery")
        
        discovered_books = []
        
        # Strategy 1: PRIORITY - Autonomous GitHub discovery (not hardcoded)
        if not openstax_only:
            try:
                logger.info("Starting autonomous GitHub repository discovery...")
                github_books = self._discover_github_repositories_sequential()
                discovered_books.extend(github_books)
                logger.info(f"Autonomous GitHub discovery found {len(github_books)} books")
            except Exception as e:
                logger.warning(f"GitHub discovery failed (rate limiting expected): {e}")
        else:
            logger.info("Skipping CNX discovery due to --openstax-only flag")
        
        # Strategy 2: OpenStax organization discovery
        try:
            logger.info("Discovering OpenStax organization repositories...")
            openstax_books = self._discover_openstax_org_comprehensive()
            discovered_books.extend(openstax_books)
            logger.info(f"OpenStax org discovery found {len(openstax_books)} additional books")
        except Exception as e:
            logger.warning(f"OpenStax org discovery failed: {e}")
        
        # Strategy 3: Enhanced known books as fallback only if discovery fails
        if len(discovered_books) < 20:  # Only as fallback if autonomous discovery failed
            try:
                logger.info("Using enhanced catalog as fallback (autonomous discovery had limited results)...")
                known_books = self._get_enhanced_known_books()
                discovered_books.extend(known_books)
                logger.info(f"Fallback catalog loaded {len(known_books)} OpenStax books")
            except Exception as e:
                logger.warning(f"Enhanced catalog failed: {e}")
        
        # Strategy 3: Try OpenStax website PDF discovery (unless git_only is enabled)
        if not git_only:
            try:
                logger.info("Attempting OpenStax PDF discovery...")
                pdf_books = self._discover_openstax_pdfs()
                discovered_books.extend(pdf_books)
                logger.info(f"PDF discovery found {len(pdf_books)} books")
            except Exception as e:
                logger.warning(f"PDF discovery failed: {e}")
        else:
            logger.info("Skipping PDF discovery due to --git-only flag")
        
        # Strategy 4: Fallback to original known repositories if needed
        if len(discovered_books) < 20:
            try:
                logger.warning("Low book count, adding fallback known repositories...")
                fallback_books = self.config.get_known_repositories()
                discovered_books.extend(fallback_books)
                logger.info(f"Fallback added {len(fallback_books)} books")
            except Exception as e:
                logger.warning(f"Fallback failed: {e}")
        
        # Deduplicate based on repository name
        unique_books = self._deduplicate_books(discovered_books)
        
        logger.info(f"Discovered {len(unique_books)} unique books total")
        return unique_books
    
    def _discover_github_repositories_sequential(self) -> List[Dict[str, Any]]:
        """Discover books from GitHub using sequential requests to avoid rate limiting."""
        logger.info("Auto-discovering GitHub repositories (sequential mode)")
        
        books = []
        
        # Conservative search strategy - one at a time with delays
        try:
            logger.info("Running cnx-user-books discovery (limited)...")
            # Get only the first page to avoid rate limits
            strategy_books = self._discover_cnx_user_books_org_limited()
            books.extend(strategy_books)
            logger.info(f"cnx-user-books found {len(strategy_books)} books")
            
            # Add delay between strategies
            time.sleep(5)
            
            if len(books) < 20:  # Only continue if we need more
                logger.info("Running OpenStax org discovery (limited)...")
                strategy_books = self._discover_openstax_org_limited()
                books.extend(strategy_books)
                logger.info(f"openstax-org found {len(strategy_books)} books")
                    
        except Exception as e:
            logger.warning(f"Sequential discovery failed: {e}")
        
        logger.info(f"Auto-discovered {len(books)} repositories from GitHub (sequential)")
        return books
    
    def _discover_cnx_user_books_org_limited(self) -> List[Dict[str, Any]]:
        """Discover repositories from cnx-user-books organization (limited to avoid rate limits)."""
        logger.info("Discovering cnx-user-books organization repositories (limited)")
        
        books = []
        
        try:
            time.sleep(self.config.request_delay_seconds)
            
            # Get only the first page to avoid rate limits
            org_url = f"{self.config.github_api_base_url}/orgs/{self.config.cnx_user_books_org}/repos"
            params = {
                'type': 'public',
                'sort': 'updated',
                'direction': 'desc',
                'per_page': 30,  # Reduced from 100
                'page': 1
            }
            
            response = self._make_request(org_url, params=params)
            if response and response.status_code == 200:
                repos = response.json()
                
                for repo in repos:
                    # Filter for textbook-related repositories
                    if self._is_textbook_repository(repo):
                        book_info = self._extract_book_info_from_repo(repo)
                        if book_info:
                            books.append(book_info)
                
        except Exception as e:
            logger.warning(f"Error discovering cnx-user-books org (limited): {e}")
        
        logger.info(f"Found {len(books)} books from cnx-user-books organization (limited)")
        return books
    
    def _discover_openstax_org_limited(self) -> List[Dict[str, Any]]:
        """Discover repositories from OpenStax organization (limited)."""
        logger.info("Discovering OpenStax organization repositories (limited)")
        
        books = []
        
        try:
            time.sleep(self.config.request_delay_seconds)
            
            org_url = f"{self.config.github_api_base_url}/orgs/openstax/repos"
            params = {'type': 'public', 'per_page': 30}  # Reduced from 100
            
            response = self._make_request(org_url, params=params)
            if response and response.status_code == 200:
                repos = response.json()
                
                for repo in repos:
                    if self._is_textbook_repository(repo):
                        book_info = self._extract_book_info_from_repo(repo)
                        if book_info:
                            books.append(book_info)
            
        except Exception as e:
            logger.warning(f"Error discovering openstax organization (limited): {e}")
        
        logger.info(f"Found {len(books)} books from OpenStax organizations (limited)")
        return books
    
    def _discover_cnx_user_books_org(self) -> List[Dict[str, Any]]:
        """Discover all repositories from cnx-user-books organization."""
        logger.info("Discovering cnx-user-books organization repositories")
        
        books = []
        page = 1
        per_page = 100  # Maximum allowed by GitHub API
        
        while True:
            try:
                time.sleep(self.config.request_delay_seconds)
                
                # Get all repositories from the organization
                org_url = f"{self.config.github_api_base_url}/orgs/{self.config.cnx_user_books_org}/repos"
                params = {
                    'type': 'public',
                    'sort': 'updated',
                    'direction': 'desc',
                    'per_page': per_page,
                    'page': page
                }
                
                response = self._make_request(org_url, params=params)
                if not response or response.status_code != 200:
                    break
                
                repos = response.json()
                if not repos:  # No more repositories
                    break
                
                for repo in repos:
                    # Filter for textbook-related repositories
                    if self._is_textbook_repository(repo):
                        book_info = self._extract_book_info_from_repo(repo)
                        if book_info:
                            books.append(book_info)
                
                # Check if we got fewer results than requested (last page)
                if len(repos) < per_page:
                    break
                
                page += 1
                
            except Exception as e:
                logger.warning(f"Error discovering cnx-user-books org (page {page}): {e}")
                break
        
        logger.info(f"Found {len(books)} books from cnx-user-books organization")
        return books
    
    def _discover_openstax_org_comprehensive(self) -> List[Dict[str, Any]]:
        """Comprehensive discovery of OpenStax organization repositories with pagination."""
        logger.info("Comprehensive OpenStax organization discovery")
        
        books = []
        organizations = ['openstax', 'cnx-user-books']  # Focus on the main OpenStax orgs
        
        for org in organizations:
            try:
                logger.info(f"Discovering repositories from {org} organization...")
                page = 1
                per_page = 100
                
                while True:
                    time.sleep(self.config.request_delay_seconds)
                    
                    org_url = f"{self.config.github_api_base_url}/orgs/{org}/repos"
                    params = {
                        'type': 'public', 
                        'per_page': per_page,
                        'page': page,
                        'sort': 'updated',
                        'direction': 'desc'
                    }
                    
                    response = self._make_request(org_url, params=params)
                    if not response or response.status_code != 200:
                        logger.warning(f"Failed to get page {page} from {org}: {response.status_code if response else 'No response'}")
                        break
                    
                    repos = response.json()
                    if not repos:  # No more repositories
                        break
                    
                    logger.info(f"Processing page {page} with {len(repos)} repositories from {org}")
                    
                    for repo in repos:
                        if self._is_textbook_repository(repo):
                            book_info = self._extract_book_info_from_repo(repo)
                            if book_info:
                                books.append(book_info)
                    
                    # Check if we got fewer results than requested (last page)
                    if len(repos) < per_page:
                        break
                        
                    page += 1
                    
                    # Safety limit to prevent infinite loops
                    if page > 10:
                        logger.warning(f"Reached page limit for {org}")
                        break
                
            except Exception as e:
                logger.warning(f"Error discovering {org} organization: {e}")
                continue
        
        logger.info(f"Found {len(books)} books from comprehensive OpenStax organizations discovery")
        return books
    
    def _discover_openstax_org(self) -> List[Dict[str, Any]]:
        """Discover repositories from OpenStax organization."""
        logger.info("Discovering OpenStax organization repositories")
        
        books = []
        organizations = ['openstax', 'OpenStax']
        
        for org in organizations:
            try:
                time.sleep(self.config.request_delay_seconds)
                
                org_url = f"{self.config.github_api_base_url}/orgs/{org}/repos"
                params = {'type': 'public', 'per_page': 100}
                
                response = self._make_request(org_url, params=params)
                if response and response.status_code == 200:
                    repos = response.json()
                    
                    for repo in repos:
                        if self._is_textbook_repository(repo):
                            book_info = self._extract_book_info_from_repo(repo)
                            if book_info:
                                books.append(book_info)
                
            except Exception as e:
                logger.warning(f"Error discovering {org} organization: {e}")
                continue
        
        logger.info(f"Found {len(books)} books from OpenStax organizations")
        return books
    
    def _discover_general_textbook_repos(self) -> List[Dict[str, Any]]:
        """Discover textbook repositories using general search terms."""
        logger.info("Discovering general textbook repositories")
        
        books = []
        search_terms = [
            'openstax textbook',
            'open textbook cnx',
            'connexions textbook',
            'oer textbook',
            'open educational resources'
        ]
        
        for term in search_terms:
            try:
                time.sleep(self.config.request_delay_seconds)
                
                search_url = f"{self.config.github_api_base_url}/search/repositories"
                params = {
                    'q': f'{term} language:tex language:xml',
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': 50
                }
                
                response = self._make_request(search_url, params=params)
                if response and response.status_code == 200:
                    data = response.json()
                    
                    for repo in data.get('items', []):
                        if self._is_textbook_repository(repo):
                            book_info = self._extract_book_info_from_repo(repo)
                            if book_info:
                                books.append(book_info)
                
            except Exception as e:
                logger.warning(f"Error searching term '{term}': {e}")
                continue
        
        logger.info(f"Found {len(books)} books from general search")
        return books
    
    def _discover_related_organizations(self) -> List[Dict[str, Any]]:
        """Discover repositories from related educational organizations."""
        logger.info("Discovering related organization repositories")
        
        books = []
        related_orgs = [
            'OpenStax',
            'MIT-OCW',
            'edX',
            'ocw-ui',
            'oer-commons'
        ]
        
        for org in related_orgs:
            try:
                time.sleep(self.config.request_delay_seconds)
                
                # Search for textbook repositories in these organizations
                search_url = f"{self.config.github_api_base_url}/search/repositories"
                params = {
                    'q': f'org:{org} textbook OR book OR course',
                    'sort': 'updated',
                    'order': 'desc',
                    'per_page': 30
                }
                
                response = self._make_request(search_url, params=params)
                if response and response.status_code == 200:
                    data = response.json()
                    
                    for repo in data.get('items', []):
                        if self._is_textbook_repository(repo):
                            book_info = self._extract_book_info_from_repo(repo)
                            if book_info:
                                books.append(book_info)
                
            except Exception as e:
                logger.warning(f"Error discovering {org} organization: {e}")
                continue
        
        logger.info(f"Found {len(books)} books from related organizations")
        return books
    
    def _is_textbook_repository(self, repo: Dict[str, Any]) -> bool:
        """Determine if a repository is likely a textbook with enhanced detection."""
        repo_name = (repo.get('name') or '').lower()
        description = (repo.get('description') or '').lower()
        owner = (repo.get('owner', {}).get('login') or '').lower()
        
        # Get indicators from data configuration (data-driven)
        strong_indicators = self.data_config.get_strong_indicators()
        subject_indicators = self.data_config.get_subject_indicators()
        educational_indicators = self.data_config.get_educational_indicators()
        exclude_indicators = self.data_config.get_exclude_indicators()
        
        # Check for strong indicators first
        has_strong = any(indicator in repo_name for indicator in strong_indicators)
        if has_strong:
            # But still check for exclusions even with strong indicators
            has_negative = any(indicator in repo_name or indicator in description 
                             for indicator in exclude_indicators)
            if not has_negative:
                return True
        
        # For trusted organizations, be more inclusive (data-driven)
        trusted_orgs = [org.lower() for org in self.data_config.get_trusted_organizations()]
        if owner in trusted_orgs:
            # Check if it has subject or educational indicators
            has_subject = any(indicator in repo_name or indicator in description 
                            for indicator in subject_indicators)
            has_educational = any(indicator in repo_name or indicator in description 
                                for indicator in educational_indicators)
            
            if has_subject or has_educational:
                # Check it's not explicitly excluded
                has_negative = any(indicator in repo_name or indicator in description 
                                 for indicator in exclude_indicators)
                if not has_negative:
                    return True
        
        # General checks for other repositories
        has_positive = any(indicator in repo_name or indicator in description 
                          for indicator in subject_indicators + educational_indicators)
        
        # Check if repository has negative indicators
        has_negative = any(indicator in repo_name or indicator in description 
                          for indicator in exclude_indicators)
        
        # Additional quality checks (data-driven)
        quality_thresholds = self.data_config.get_quality_thresholds()
        min_size = quality_thresholds.get('min_size_kb', 50)
        prefer_non_forks = quality_thresholds.get('prefer_non_forks', True)
        
        has_reasonable_size = repo.get('size', 0) > min_size
        not_fork = not repo.get('fork', False) if prefer_non_forks else True
        
        return has_positive and not has_negative and has_reasonable_size and not_fork
    
    def _search_additional_repositories(self) -> List[Dict[str, Any]]:
        """Search for additional OpenStax repositories with different naming patterns."""
        logger.info("Searching for additional repositories")
        
        books = []
        
        # Get subject search patterns from data configuration (data-driven)  
        subjects = self.data_config.get_subject_search_patterns()
        
        for subject in subjects:
            try:
                time.sleep(self.config.request_delay_seconds)
                
                # Search for repositories containing subject
                search_url = f"{self.config.github_api_base_url}/search/repositories"
                params = {
                    'q': f'{subject} openstax OR cnx',
                    'sort': 'stars',
                    'order': 'desc'
                }
                
                response = self._make_request(search_url, params=params)
                if response and response.status_code == 200:
                    data = response.json()
                    
                    for repo in data.get('items', [])[:5]:  # Limit to top 5
                        if self._is_openstax_repository(repo):
                            book_info = self._extract_book_info_from_repo(repo)
                            if book_info:
                                books.append(book_info)
                
            except Exception as e:
                logger.warning(f"Error searching subject {subject}: {e}")
                continue
        
        logger.info(f"Found {len(books)} additional books")
        return books
    
    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = None) -> Optional[requests.Response]:
        """Make HTTP request with retry logic."""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(
                    url, 
                    params=params,
                    timeout=self.config.timeout_seconds
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 403:
                    logger.warning(f"Rate limited (403) for {url}, waiting longer")
                    time.sleep(self.config.request_delay_seconds * 2)
                elif response.status_code == 404:
                    logger.debug(f"Not found (404): {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}: {e}")
            
            if attempt < max_retries:
                wait_time = self.config.request_delay_seconds * (2 ** attempt)
                time.sleep(min(wait_time, 30))  # Cap at 30 seconds
        
        logger.error(f"Failed to fetch {url} after {max_retries + 1} attempts")
        return None
    
    def _extract_book_info_from_repo(self, repo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract book information from GitHub repository data."""
        try:
            name = repo.get('name', '')
            description = repo.get('description', '')
            
            # Extract subject from repository name or description
            subject = self._extract_subject(name, description)
            
            # Clean up book name
            book_name = self._clean_book_name(name)
            
            # Extract educational level hints
            level_hint = self._extract_level_hint(name, description)
            
            book_info = {
                'name': book_name,
                'repo': name,
                'org': repo.get('owner', {}).get('login', 'unknown'),
                'subject': subject,
                'description': description,
                'url': repo.get('html_url', ''),
                'clone_url': repo.get('clone_url', ''),
                'updated_at': repo.get('updated_at', ''),
                'size_kb': repo.get('size', 0),
                'stars': repo.get('stargazers_count', 0),
                'source': 'github_discovery',
                'format': 'git',  # Mark all GitHub repositories as git format
                'level_hint': level_hint  # Add level hint for better categorization
            }
            
            return book_info
            
        except Exception as e:
            logger.warning(f"Error extracting book info from repo: {e}")
            return None
    
    def _extract_subject(self, name: str, description: str) -> str:
        """Extract subject area from repository name and description."""
        text = f"{name} {description}".lower()
        
        # Subject mapping
        subject_patterns = {
            'Physics': ['physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics'],
            'Biology': ['biology', 'anatomy', 'physiology', 'microbiology', 'life science'],
            'Chemistry': ['chemistry', 'organic chemistry', 'chemical'],
            'Mathematics': ['math', 'calculus', 'algebra', 'trigonometry', 'precalculus'],
            'Statistics': ['statistics', 'statistical', 'probability'],
            'Economics': ['economics', 'microeconomics', 'macroeconomics'],
            'Psychology': ['psychology', 'psychological'],
            'Sociology': ['sociology', 'social'],
            'Business': ['business', 'management', 'organizational'],
            'Computer Science': ['computer science', 'programming', 'software']
        }
        
        for subject, patterns in subject_patterns.items():
            if any(pattern in text for pattern in patterns):
                return subject
        
        return 'Other'
    
    def _extract_level_hint(self, name: str, description: str) -> Optional[str]:
        """
        Extract educational level hints from repository name and description.
        
        Provides hints to help the repository manager make better categorization decisions.
        """
        text = f"{name} {description}".lower()
        
        # High school level indicators
        hs_indicators = [
            'high school', 'high-school', 'hs-', 'secondary',
            'ap ', 'advanced placement', 'preparatory', 'prep'
        ]
        
        # University level indicators  
        university_indicators = [
            'university', 'college', 'undergraduate', 'intro', 'survey'
        ]
        
        # Graduate level indicators
        graduate_indicators = [
            'graduate', 'advanced', 'research', 'masters', 'doctoral', 'phd'
        ]
        
        # Subject-specific level analysis
        if 'physics' in name:
            if name == 'osbooks-physics':
                return 'HighSchool'  # The generic physics book is high school
            elif 'college-physics' in name:
                return 'University'  # College physics is introductory university
            elif 'university-physics' in name:
                return 'University'  # University physics is advanced university
        
        # Check for explicit level indicators
        for indicator in hs_indicators:
            if indicator in text:
                return 'HighSchool'
                
        for indicator in graduate_indicators:
            if indicator in text:
                return 'Graduate'
                
        for indicator in university_indicators:
            if indicator in text:
                return 'University'
        
        # Mathematics level hints
        if any(term in name for term in ['prealgebra', 'pre-algebra', 'basic-math']):
            return 'HighSchool'
        elif any(term in name for term in ['calculus', 'differential', 'linear-algebra']):
            return 'University'
        
        return None  # No specific level hint available
    
    def _clean_book_name(self, repo_name: str) -> str:
        """Clean repository name to create readable book name."""
        # Remove common prefixes
        name = repo_name
        prefixes = ['cnxbook-', 'osbooks-', 'derived-from-osbooks-', 'openstax-']
        
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Replace hyphens and underscores with spaces
        name = name.replace('-', ' ').replace('_', ' ')
        
        # Title case
        name = name.title()
        
        # Clean up common words
        replacements = {
            'Ap': 'AP',
            'Usa': 'USA',
            'Us': 'US',
            'Uk': 'UK'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name
    
    def _detect_subject_from_text(self, text: str) -> str:
        """Detect subject area from any text."""
        text = text.lower()
        
        # Subject mapping (same as in _extract_subject)
        subject_patterns = {
            'Physics': ['physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'optics'],
            'Biology': ['biology', 'anatomy', 'physiology', 'microbiology', 'life science'],
            'Chemistry': ['chemistry', 'organic chemistry', 'chemical'],
            'Mathematics': ['math', 'calculus', 'algebra', 'trigonometry', 'precalculus'],
            'Statistics': ['statistics', 'statistical', 'probability'],
            'Economics': ['economics', 'microeconomics', 'macroeconomics'],
            'Psychology': ['psychology', 'psychological'],
            'Sociology': ['sociology', 'social'],
            'Business': ['business', 'management', 'organizational'],
            'Computer Science': ['computer science', 'programming', 'software']
        }
        
        for subject, patterns in subject_patterns.items():
            if any(pattern in text for pattern in patterns):
                return subject
        
        return 'Other'
    
    def _discover_openstax_pdfs(self) -> List[Dict[str, Any]]:
        """Auto-discover PDF books from OpenStax website."""
        logger.info("Auto-discovering OpenStax PDFs")
        
        books = []
        
        try:
            # OpenStax subjects pages to scrape
            subject_pages = [
                f"{self.config.openstax_base_url}/subjects/math",
                f"{self.config.openstax_base_url}/subjects/science", 
                f"{self.config.openstax_base_url}/subjects/social-sciences",
                f"{self.config.openstax_base_url}/subjects/humanities",
                f"{self.config.openstax_base_url}/subjects/business"
            ]
            
            for page_url in subject_pages:
                time.sleep(self.config.request_delay_seconds)
                
                try:
                    response = self._make_request(page_url)
                    if response and response.status_code == 200:
                        pdf_books = self._extract_pdfs_from_page(response.text, page_url)
                        books.extend(pdf_books)
                        
                except Exception as e:
                    logger.warning(f"Error scraping {page_url}: {e}")
                    continue
            
            # Also try the main books page
            time.sleep(self.config.request_delay_seconds)
            try:
                main_books_url = f"{self.config.openstax_base_url}/books"
                response = self._make_request(main_books_url)
                if response and response.status_code == 200:
                    main_pdf_books = self._extract_pdfs_from_page(response.text, main_books_url)
                    books.extend(main_pdf_books)
            except Exception as e:
                logger.warning(f"Error scraping main books page: {e}")
            
        except Exception as e:
            logger.error(f"Error in PDF auto-discovery: {e}")
        
        logger.info(f"Auto-discovered {len(books)} PDF books")
        return books
    
    def _extract_pdfs_from_page(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """Extract PDF download links and book information from HTML page."""
        books = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Look for PDF links with various patterns
            pdf_patterns = [
                'a[href*=".pdf"]',
                'a[href*="download"]',
                'a[href*="pdf"]',
                '.download-link',
                '.pdf-download'
            ]
            
            for pattern in pdf_patterns:
                links = soup.select(pattern)
                
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    
                    # Skip if not actually a PDF link
                    if not (href.endswith('.pdf') or 'pdf' in href.lower() or 'download' in href.lower()):
                        continue
                    
                    # Make absolute URL
                    if href.startswith('/'):
                        pdf_url = self.config.openstax_base_url + href
                    elif not href.startswith('http'):
                        pdf_url = urljoin(base_url, href)
                    else:
                        pdf_url = href
                    
                    # Extract book information
                    book_info = self._extract_book_info_from_pdf_link(link, pdf_url, soup)
                    if book_info:
                        books.append(book_info)
            
            # Also look for book cards/sections that might contain download links
            book_sections = soup.find_all(['div', 'section'], class_=re.compile(r'book|textbook|title'))
            for section in book_sections:
                pdf_link = section.find('a', href=re.compile(r'\.pdf|download'))
                if pdf_link:
                    href = pdf_link.get('href', '')
                    if href.startswith('/'):
                        pdf_url = self.config.openstax_base_url + href
                    else:
                        pdf_url = href
                    
                    book_info = self._extract_book_info_from_pdf_section(section, pdf_url)
                    if book_info:
                        books.append(book_info)
                        
        except ImportError:
            logger.warning("BeautifulSoup not available for PDF discovery - install beautifulsoup4")
        except Exception as e:
            logger.warning(f"Error extracting PDFs from page: {e}")
        
        return books
    
    def _extract_book_info_from_pdf_link(self, link_element, pdf_url: str, soup) -> Optional[Dict[str, Any]]:
        """Extract book information from a PDF link element."""
        try:
            # Try to find the book title from the link text or nearby elements
            title = link_element.get_text(strip=True)
            
            # Look for title in parent elements
            parent = link_element.parent
            for _ in range(3):  # Check up to 3 levels up
                if parent:
                    # Look for heading elements
                    heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if heading:
                        title = heading.get_text(strip=True)
                        break
                    
                    # Look for title class
                    title_elem = parent.find(class_=re.compile(r'title|name|book'))
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        break
                    
                    parent = parent.parent
            
            # Clean up title
            if not title or title.lower() in ['download', 'pdf', 'download pdf']:
                # Extract from URL
                import os
                title = os.path.basename(pdf_url).replace('.pdf', '').replace('-', ' ').title()
            
            # Detect subject and level
            subject = self._detect_subject_from_text(title)
            level = self._detect_level_from_title(title)
            
            return {
                'name': title,
                'format': 'pdf',
                'url': pdf_url,
                'subject': subject,
                'level': level,
                'source': 'openstax_website',
                'type': 'pdf'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting book info from PDF link: {e}")
            return None
    
    def _extract_book_info_from_pdf_section(self, section, pdf_url: str) -> Optional[Dict[str, Any]]:
        """Extract book information from a section containing a PDF link."""
        try:
            # Look for title in the section
            title_elem = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not title_elem:
                title_elem = section.find(class_=re.compile(r'title|name|book'))
            
            title = title_elem.get_text(strip=True) if title_elem else "Unknown Book"
            
            # Look for author information
            author_elem = section.find(class_=re.compile(r'author|by'))
            authors = []
            if author_elem:
                authors = [author_elem.get_text(strip=True)]
            
            # Detect subject and level
            section_text = section.get_text().lower()
            subject = self._detect_subject_from_text(section_text)
            level = self._detect_level_from_title(title + " " + section_text)
            
            return {
                'name': title,
                'authors': authors,
                'format': 'pdf',
                'url': pdf_url,
                'subject': subject,
                'level': level,
                'source': 'openstax_website',
                'type': 'pdf'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting book info from PDF section: {e}")
            return None
    
    def _detect_level_from_title(self, title: str) -> str:
        """Detect educational level from title or text."""
        text = title.lower()
        
        # High school indicators (most specific first)
        high_school_patterns = [
            'high school', 'high-school', 'highschool', 'hs ',
            'ap course', 'ap physics', 'ap biology', 'ap chemistry',
            'pre-algebra', 'prealgebra', 'basic music theory',
            'introductory', 'fundamentals of', 'basics of',
            # URL patterns that suggest high school level
            'physics-web', 'biology-web', 'chemistry-web'
        ]
        
        # Graduate level indicators
        graduate_patterns = [
            'graduate', 'phd', 'doctoral', 'advanced research',
            'signal processing', 'machine learning', 'seismic imaging',
            'compressive sensing', 'signal-and-information-processing'
        ]
        
        # Check for high school indicators
        for pattern in high_school_patterns:
            if pattern in text:
                return 'HighSchool'
        
        # Check for graduate indicators  
        for pattern in graduate_patterns:
            if pattern in text:
                return 'Graduate'
        
        # Default to University level
        return 'University'
    
    def _is_openstax_repository(self, repo: Dict[str, Any]) -> bool:
        """Check if repository appears to be an OpenStax textbook."""
        name = (repo.get('name') or '').lower()
        description = (repo.get('description') or '').lower()
        owner = (repo.get('owner', {}).get('login') or '').lower()
        
        # OpenStax indicators
        openstax_indicators = [
            'openstax', 'cnx', 'connexions', 'rice university',
            'open textbook', 'oer', 'creative commons'
        ]
        
        # Repository naming patterns
        name_patterns = [
            'cnxbook-', 'osbooks-', 'derived-from-osbooks-'
        ]
        
        # Check indicators
        text = f"{name} {description} {owner}"
        has_indicator = any(indicator in text for indicator in openstax_indicators)
        
        # Check naming patterns
        has_pattern = any(name.startswith(pattern) for pattern in name_patterns)
        
        # Check if it's from known OpenStax organizations
        known_orgs = ['cnx-user-books', 'openstax', 'openstax-org']
        is_known_org = owner in known_orgs
        
        return has_indicator or has_pattern or is_known_org
    
    def _deduplicate_books(self, books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate books based on repository name."""
        seen_repos = set()
        unique_books = []
        
        for book in books:
            repo_id = book.get('repo', book.get('name', 'unknown'))
            if repo_id not in seen_repos:
                seen_repos.add(repo_id)
                unique_books.append(book)
        
        return unique_books
    
    def _get_enhanced_known_books(self) -> List[Dict[str, Any]]:
        """Get comprehensive OpenStax catalog based on current 70+ title list."""
        logger.info("Loading comprehensive OpenStax catalog (70+ books)")
        
        # Comprehensive OpenStax catalog as of December 2024 (70+ books)
        # All clone_url entries use SSH URLs (git@github.com:org/repo.git) for authentication
        enhanced_catalog = [
            # Math & Statistics
            {"name": "Algebra & Trigonometry 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-algebra-trigonometry", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-algebra-trigonometry.git"},
            {"name": "Prealgebra 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-prealgebra", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-prealgebra.git"},
            {"name": "Elementary Algebra 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-elementary-algebra", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-elementary-algebra.git"},
            {"name": "Intermediate Algebra 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-intermediate-algebra", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-intermediate-algebra.git"},
            {"name": "College Algebra 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-college-algebra", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-algebra.git"},
            {"name": "College Algebra Corequisite Support", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-college-algebra-corequisite", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-algebra-corequisite.git"},
            {"name": "Precalculus 2e", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-precalculus", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-precalculus.git"},
            {"name": "Contemporary Mathematics", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-contemporary-mathematics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-contemporary-mathematics.git"},
            {"name": "Calculus Volume 1", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-calculus-volume-1", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-calculus-volume-1.git"},
            {"name": "Calculus Volume 2", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-calculus-volume-2", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-calculus-volume-2.git"},
            {"name": "Calculus Volume 3", "subject": "Mathematics", "source": "openstax_catalog", "repo": "osbooks-calculus-volume-3", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-calculus-volume-3.git"},
            {"name": "Introductory Statistics 2e", "subject": "Statistics", "source": "openstax_catalog", "repo": "osbooks-introductory-statistics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introductory-statistics.git"},
            {"name": "Introductory Business Statistics 2e", "subject": "Statistics", "source": "openstax_catalog", "repo": "osbooks-introductory-business-statistics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introductory-business-statistics.git"},
            
            # Science & Health
            {"name": "Anatomy & Physiology 2e", "subject": "Biology", "source": "openstax_catalog", "repo": "osbooks-anatomy-physiology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-anatomy-physiology.git"},
            {"name": "Astronomy 2e", "subject": "Physics", "source": "openstax_catalog", "repo": "osbooks-astronomy", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-astronomy.git"},
            {"name": "Biology 2e", "subject": "Biology", "source": "openstax_catalog", "repo": "osbooks-biology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-biology.git"},
            {"name": "Concepts of Biology", "subject": "Biology", "source": "openstax_catalog", "repo": "osbooks-concepts-biology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-concepts-biology.git"},
            {"name": "Chemistry 2e", "subject": "Chemistry", "source": "openstax_catalog", "repo": "osbooks-chemistry", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-chemistry.git"},
            {"name": "Chemistry: Atoms First 2e", "subject": "Chemistry", "source": "openstax_catalog", "repo": "osbooks-chemistry-atoms-first", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-chemistry-atoms-first.git"},
            {"name": "College Physics 2e", "subject": "Physics", "source": "openstax_catalog", "repo": "osbooks-college-physics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-physics.git"},
            {"name": "University Physics Volume 1", "subject": "Physics", "source": "openstax_catalog", "repo": "cnxbook-university-physics-volume-1", "org": "cnx-user-books", "format": "git", "clone_url": "git@github.com:cnx-user-books/cnxbook-university-physics-volume-1.git"},
            {"name": "University Physics Volume 2", "subject": "Physics", "source": "openstax_catalog", "repo": "cnxbook-university-physics-volume-2", "org": "cnx-user-books", "format": "git", "clone_url": "git@github.com:cnx-user-books/cnxbook-university-physics-volume-2.git"},
            {"name": "University Physics Volume 3", "subject": "Physics", "source": "openstax_catalog", "repo": "cnxbook-university-physics-volume-3", "org": "cnx-user-books", "format": "git", "clone_url": "git@github.com:cnx-user-books/cnxbook-university-physics-volume-3.git"},
            {"name": "Microbiology", "subject": "Biology", "source": "openstax_catalog", "repo": "osbooks-microbiology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-microbiology.git"},
            {"name": "Organic Chemistry", "subject": "Chemistry", "source": "openstax_catalog", "repo": "osbooks-organic-chemistry", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-organic-chemistry.git"},
            
            # Business & Economics  
            {"name": "Introduction to Business", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-introduction-business", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-business.git"},
            {"name": "Entrepreneurship", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-entrepreneurship", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-entrepreneurship.git"},
            {"name": "Financial Accounting", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-financial-accounting", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-financial-accounting.git"},
            {"name": "Managerial Accounting", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-managerial-accounting", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-managerial-accounting.git"},
            {"name": "Principles of Finance", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-principles-finance", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-finance.git"},
            {"name": "Principles of Management", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-principles-management", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-management.git"},
            {"name": "Principles of Marketing", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-principles-marketing", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-marketing.git"},
            {"name": "Business Ethics", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-business-ethics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-business-ethics.git"},
            {"name": "Business Law I Essentials", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-business-law-essentials", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-business-law-essentials.git"},
            {"name": "Introduction to Intellectual Property", "subject": "Business", "source": "openstax_catalog", "repo": "osbooks-introduction-intellectual-property", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-intellectual-property.git"},
            {"name": "Principles of Economics 3e", "subject": "Economics", "source": "openstax_catalog", "repo": "osbooks-principles-economics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-economics.git"},
            {"name": "Principles of Microeconomics 3e", "subject": "Economics", "source": "openstax_catalog", "repo": "osbooks-principles-microeconomics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-microeconomics.git"},
            {"name": "Principles of Macroeconomics 3e", "subject": "Economics", "source": "openstax_catalog", "repo": "osbooks-principles-macroeconomics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-macroeconomics.git"},
            {"name": "Principles of Microeconomics 2e AP", "subject": "Economics", "source": "openstax_catalog", "repo": "osbooks-principles-microeconomics-ap", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-microeconomics-ap.git"},
            {"name": "Principles of Macroeconomics 2e AP", "subject": "Economics", "source": "openstax_catalog", "repo": "osbooks-principles-macroeconomics-ap", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-principles-macroeconomics-ap.git"},
            
            # Social Sciences
            {"name": "American Government 3e", "subject": "Political Science", "source": "openstax_catalog", "repo": "osbooks-american-government", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-american-government.git"},
            {"name": "Introduction to Sociology 3e", "subject": "Sociology", "source": "openstax_catalog", "repo": "osbooks-introduction-sociology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-sociology.git"},
            {"name": "Introduction to Anthropology", "subject": "Anthropology", "source": "openstax_catalog", "repo": "osbooks-introduction-anthropology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-anthropology.git"},
            {"name": "Introduction to Political Science", "subject": "Political Science", "source": "openstax_catalog", "repo": "osbooks-introduction-political-science", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-political-science.git"},
            {"name": "Psychology 2e", "subject": "Psychology", "source": "openstax_catalog", "repo": "osbooks-psychology", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-psychology.git"},
            
            # Humanities & College Success
            {"name": "Introduction to Philosophy", "subject": "Philosophy", "source": "openstax_catalog", "repo": "osbooks-introduction-philosophy", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-introduction-philosophy.git"},
            {"name": "U.S. History", "subject": "History", "source": "openstax_catalog", "repo": "osbooks-us-history", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-us-history.git"},
            {"name": "World History Volume 1", "subject": "History", "source": "openstax_catalog", "repo": "osbooks-world-history-volume-1", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-world-history-volume-1.git"},
            {"name": "World History Volume 2", "subject": "History", "source": "openstax_catalog", "repo": "osbooks-world-history-volume-2", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-world-history-volume-2.git"},
            {"name": "Writing Guide with Handbook", "subject": "English", "source": "openstax_catalog", "repo": "osbooks-writing-guide", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-writing-guide.git"},
            {"name": "College Success", "subject": "Study Skills", "source": "openstax_catalog", "repo": "osbooks-college-success", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-success.git"},
            {"name": "College Success Concise", "subject": "Study Skills", "source": "openstax_catalog", "repo": "osbooks-college-success-concise", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-success-concise.git"},
            
            # High-School / AP & K-12
            {"name": "Biology for AP Courses", "subject": "Biology", "source": "openstax_catalog", "repo": "osbooks-biology-ap", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-biology-ap.git"},
            {"name": "College Physics for AP Courses 2e", "subject": "Physics", "source": "openstax_catalog", "repo": "osbooks-college-physics-ap", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-college-physics-ap.git"},
            {"name": "High School Physics", "subject": "Physics", "source": "openstax_catalog", "repo": "osbooks-high-school-physics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-high-school-physics.git"},
            {"name": "Physics (High School PDF)", "subject": "Physics", "source": "openstax_website", "format": "pdf", "url": "https://assets.openstax.org/oscms-prodcms/media/documents/Physics-WEB_Sab7RrQ.pdf", "type": "pdf", "level": "HighSchool"},
            {"name": "High School Statistics", "subject": "Statistics", "source": "openstax_catalog", "repo": "osbooks-high-school-statistics", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-high-school-statistics.git"},
            {"name": "Preparing for College Success", "subject": "Study Skills", "source": "openstax_catalog", "repo": "osbooks-preparing-college-success", "org": "openstax", "format": "git", "clone_url": "git@github.com:openstax/osbooks-preparing-college-success.git"},
        ]
        
        logger.info(f"Enhanced catalog contains {len(enhanced_catalog)} OpenStax books")
        return enhanced_catalog
    
    def discover_pdf_sources(self, book_name: str) -> List[str]:
        """
        Discover PDF download sources for a given book name.
        
        This method would search OpenStax website and other sources
        for direct PDF download links.
        """
        logger.info(f"Discovering PDF sources for: {book_name}")
        
        # This would be implemented to search OpenStax website
        # for PDF download links. For now, return empty list.
        
        return []
    def _filter_invalid_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter to include ONLY OpenStax repositories.
        
        This ensures only official OpenStax textbooks are included in the collection.
        """
        valid_repos = []
        
        for repo in repositories:
            clone_url = repo.get('clone_url', '')
            repo_name = repo.get('repo', '')
            
            # ONLY allow OpenStax repositories
            is_openstax = False
            
            # Check for OpenStax organization
            if 'openstax' in clone_url.lower():
                is_openstax = True
            
            # Check for osbooks- prefix (OpenStax book naming convention)
            elif repo_name.startswith('osbooks-'):
                is_openstax = True
            
            # Allow through if it's OpenStax
            if is_openstax:
                valid_repos.append(repo)
                logger.debug(f"Including OpenStax repository: {repo_name}")
            else:
                logger.debug(f"Filtering out non-OpenStax repository: {repo_name} ({clone_url})")
        
        # Add OpenAlex classification to valid repositories
        if valid_repos:
            valid_repos = self._add_openalex_classification(valid_repos)
        
        logger.info(f"OpenStax-only filter: {len(repositories)} -> {len(valid_repos)} repositories")
        return valid_repos
    
    def _add_openalex_classification(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add OpenAlex discipline classification to discovered repositories.
        
        Args:
            repositories: List of repository dictionaries
            
        Returns:
            Repositories enhanced with OpenAlex classification metadata
        """
        try:
            from .openalex_disciplines import get_hierarchy
            hierarchy = get_hierarchy()
            
            for repo in repositories:
                repo_name = repo.get('repo', '')
                
                # Extract subject from repository name
                subject = self._extract_subject_from_repo_name(repo_name)
                
                # Classify with OpenAlex
                field = hierarchy.classify_subject(subject)
                if field:
                    domain = hierarchy.get_domain(field.parent_id)
                    repo['openalex_classification'] = {
                        'field': field.display_name,
                        'field_id': field.id,
                        'domain': domain.display_name if domain else 'Unknown',
                        'subject': subject
                    }
                    logger.debug(f"Classified {repo_name}: {subject} → {field.display_name}")
                else:
                    repo['openalex_classification'] = {
                        'field': 'Uncategorized',
                        'field_id': 'uncategorized',
                        'domain': 'Uncategorized',
                        'subject': subject
                    }
                    logger.debug(f"Could not classify {repo_name} with subject: {subject}")
        
        except ImportError:
            logger.warning("OpenAlex disciplines module not available - skipping classification")
        except Exception as e:
            logger.warning(f"Error adding OpenAlex classification: {e}")
        
        return repositories
    
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
        
        # Subject extraction patterns
        subject_mappings = {
            'physics': 'physics',
            'college-physics': 'physics',
            'university-physics': 'physics',
            'astronomy': 'astronomy',
            'biology': 'biology',
            'anatomy': 'anatomy',
            'physiology': 'physiology',
            'microbiology': 'microbiology',
            'chemistry': 'chemistry',
            'organic-chemistry': 'chemistry',
            'psychology': 'psychology',
            'sociology': 'sociology',
            'introduction-sociology': 'sociology',
            'business': 'business',
            'entrepreneurship': 'business',
            'principles-finance': 'finance',
            'principles-marketing': 'marketing',
            'introduction-business': 'business',
            'business-ethics': 'business',
            'introduction-intellectual-property': 'business',
            'economics': 'economics',
            'principles-economics': 'economics',
            'mathematics': 'mathematics',
            'contemporary-mathematics': 'mathematics',
            'prealgebra': 'mathematics',
            'introductory-statistics': 'statistics',
            'american-government': 'political-science',
            'introduction-political-science': 'political-science',
            'us-history': 'history',
            'writing-guide': 'english',
            'introduction-philosophy': 'philosophy',
            'introduction-anthropology': 'anthropology',
            'college-success': 'study-skills'
        }
        
        # Find best match
        for pattern, subject in subject_mappings.items():
            if pattern in name:
                return subject
        
        # Fallback to first significant word
        words = name.replace('-', ' ').split()
        if words:
            return words[0]
        
        return 'unknown'