"""
Repository tracking system to prevent duplicate clones and manage repository inventory.

This module maintains a master list of all cloned repositories and prevents
duplicate cloning operations that caused recursive nested repository structures.
"""

import os
import json
import logging
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime

from .config import OpenBooksConfig

logger = logging.getLogger(__name__)


@dataclass
class RepositoryRecord:
    """Record for tracking a cloned repository."""
    repo_name: str
    clone_url: str
    local_path: str
    org: str
    subject: str
    discipline: str
    educational_level: str
    size_mb: float
    cloned_at: str
    last_updated: str
    status: str  # 'active', 'failed', 'duplicate', 'moved'
    sha256_hash: str  # Hash of clone_url for fast duplicate detection


class RepositoryTracker:
    """Manages repository inventory to prevent duplicates and track clones."""
    
    def __init__(self, config: OpenBooksConfig):
        """Initialize repository tracker with configuration."""
        self.config = config
        self.inventory_path = Path(config.metadata_path) / "repository_inventory.json"
        self.inventory_path.parent.mkdir(parents=True, exist_ok=True)
        self.repositories: Dict[str, RepositoryRecord] = {}
        self.load_inventory()
    
    def load_inventory(self) -> None:
        """Load existing repository inventory from disk."""
        if not self.inventory_path.exists():
            logger.info("No existing repository inventory found, starting fresh")
            return
        
        try:
            with open(self.inventory_path, 'r') as f:
                data = json.load(f)
            
            self.repositories = {}
            for repo_key, repo_data in data.get('repositories', {}).items():
                record = RepositoryRecord(**repo_data)
                self.repositories[repo_key] = record
            
            logger.info(f"Loaded inventory with {len(self.repositories)} repositories")
            
        except Exception as e:
            logger.error(f"Error loading repository inventory: {e}")
            logger.info("Starting with empty inventory")
            self.repositories = {}
    
    def save_inventory(self) -> None:
        """Save repository inventory to disk."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_repositories': len(self.repositories),
                'repositories': {
                    key: asdict(record) for key, record in self.repositories.items()
                }
            }
            
            with open(self.inventory_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved inventory with {len(self.repositories)} repositories")
            
        except Exception as e:
            logger.error(f"Error saving repository inventory: {e}")
    
    def _generate_repo_key(self, clone_url: str, local_path: str) -> str:
        """Generate a unique key for repository identification."""
        # Use both clone_url and repo name for uniqueness
        repo_name = Path(local_path).name
        key_string = f"{clone_url}:{repo_name}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _generate_url_hash(self, clone_url: str) -> str:
        """Generate SHA256 hash of clone URL for duplicate detection."""
        return hashlib.sha256(clone_url.encode()).hexdigest()
    
    def is_repository_tracked(self, clone_url: str, local_path: str) -> bool:
        """Check if repository is already tracked (prevents duplicates)."""
        repo_key = self._generate_repo_key(clone_url, local_path)
        url_hash = self._generate_url_hash(clone_url)
        
        # Check by exact key match
        if repo_key in self.repositories:
            record = self.repositories[repo_key]
            if record.status == 'active' and Path(record.local_path).exists():
                return True
        
        # Check by local path (for existing repositories)
        for record in self.repositories.values():
            if record.local_path == local_path and record.status == 'active':
                if Path(record.local_path).exists():
                    logger.info(f"Repository already exists at {record.local_path}")
                    # Update the record with the real clone URL if it was a file:// URL
                    if record.clone_url.startswith('file://') and not clone_url.startswith('file://'):
                        record.clone_url = clone_url
                        record.sha256_hash = self._generate_url_hash(clone_url)
                        record.last_updated = datetime.now().isoformat()
                        self.save_inventory()
                        logger.info(f"Updated clone URL for {record.repo_name}")
                    return True
        
        # Check by URL hash (catches same repo cloned to different paths)
        for record in self.repositories.values():
            if record.sha256_hash == url_hash and record.status == 'active':
                if Path(record.local_path).exists():
                    logger.warning(f"Repository already exists at {record.local_path}")
                    return True
        
        return False
    
    def find_duplicate_repositories(self) -> List[Dict[str, Any]]:
        """Find repositories that are duplicates based on clone URL."""
        url_hash_map: Dict[str, List[RepositoryRecord]] = {}
        
        # Group repositories by URL hash
        for record in self.repositories.values():
            if record.status == 'active':
                if record.sha256_hash not in url_hash_map:
                    url_hash_map[record.sha256_hash] = []
                url_hash_map[record.sha256_hash].append(record)
        
        # Find groups with multiple repositories
        duplicates = []
        for url_hash, records in url_hash_map.items():
            if len(records) > 1:
                # Sort by cloned_at to identify primary and duplicates
                records.sort(key=lambda r: r.cloned_at)
                primary = records[0]
                duplicate_records = records[1:]
                
                duplicates.append({
                    'clone_url': primary.clone_url,
                    'primary': {
                        'path': primary.local_path,
                        'cloned_at': primary.cloned_at,
                        'size_mb': primary.size_mb
                    },
                    'duplicates': [
                        {
                            'path': dup.local_path,
                            'cloned_at': dup.cloned_at,
                            'size_mb': dup.size_mb
                        } for dup in duplicate_records
                    ]
                })
        
        return duplicates
    
    def find_nested_repositories(self) -> List[Dict[str, str]]:
        """Find repositories that are nested inside other repositories."""
        nested = []
        
        # Sort paths by length to check shorter paths first
        sorted_records = sorted(
            [(r.local_path, r.repo_name) for r in self.repositories.values() if r.status == 'active'],
            key=lambda x: len(x[0])
        )
        
        for i, (path_a, name_a) in enumerate(sorted_records):
            for path_b, name_b in sorted_records[i+1:]:
                # Check if path_b is nested inside path_a
                if path_b.startswith(path_a + os.sep) and Path(path_b).exists():
                    nested.append({
                        'parent_repo': name_a,
                        'parent_path': path_a,
                        'nested_repo': name_b,
                        'nested_path': path_b
                    })
        
        return nested
    
    def add_repository(self, repo_info: Dict[str, Any], local_path: str, 
                      size_mb: float = 0.0) -> str:
        """Add a successfully cloned repository to the inventory."""
        clone_url = repo_info.get('clone_url', '')
        repo_name = repo_info.get('repo', Path(local_path).name)
        
        repo_key = self._generate_repo_key(clone_url, local_path)
        
        record = RepositoryRecord(
            repo_name=repo_name,
            clone_url=clone_url,
            local_path=local_path,
            org=repo_info.get('org', ''),
            subject=repo_info.get('subject', ''),
            discipline=repo_info.get('discipline', ''),
            educational_level=repo_info.get('educational_level', ''),
            size_mb=size_mb,
            cloned_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            status='active',
            sha256_hash=self._generate_url_hash(clone_url)
        )
        
        self.repositories[repo_key] = record
        self.save_inventory()
        
        logger.info(f"Added repository to inventory: {repo_name}")
        return repo_key
    
    def update_repository_status(self, clone_url: str, local_path: str, 
                                status: str) -> None:
        """Update the status of a repository in the inventory."""
        repo_key = self._generate_repo_key(clone_url, local_path)
        
        if repo_key in self.repositories:
            self.repositories[repo_key].status = status
            self.repositories[repo_key].last_updated = datetime.now().isoformat()
            self.save_inventory()
            logger.debug(f"Updated repository status: {status}")
    
    def remove_repository(self, clone_url: str, local_path: str) -> bool:
        """Remove a repository from the inventory."""
        repo_key = self._generate_repo_key(clone_url, local_path)
        
        if repo_key in self.repositories:
            del self.repositories[repo_key]
            self.save_inventory()
            logger.info(f"Removed repository from inventory: {Path(local_path).name}")
            return True
        return False
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked repositories."""
        total_repos = len(self.repositories)
        active_repos = sum(1 for r in self.repositories.values() if r.status == 'active')
        total_size_mb = sum(r.size_mb for r in self.repositories.values() if r.status == 'active')
        
        # Count by discipline
        discipline_counts = {}
        for record in self.repositories.values():
            if record.status == 'active':
                discipline = record.discipline or 'Unknown'
                discipline_counts[discipline] = discipline_counts.get(discipline, 0) + 1
        
        # Count by educational level
        level_counts = {}
        for record in self.repositories.values():
            if record.status == 'active':
                level = record.educational_level or 'Unknown'
                level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'total_repositories': total_repos,
            'active_repositories': active_repos,
            'total_size_mb': round(total_size_mb, 1),
            'total_size_gb': round(total_size_mb / 1024, 2),
            'by_discipline': discipline_counts,
            'by_educational_level': level_counts,
            'duplicates_found': len(self.find_duplicate_repositories()),
            'nested_repositories': len(self.find_nested_repositories())
        }
    
    def scan_existing_repositories(self) -> Dict[str, Any]:
        """Scan the Books directory to build inventory from existing repositories."""
        logger.info("Scanning existing repositories to build inventory...")
        
        scanned_count = 0
        added_count = 0
        
        # Find all .git directories
        books_path = Path(self.config.books_path)
        for git_dir in books_path.rglob('.git'):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                scanned_count += 1
                
                # Skip if already tracked
                repo_name = repo_path.name
                fake_clone_url = f"file://{repo_path}"  # Temporary URL for existing repos
                
                if not self.is_repository_tracked(fake_clone_url, str(repo_path)):
                    # Try to determine subject/discipline from path
                    path_parts = repo_path.parts
                    subject = 'Unknown'
                    discipline = 'Unknown'
                    educational_level = 'Unknown'
                    
                    # Parse educational directory structure
                    if 'Textbooks' in path_parts:
                        idx = path_parts.index('Textbooks')
                        if idx + 2 < len(path_parts):
                            discipline = path_parts[idx + 1]
                            educational_level = path_parts[idx + 2]
                            subject = discipline
                    
                    # Calculate size
                    size_mb = self._calculate_directory_size(repo_path) / (1024 * 1024)
                    
                    # Create repository info
                    repo_info = {
                        'repo': repo_name,
                        'clone_url': fake_clone_url,
                        'org': 'existing',
                        'subject': subject,
                        'discipline': discipline,
                        'educational_level': educational_level
                    }
                    
                    self.add_repository(repo_info, str(repo_path), size_mb)
                    added_count += 1
        
        logger.info(f"Scanned {scanned_count} repositories, added {added_count} to inventory")
        
        return {
            'scanned': scanned_count,
            'added': added_count,
            'total_in_inventory': len(self.repositories)
        }
    
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating size for {path}: {e}")
        return total_size
    
    def generate_cleanup_plan(self) -> Dict[str, Any]:
        """Generate a plan for cleaning up duplicates and nested repositories."""
        duplicates = self.find_duplicate_repositories()
        nested = self.find_nested_repositories()
        
        cleanup_plan = {
            'duplicate_repositories': len(duplicates),
            'nested_repositories': len(nested),
            'actions': []
        }
        
        # Plan for duplicate cleanup
        for dup_group in duplicates:
            primary_path = dup_group['primary']['path']
            for dup in dup_group['duplicates']:
                cleanup_plan['actions'].append({
                    'action': 'remove_duplicate',
                    'path': dup['path'],
                    'reason': f"Duplicate of {primary_path}",
                    'size_mb': dup['size_mb']
                })
        
        # Plan for nested repository cleanup
        for nested_repo in nested:
            cleanup_plan['actions'].append({
                'action': 'move_nested',
                'path': nested_repo['nested_path'],
                'parent': nested_repo['parent_path'],
                'reason': f"Nested inside {nested_repo['parent_repo']}"
            })
        
        total_cleanup_size = sum(
            action.get('size_mb', 0) for action in cleanup_plan['actions'] 
            if action['action'] == 'remove_duplicate'
        )
        cleanup_plan['total_cleanup_size_mb'] = round(total_cleanup_size, 1)
        cleanup_plan['total_cleanup_size_gb'] = round(total_cleanup_size / 1024, 2)
        
        return cleanup_plan
    
    def get_master_repository_list(self) -> List[Dict[str, Any]]:
        """Get the current master list of all active repositories."""
        return [
            {
                'repo_name': record.repo_name,
                'clone_url': record.clone_url,
                'local_path': record.local_path,
                'org': record.org,
                'subject': record.subject,
                'discipline': record.discipline,
                'educational_level': record.educational_level,
                'size_mb': record.size_mb,
                'cloned_at': record.cloned_at,
                'status': record.status
            }
            for record in self.repositories.values()
            if record.status == 'active'
        ]