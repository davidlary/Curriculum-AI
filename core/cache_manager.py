#!/usr/bin/env python3
"""
Centralized Cache Manager
Handles caching for academic levels, normalized TOCs, and processing results.

This manager provides:
1. Academic level classification caching
2. Normalized TOC result caching  
3. Processing metadata and metrics caching
4. Cache validation and expiration
5. Cross-pipeline cache sharing
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'key': self.key,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary loaded from JSON."""
        return cls(
            key=data['key'],
            data=data['data'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            metadata=data['metadata']
        )

class CacheManager:
    """Centralized cache manager for curriculum processing pipeline."""
    
    def __init__(self, cache_base_dir: Optional[Path] = None):
        self.cache_base_dir = cache_base_dir or Path("Cache")
        
        # Create cache directories
        self.cache_dirs = {
            'academic_levels': self.cache_base_dir / "AcademicLevels",
            'normalized_tocs': self.cache_base_dir / "NormalizedTOCs", 
            'processing_results': self.cache_base_dir / "ProcessingResults",
            'book_metadata': self.cache_base_dir / "BookMetadata",
            'quality_metrics': self.cache_base_dir / "QualityMetrics"
        }
        
        for cache_dir in self.cache_dirs.values():
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.default_ttl = timedelta(hours=24)  # 24 hour default TTL
        self.max_cache_size = 1000  # Max entries per category
        
        logger.info(f"CacheManager initialized with base directory: {self.cache_base_dir}")
    
    def _generate_cache_key(self, discipline: str, language: str, 
                          processing_type: str, **kwargs) -> str:
        """Generate consistent cache key."""
        key_components = [discipline, language, processing_type]
        
        # Add sorted kwargs for consistency
        for k, v in sorted(kwargs.items()):
            key_components.append(f"{k}={v}")
        
        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _get_cache_file(self, cache_type: str, cache_key: str) -> Path:
        """Get cache file path for given type and key."""
        if cache_type not in self.cache_dirs:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        return self.cache_dirs[cache_type] / f"{cache_key}.json"
    
    def cache_academic_levels(self, discipline: str, language: str, 
                            levels_data: Dict[str, Any], 
                            ttl: Optional[timedelta] = None) -> None:
        """Cache academic level classifications for books."""
        cache_key = self._generate_cache_key(discipline, language, "academic_levels")
        
        entry = CacheEntry(
            key=cache_key,
            data=levels_data,
            created_at=datetime.now(),
            expires_at=datetime.now() + (ttl or self.default_ttl),
            metadata={
                'discipline': discipline,
                'language': language,
                'processing_type': 'academic_levels',
                'book_count': len(levels_data.get('books', [])),
                'levels_found': list(set(book.get('academic_level') for book in levels_data.get('books', [])))
            }
        )
        
        self._save_cache_entry('academic_levels', entry)
        logger.info(f"Cached academic levels for {discipline}/{language}: {len(levels_data.get('books', []))} books")
    
    def get_academic_levels(self, discipline: str, language: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached academic level classifications."""
        cache_key = self._generate_cache_key(discipline, language, "academic_levels")
        entry = self._load_cache_entry('academic_levels', cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Found cached academic levels for {discipline}/{language}")
            return entry.data
        
        if entry and entry.is_expired():
            logger.info(f"Academic levels cache expired for {discipline}/{language}")
            self._remove_cache_entry('academic_levels', cache_key)
        
        return None
    
    def cache_normalized_toc(self, discipline: str, language: str, 
                           toc_data: Dict[str, Any], 
                           processing_method: str = "llm_enhanced",
                           ttl: Optional[timedelta] = None) -> None:
        """Cache normalized TOC results."""
        cache_key = self._generate_cache_key(
            discipline, language, "normalized_toc", 
            method=processing_method
        )
        
        entry = CacheEntry(
            key=cache_key,
            data=toc_data,
            created_at=datetime.now(),
            expires_at=datetime.now() + (ttl or self.default_ttl),
            metadata={
                'discipline': discipline,
                'language': language,
                'processing_type': 'normalized_toc',
                'processing_method': processing_method,
                'total_topics': len(toc_data.get('normalized_topics', [])),
                'academic_levels': toc_data.get('academic_levels', []),
                'source_books': toc_data.get('source_books', [])
            }
        )
        
        self._save_cache_entry('normalized_tocs', entry)
        logger.info(f"Cached normalized TOC for {discipline}/{language} ({processing_method}): {len(toc_data.get('normalized_topics', []))} topics")
    
    def get_normalized_toc(self, discipline: str, language: str, 
                          processing_method: str = "llm_enhanced") -> Optional[Dict[str, Any]]:
        """Retrieve cached normalized TOC."""
        cache_key = self._generate_cache_key(
            discipline, language, "normalized_toc", 
            method=processing_method
        )
        entry = self._load_cache_entry('normalized_tocs', cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Found cached normalized TOC for {discipline}/{language} ({processing_method})")
            return entry.data
        
        if entry and entry.is_expired():
            logger.info(f"Normalized TOC cache expired for {discipline}/{language}")
            self._remove_cache_entry('normalized_tocs', cache_key)
        
        return None
    
    def cache_processing_result(self, discipline: str, language: str, 
                              step: str, result_data: Dict[str, Any],
                              ttl: Optional[timedelta] = None) -> None:
        """Cache processing results from pipeline steps."""
        cache_key = self._generate_cache_key(discipline, language, f"step_{step}")
        
        entry = CacheEntry(
            key=cache_key,
            data=result_data,
            created_at=datetime.now(),
            expires_at=datetime.now() + (ttl or self.default_ttl),
            metadata={
                'discipline': discipline,
                'language': language,
                'processing_type': f'step_{step}',
                'step': step,
                'success': result_data.get('success', False),
                'processing_time': result_data.get('processing_time', 0)
            }
        )
        
        self._save_cache_entry('processing_results', entry)
        logger.info(f"Cached step {step} result for {discipline}/{language}")
    
    def get_processing_result(self, discipline: str, language: str, 
                            step: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached processing result."""
        cache_key = self._generate_cache_key(discipline, language, f"step_{step}")
        entry = self._load_cache_entry('processing_results', cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Found cached result for step {step} ({discipline}/{language})")
            return entry.data
        
        if entry and entry.is_expired():
            logger.info(f"Step {step} cache expired for {discipline}/{language}")
            self._remove_cache_entry('processing_results', cache_key)
        
        return None
    
    def cache_book_metadata(self, discipline: str, language: str,
                          books_data: Dict[str, Any],
                          ttl: Optional[timedelta] = None) -> None:
        """Cache book discovery metadata."""
        cache_key = self._generate_cache_key(discipline, language, "book_metadata")
        
        entry = CacheEntry(
            key=cache_key,
            data=books_data,
            created_at=datetime.now(),
            expires_at=datetime.now() + (ttl or self.default_ttl),
            metadata={
                'discipline': discipline,
                'language': language,
                'processing_type': 'book_metadata',
                'total_books': len(books_data.get('books', [])),
                'discovery_timestamp': books_data.get('discovery_timestamp')
            }
        )
        
        self._save_cache_entry('book_metadata', entry)
        logger.info(f"Cached book metadata for {discipline}/{language}: {len(books_data.get('books', []))} books")
    
    def get_book_metadata(self, discipline: str, language: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached book metadata."""
        cache_key = self._generate_cache_key(discipline, language, "book_metadata")
        entry = self._load_cache_entry('book_metadata', cache_key)
        
        if entry and not entry.is_expired():
            logger.info(f"Found cached book metadata for {discipline}/{language}")
            return entry.data
        
        if entry and entry.is_expired():
            logger.info(f"Book metadata cache expired for {discipline}/{language}")
            self._remove_cache_entry('book_metadata', cache_key)
        
        return None
    
    def invalidate_cache(self, discipline: str, language: str, 
                        cache_types: Optional[List[str]] = None) -> None:
        """Invalidate cache entries for a discipline/language."""
        if cache_types is None:
            cache_types = list(self.cache_dirs.keys())
        
        for cache_type in cache_types:
            cache_dir = self.cache_dirs[cache_type]
            pattern = f"{discipline}_{language}_*"
            
            for cache_file in cache_dir.glob(f"*.json"):
                try:
                    entry = self._load_cache_entry(cache_type, cache_file.stem)
                    if entry and entry.metadata.get('discipline') == discipline and \
                       entry.metadata.get('language') == language:
                        cache_file.unlink()
                        logger.info(f"Invalidated {cache_type} cache for {discipline}/{language}")
                except Exception as e:
                    logger.warning(f"Error invalidating cache {cache_file}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}
        
        for cache_type, cache_dir in self.cache_dirs.items():
            cache_files = list(cache_dir.glob("*.json"))
            valid_entries = 0
            expired_entries = 0
            
            for cache_file in cache_files:
                try:
                    entry = self._load_cache_entry(cache_type, cache_file.stem)
                    if entry:
                        if entry.is_expired():
                            expired_entries += 1
                        else:
                            valid_entries += 1
                except Exception:
                    pass
            
            stats[cache_type] = {
                'total_files': len(cache_files),
                'valid_entries': valid_entries,
                'expired_entries': expired_entries
            }
        
        return stats
    
    def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        removed_count = 0
        
        for cache_type, cache_dir in self.cache_dirs.items():
            for cache_file in cache_dir.glob("*.json"):
                try:
                    entry = self._load_cache_entry(cache_type, cache_file.stem)
                    if entry and entry.is_expired():
                        cache_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed expired cache: {cache_file}")
                except Exception as e:
                    logger.warning(f"Error checking cache expiry {cache_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")
    
    def _save_cache_entry(self, cache_type: str, entry: CacheEntry) -> None:
        """Save cache entry to file."""
        cache_file = self._get_cache_file(cache_type, entry.key)
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save cache entry {cache_file}: {e}")
            raise
    
    def _load_cache_entry(self, cache_type: str, cache_key: str) -> Optional[CacheEntry]:
        """Load cache entry from file."""
        cache_file = self._get_cache_file(cache_type, cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CacheEntry.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache entry {cache_file}: {e}")
            return None
    
    def _remove_cache_entry(self, cache_type: str, cache_key: str) -> None:
        """Remove cache entry file."""
        cache_file = self._get_cache_file(cache_type, cache_key)
        
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove cache entry {cache_file}: {e}")

# Global cache manager instance
_global_cache_manager = None

def get_cache_manager(cache_base_dir: Optional[Path] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(cache_base_dir)
    
    return _global_cache_manager

def reset_cache_manager() -> None:
    """Reset global cache manager (useful for testing)."""
    global _global_cache_manager
    _global_cache_manager = None