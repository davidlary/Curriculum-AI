"""
Shared utilities for the six-level hierarchical curriculum generator.

This module provides common functionality across all curriculum generation steps:
- LLM API client management with retry logic
- Configuration management with environment variables
- Standards document caching system
- Data validation schemas
- Logging utilities with structured output
- File I/O helpers for JSON/cache management
"""

import os
import json
import time
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests
from openai import OpenAI


@dataclass
class CurriculumConfig:
    """Configuration settings for curriculum generation pipeline."""
    
    # LLM Settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 30
    
    # Classification Settings
    elective_frequency_threshold: float = 0.20
    elective_keywords: List[str] = None
    academic_level_priority: List[str] = None
    
    # Hierarchy Settings
    enforce_six_levels: bool = True
    include_foundational_content: bool = True
    foundational_topics: List[str] = None
    
    # Prerequisites Settings
    confidence_threshold: float = 0.7
    max_prerequisite_depth: int = 5
    cycle_detection_enabled: bool = True
    
    # Standards Settings
    standards_cache_days: int = 30
    auto_discover_standards: bool = True
    supported_standards: List[str] = None
    
    # Export Settings
    export_formats: List[str] = None
    include_metadata: bool = True
    validate_exports: bool = True
    
    # Performance Settings
    cache_enabled: bool = True
    cache_directory: str = "cache"
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Initialize default values for list fields."""
        if self.elective_keywords is None:
            self.elective_keywords = [
                "astronomy", "astrophysics", "biophysics", "environmental",
                "nuclear", "particle", "quantum field", "cosmology",
                "nanotechnology", "medical physics", "geophysics"
            ]
        
        if self.academic_level_priority is None:
            self.academic_level_priority = ["high_school", "undergraduate", "graduate"]
        
        if self.foundational_topics is None:
            self.foundational_topics = [
                "Units and Measurements", "Scientific Notation", "Significant Figures",
                "Dimensional Analysis", "Problem-solving Strategies", "Mathematical Prerequisites"
            ]
        
        if self.supported_standards is None:
            self.supported_standards = [
                "MCAT", "IB_HL", "IB_SL", "A_Level", "IGCSE", "ABET", "ISO", "UNESCO"
            ]
        
        if self.export_formats is None:
            self.export_formats = ["tsv", "json", "dot", "duckdb"]


class CurriculumLogger:
    """Enhanced logging utilities with structured output and performance tracking."""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.performance_data = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.info(f"Starting {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            self.logger.warning(f"Timer for {operation} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        self.performance_data[operation] = duration
        self.logger.info(f"Completed {operation} in {duration:.2f}s")
        del self.start_times[operation]
        return duration
    
    def log_performance_summary(self) -> None:
        """Log summary of all performance data."""
        if not self.performance_data:
            return
        
        total_time = sum(self.performance_data.values())
        self.logger.info("Performance Summary:")
        self.logger.info(f"Total time: {total_time:.2f}s")
        
        for operation, duration in sorted(self.performance_data.items(), 
                                        key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100
            self.logger.info(f"  {operation}: {duration:.2f}s ({percentage:.1f}%)")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)


class LLMClient:
    """ChatGPT-4 client with retry logic and caching."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.client = None
        self.cache = CacheManager(config.cache_directory, logger)
        
        if config.openai_api_key:
            self.client = OpenAI(api_key=config.openai_api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()
        else:
            self.logger.warning("No OpenAI API key found. LLM features will be disabled.")
    
    def is_available(self) -> bool:
        """Check if LLM client is available."""
        return self.client is not None
    
    def generate_completion(self, 
                          prompt: str, 
                          system_message: str = "",
                          cache_key: str = None,
                          temperature: float = 0.1) -> Optional[str]:
        """Generate completion with retry logic and optional caching."""
        if not self.is_available():
            self.logger.error("LLM client not available")
            return None
        
        # Check cache first
        if cache_key and self.config.cache_enabled:
            cached_result = self.cache.get_llm_response(cache_key)
            if cached_result:
                self.logger.debug(f"Using cached LLM response for key: {cache_key[:20]}...")
                return cached_result
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"LLM request attempt {attempt + 1}")
                response = self.client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.config.request_timeout
                )
                
                result = response.choices[0].message.content
                
                # Cache successful response
                if cache_key and self.config.cache_enabled:
                    self.cache.store_llm_response(cache_key, result)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(f"All LLM request attempts failed: {e}")
        
        return None


class CacheManager:
    """High-performance caching system for LLM responses and standards documents."""
    
    def __init__(self, cache_dir: str, logger: CurriculumLogger):
        self.cache_dir = Path(cache_dir)
        self.logger = logger
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.cache_dir / "llm").mkdir(exist_ok=True)
        (self.cache_dir / "standards").mkdir(exist_ok=True)
        (self.cache_dir / "data").mkdir(exist_ok=True)
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """Generate cache file path."""
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / cache_type / f"{hashed_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 1) -> bool:
        """Check if cache file is valid and not expired."""
        if not cache_path.exists():
            return False
        
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(days=max_age_days)
    
    def get_llm_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached LLM response."""
        cache_path = self._get_cache_path("llm", cache_key)
        
        if self._is_cache_valid(cache_path, max_age_days=1):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load LLM cache: {e}")
        
        return None
    
    def store_llm_response(self, cache_key: str, response: str) -> None:
        """Store LLM response in cache."""
        cache_path = self._get_cache_path("llm", cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(response, f)
        except Exception as e:
            self.logger.warning(f"Failed to store LLM cache: {e}")
    
    def get_standards_document(self, standard_name: str) -> Optional[Dict]:
        """Retrieve cached standards document."""
        cache_path = self._get_cache_path("standards", standard_name)
        
        if self._is_cache_valid(cache_path, max_age_days=30):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load standards cache: {e}")
        
        return None
    
    def store_standards_document(self, standard_name: str, document: Dict) -> None:
        """Store standards document in cache."""
        cache_path = self._get_cache_path("standards", standard_name)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(document, f)
        except Exception as e:
            self.logger.warning(f"Failed to store standards cache: {e}")
    
    def clear_cache(self, cache_type: str = None) -> None:
        """Clear cache files."""
        if cache_type:
            cache_subdir = self.cache_dir / cache_type
            if cache_subdir.exists():
                for file in cache_subdir.glob("*.pkl"):
                    file.unlink()
                self.logger.info(f"Cleared {cache_type} cache")
        else:
            for subdir in ["llm", "standards", "data"]:
                cache_subdir = self.cache_dir / subdir
                if cache_subdir.exists():
                    for file in cache_subdir.glob("*.pkl"):
                        file.unlink()
            self.logger.info("Cleared all caches")


class DataValidator:
    """Data validation schemas and utilities."""
    
    @staticmethod
    def validate_toc_data(data: Dict) -> Tuple[bool, List[str]]:
        """Validate TOC extraction data format."""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return False, errors
        
        if "tocs" not in data and "tocs_by_level" not in data:
            errors.append("Data must contain 'tocs' or 'tocs_by_level' key")
        
        # Validate book entries
        tocs = data.get("tocs", data.get("tocs_by_level", {}))
        if isinstance(tocs, dict):
            for level, books in tocs.items():
                if not isinstance(books, list):
                    errors.append(f"Level '{level}' must contain a list of books")
                    continue
                
                for i, book in enumerate(books):
                    if not isinstance(book, dict):
                        errors.append(f"Book {i} in level '{level}' must be a dictionary")
                        continue
                    
                    required_fields = ["book_title", "toc_entries"]
                    for field in required_fields:
                        if field not in book:
                            errors.append(f"Book {i} missing required field: {field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_six_level_hierarchy(data: Dict) -> Tuple[bool, List[str]]:
        """Validate six-level hierarchy structure."""
        errors = []
        
        def validate_level(obj, current_level: int, path: str = ""):
            if current_level > 6:
                errors.append(f"Hierarchy exceeds 6 levels at path: {path}")
                return
            
            if current_level == 6:
                # Level 6 must be a list
                if not isinstance(obj, list):
                    errors.append(f"Level 6 must be a list at path: {path}")
                return
            
            if not isinstance(obj, dict):
                errors.append(f"Level {current_level} must be a dict at path: {path}")
                return
            
            for key, value in obj.items():
                new_path = f"{path}/{key}" if path else key
                validate_level(value, current_level + 1, new_path)
        
        if isinstance(data, dict):
            if "core" in data:
                validate_level(data["core"], 1, "core")
            if "electives" in data:
                validate_level(data["electives"], 1, "electives")
        else:
            validate_level(data, 1)
        
        return len(errors) == 0, errors


class FileManager:
    """File I/O utilities with error handling and validation."""
    
    def __init__(self, logger: CurriculumLogger):
        self.logger = logger
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """Load JSON file with error handling."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
        
        return None
    
    def save_json(self, data: Dict, file_path: Union[str, Path], 
                  indent: int = 2) -> bool:
        """Save data to JSON file with error handling."""
        file_path = Path(file_path)
        
        try:
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            self.logger.info(f"Saved data to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {file_path}: {e}")
            return False
    
    def backup_file(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Create backup of existing file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")
        
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return None


def load_config(config_path: str = "config/curriculum_config.json") -> CurriculumConfig:
    """Load configuration from file or environment variables."""
    config = CurriculumConfig()
    
    # Load from file if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")
    
    # Override with environment variables
    if os.getenv("OPENAI_API_KEY"):
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if os.getenv("ELECTIVE_THRESHOLD"):
        try:
            config.elective_frequency_threshold = float(os.getenv("ELECTIVE_THRESHOLD"))
        except ValueError:
            pass
    
    return config


def save_config(config: CurriculumConfig, config_path: str = "config/curriculum_config.json") -> bool:
    """Save configuration to file."""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False


def create_performance_report(performance_data: Dict[str, float], 
                            output_path: str = "performance_report.json") -> bool:
    """Create performance benchmarking report."""
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": sum(performance_data.values()),
            "operations": performance_data,
            "metrics": {
                "fastest_operation": min(performance_data.items(), key=lambda x: x[1])[0],
                "slowest_operation": max(performance_data.items(), key=lambda x: x[1])[0],
                "average_duration": sum(performance_data.values()) / len(performance_data)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to create performance report: {e}")
        return False


# Initialize default logger
default_logger = CurriculumLogger("curriculum_utils")