"""
Data-driven configuration management for OpenBooks project.

This module provides centralized access to all data-driven configuration,
replacing hardcoded values throughout the system.
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataConfig:
    """Central data configuration manager."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize data configuration.
        
        Args:
            config_dir: Directory containing configuration files. 
                       If None, auto-detects from project structure.
        """
        self.config_dir = config_dir or self._detect_config_dir()
        self._discovery_indicators = None
        self._application_defaults = None
        self._load_configurations()
    
    def _detect_config_dir(self) -> str:
        """Auto-detect configuration directory based on project structure."""
        # Start from current file location
        current_dir = Path(__file__).parent
        
        # Look for config directory in parent directories
        for parent in [current_dir.parent, current_dir.parent.parent]:
            config_path = parent / "config"
            if config_path.exists() and config_path.is_dir():
                return str(config_path)
        
        # Fallback to relative path
        fallback_path = os.path.join(os.getcwd(), "config")
        if not os.path.exists(fallback_path):
            os.makedirs(fallback_path, exist_ok=True)
        return fallback_path
    
    def _load_configurations(self):
        """Load all configuration files."""
        try:
            self._discovery_indicators = self._load_yaml_file("discovery_indicators.yaml")
            self._application_defaults = self._load_yaml_file("application_defaults.yaml")
            logger.info("Data configurations loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data configurations: {e}")
            # Initialize with empty dicts to prevent crashes
            self._discovery_indicators = {}
            self._application_defaults = {}
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        file_path = os.path.join(self.config_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            return {}
    
    # Discovery Indicators Configuration
    def get_strong_indicators(self) -> List[str]:
        """Get strong indicators for educational content."""
        return self._discovery_indicators.get('strong_indicators', [])
    
    def get_subject_indicators(self) -> List[str]:
        """Get subject-based indicators."""
        return self._discovery_indicators.get('subject_indicators', [])
    
    def get_educational_indicators(self) -> List[str]:
        """Get educational content indicators."""
        return self._discovery_indicators.get('educational_indicators', [])
    
    def get_exclude_indicators(self) -> List[str]:
        """Get exclusion indicators."""
        return self._discovery_indicators.get('exclude_indicators', [])
    
    def get_trusted_organizations(self) -> List[str]:
        """Get trusted organizations for discovery."""
        return self._discovery_indicators.get('trusted_organizations', [])
    
    def get_subject_search_patterns(self) -> List[str]:
        """Get subject search patterns."""
        return self._discovery_indicators.get('subject_search_patterns', [])
    
    def get_quality_thresholds(self) -> Dict[str, Any]:
        """Get repository quality thresholds."""
        return self._discovery_indicators.get('quality_thresholds', {})
    
    # Application Defaults Configuration
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        languages = self._application_defaults.get('languages', {})
        return languages.get('supported', ['english'])
    
    def get_default_language(self) -> str:
        """Get default language."""
        languages = self._application_defaults.get('languages', {})
        return languages.get('default', 'english')
    
    def get_application_defaults(self) -> Dict[str, Any]:
        """Get application default settings."""
        return self._application_defaults.get('defaults', {})
    
    def get_educational_levels(self) -> List[str]:
        """Get educational levels."""
        return self._application_defaults.get('educational_levels', ['University'])
    
    def get_legacy_disciplines(self) -> List[str]:
        """Get legacy disciplines list."""
        return self._application_defaults.get('legacy_disciplines', [])
    
    def get_pdf_processing_defaults(self) -> Dict[str, Any]:
        """Get PDF processing default settings."""
        return self._application_defaults.get('pdf_processing', {})
    
    def get_paths_config(self) -> Dict[str, str]:
        """Get path configuration."""
        return self._application_defaults.get('paths', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self._application_defaults.get('performance', {})
    
    def get_content_validation_config(self) -> Dict[str, Any]:
        """Get content validation configuration."""
        return self._application_defaults.get('content_validation', {})
    
    def get_rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self._application_defaults.get('rate_limiting', {})
    
    def get_git_config(self) -> Dict[str, Any]:
        """Get git configuration."""
        return self._application_defaults.get('git', {})
    
    def get_source_priority(self) -> Dict[str, int]:
        """Get source priority configuration."""
        return self._application_defaults.get('source_priority', {})
    
    def get_openalex_config(self) -> Dict[str, Any]:
        """Get OpenAlex configuration."""
        return self._application_defaults.get('openalex', {})
    
    def get_api_endpoints(self) -> Dict[str, str]:
        """Get API endpoints configuration."""
        return self._application_defaults.get('api_endpoints', {})
    
    # Utility methods
    def get_default_workers(self) -> int:
        """Get default number of workers."""
        defaults = self.get_application_defaults()
        return defaults.get('workers', 20)
    
    def get_default_git_only(self) -> bool:
        """Get default git-only setting."""
        defaults = self.get_application_defaults()
        return defaults.get('git_only', False)
    
    def get_default_openstax_only(self) -> bool:
        """Get default openstax-only setting."""
        defaults = self.get_application_defaults()
        return defaults.get('openstax_only', True)
    
    def is_trusted_organization(self, org: str) -> bool:
        """Check if organization is trusted."""
        return org.lower() in [o.lower() for o in self.get_trusted_organizations()]
    
    def reload_configurations(self):
        """Reload all configuration files."""
        self._load_configurations()
        logger.info("Configuration files reloaded")


# Global instance for easy access
_global_data_config = None


def get_data_config(config_dir: Optional[str] = None) -> DataConfig:
    """Get global data configuration instance."""
    global _global_data_config
    if _global_data_config is None or config_dir is not None:
        _global_data_config = DataConfig(config_dir)
    return _global_data_config


def reload_data_config():
    """Reload global data configuration."""
    global _global_data_config
    if _global_data_config is not None:
        _global_data_config.reload_configurations()