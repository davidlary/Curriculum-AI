#!/usr/bin/env python3
"""
Unit tests for data_config module.

Tests the data-driven configuration system that replaces hardcoded values
throughout the application.

Test Goals:
- Verify data configuration loading from YAML files
- Test auto-detection of configuration directory
- Validate all configuration accessors return expected data types
- Test graceful handling of missing/invalid configuration files
- Ensure backward compatibility with existing configuration structure
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.data_config import DataConfig, get_data_config, reload_data_config


class TestDataConfig(unittest.TestCase):
    """Test suite for DataConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test configuration files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test configuration files
        self.discovery_config = {
            'strong_indicators': ['test-strong-1', 'test-strong-2'],
            'subject_indicators': ['test-subject-1', 'test-subject-2'],
            'educational_indicators': ['test-edu-1', 'test-edu-2'],
            'exclude_indicators': ['test-exclude-1', 'test-exclude-2'],
            'trusted_organizations': ['test-org-1', 'test-org-2'],
            'subject_search_patterns': ['test-pattern-1', 'test-pattern-2'],
            'quality_thresholds': {
                'min_size_kb': 100,
                'prefer_non_forks': True
            }
        }
        
        self.application_config = {
            'languages': {
                'supported': ['english', 'spanish', 'french'],
                'default': 'english'
            },
            'defaults': {
                'workers': 10,
                'git_only': True,
                'openstax_only': False,
                'check_updates': False,
                'parallel': False,
                'index': False,
                'verbose': True
            },
            'educational_levels': ['TestLevel1', 'TestLevel2'],
            'legacy_disciplines': ['TestDiscipline1', 'TestDiscipline2'],
            'performance': {
                'batch_size': 15,
                'max_workers': 15
            }
        }
        
        # Write test configuration files
        with open(os.path.join(self.test_dir, 'discovery_indicators.yaml'), 'w') as f:
            yaml.dump(self.discovery_config, f)
        
        with open(os.path.join(self.test_dir, 'application_defaults.yaml'), 'w') as f:
            yaml.dump(self.application_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_data_config_initialization(self):
        """Test DataConfig initialization with custom config directory."""
        config = DataConfig(self.test_dir)
        
        # Test that configuration was loaded correctly
        self.assertIsNotNone(config._discovery_indicators)
        self.assertIsNotNone(config._application_defaults)
    
    def test_discovery_indicators_accessors(self):
        """Test discovery indicators accessor methods."""
        config = DataConfig(self.test_dir)
        
        # Test strong indicators
        strong = config.get_strong_indicators()
        self.assertEqual(strong, ['test-strong-1', 'test-strong-2'])
        self.assertIsInstance(strong, list)
        
        # Test subject indicators
        subject = config.get_subject_indicators()
        self.assertEqual(subject, ['test-subject-1', 'test-subject-2'])
        self.assertIsInstance(subject, list)
        
        # Test educational indicators
        educational = config.get_educational_indicators()
        self.assertEqual(educational, ['test-edu-1', 'test-edu-2'])
        self.assertIsInstance(educational, list)
        
        # Test exclude indicators
        exclude = config.get_exclude_indicators()
        self.assertEqual(exclude, ['test-exclude-1', 'test-exclude-2'])
        self.assertIsInstance(exclude, list)
        
        # Test trusted organizations
        trusted = config.get_trusted_organizations()
        self.assertEqual(trusted, ['test-org-1', 'test-org-2'])
        self.assertIsInstance(trusted, list)
        
        # Test subject search patterns
        patterns = config.get_subject_search_patterns()
        self.assertEqual(patterns, ['test-pattern-1', 'test-pattern-2'])
        self.assertIsInstance(patterns, list)
        
        # Test quality thresholds
        thresholds = config.get_quality_thresholds()
        self.assertEqual(thresholds['min_size_kb'], 100)
        self.assertEqual(thresholds['prefer_non_forks'], True)
        self.assertIsInstance(thresholds, dict)
    
    def test_application_defaults_accessors(self):
        """Test application defaults accessor methods."""
        config = DataConfig(self.test_dir)
        
        # Test language configuration
        supported = config.get_supported_languages()
        self.assertEqual(supported, ['english', 'spanish', 'french'])
        self.assertIsInstance(supported, list)
        
        default_lang = config.get_default_language()
        self.assertEqual(default_lang, 'english')
        self.assertIsInstance(default_lang, str)
        
        # Test application defaults
        defaults = config.get_application_defaults()
        self.assertEqual(defaults['workers'], 10)
        self.assertEqual(defaults['git_only'], True)
        self.assertIsInstance(defaults, dict)
        
        # Test educational levels
        levels = config.get_educational_levels()
        self.assertEqual(levels, ['TestLevel1', 'TestLevel2'])
        self.assertIsInstance(levels, list)
        
        # Test legacy disciplines
        disciplines = config.get_legacy_disciplines()
        self.assertEqual(disciplines, ['TestDiscipline1', 'TestDiscipline2'])
        self.assertIsInstance(disciplines, list)
        
        # Test performance configuration
        performance = config.get_performance_config()
        self.assertEqual(performance['batch_size'], 15)
        self.assertEqual(performance['max_workers'], 15)
        self.assertIsInstance(performance, dict)
    
    def test_utility_methods(self):
        """Test utility methods."""
        config = DataConfig(self.test_dir)
        
        # Test default workers
        workers = config.get_default_workers()
        self.assertEqual(workers, 10)
        self.assertIsInstance(workers, int)
        
        # Test default git-only
        git_only = config.get_default_git_only()
        self.assertEqual(git_only, True)
        self.assertIsInstance(git_only, bool)
        
        # Test default openstax-only
        openstax_only = config.get_default_openstax_only()
        self.assertEqual(openstax_only, False)
        self.assertIsInstance(openstax_only, bool)
        
        # Test trusted organization check
        self.assertTrue(config.is_trusted_organization('test-org-1'))
        self.assertTrue(config.is_trusted_organization('TEST-ORG-2'))  # Case insensitive
        self.assertFalse(config.is_trusted_organization('unknown-org'))
    
    def test_missing_configuration_files(self):
        """Test graceful handling of missing configuration files."""
        empty_dir = tempfile.mkdtemp()
        try:
            config = DataConfig(empty_dir)
            
            # Should return empty lists/dicts instead of crashing
            self.assertEqual(config.get_strong_indicators(), [])
            self.assertEqual(config.get_supported_languages(), ['english'])  # Fallback
            self.assertEqual(config.get_application_defaults(), {})
            
        finally:
            import shutil
            shutil.rmtree(empty_dir)
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        invalid_dir = tempfile.mkdtemp()
        try:
            # Create invalid YAML file
            with open(os.path.join(invalid_dir, 'discovery_indicators.yaml'), 'w') as f:
                f.write('invalid: yaml: content:\n  - broken')
            
            config = DataConfig(invalid_dir)
            
            # Should handle invalid YAML gracefully
            self.assertEqual(config.get_strong_indicators(), [])
            
        finally:
            import shutil
            shutil.rmtree(invalid_dir)
    
    def test_config_directory_auto_detection(self):
        """Test automatic detection of configuration directory."""
        with patch('pathlib.Path.exists') as mock_exists:
            # Mock the exists check to simulate finding config directory
            mock_exists.return_value = True
            
            config = DataConfig()
            
            # Should have attempted to detect config directory
            self.assertIsNotNone(config.config_dir)
    
    def test_reload_configurations(self):
        """Test configuration reloading."""
        config = DataConfig(self.test_dir)
        
        # Get initial values
        initial_workers = config.get_default_workers()
        self.assertEqual(initial_workers, 10)
        
        # Modify configuration file
        modified_config = self.application_config.copy()
        modified_config['defaults']['workers'] = 25
        
        with open(os.path.join(self.test_dir, 'application_defaults.yaml'), 'w') as f:
            yaml.dump(modified_config, f)
        
        # Reload configuration
        config.reload_configurations()
        
        # Should reflect new values
        updated_workers = config.get_default_workers()
        self.assertEqual(updated_workers, 25)
    
    def test_global_config_instance(self):
        """Test global configuration instance management."""
        # Test getting global instance
        config1 = get_data_config(self.test_dir)
        config2 = get_data_config()  # Should return same instance
        
        self.assertIs(config1, config2)
        
        # Test reloading global instance
        reload_data_config()
        # Should still work without errors
        config3 = get_data_config()
        self.assertIsNotNone(config3)


class TestDataConfigIntegration(unittest.TestCase):
    """Integration tests for data configuration with other modules."""
    
    def test_config_integration(self):
        """Test integration with OpenBooksConfig."""
        from core.config import OpenBooksConfig
        
        # Should be able to create config without errors
        config = OpenBooksConfig()
        
        # Should have data-driven values
        self.assertIsInstance(config.educational_levels, list)
        self.assertIsInstance(config.disciplines, list)
        self.assertTrue(len(config.educational_levels) > 0)
        self.assertTrue(len(config.disciplines) > 0)
    
    def test_book_discoverer_integration(self):
        """Test integration with BookDiscoverer."""
        from core.config import OpenBooksConfig
        from core.book_discoverer import BookDiscoverer
        
        config = OpenBooksConfig()
        discoverer = BookDiscoverer(config)
        
        # Should have data_config attribute
        self.assertIsNotNone(discoverer.data_config)
        
        # Should be able to get data-driven indicators
        strong_indicators = discoverer.data_config.get_strong_indicators()
        self.assertIsInstance(strong_indicators, list)


if __name__ == '__main__':
    unittest.main()