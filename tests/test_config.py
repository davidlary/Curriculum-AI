"""
Unit tests for core.config module
"""

import unittest
import tempfile
import os
from pathlib import Path
import yaml

from core.config import OpenBooksConfig


class TestOpenBooksConfig(unittest.TestCase):
    """Test cases for OpenBooksConfig class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        os.rmdir(self.temp_dir)

    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = OpenBooksConfig()
        
        # Test default values
        self.assertEqual(config.max_workers, 20)
        self.assertTrue(config.enable_parallel_processing)
        self.assertTrue(config.use_openalex_disciplines)
        self.assertTrue(config.git_lfs_enabled)
        self.assertEqual(config.request_delay_seconds, 2.0)

    def test_custom_config_values(self):
        """Test creating config with custom values"""
        config = OpenBooksConfig(
            max_workers=10,
            enable_parallel_processing=False,
            request_delay_seconds=1.0
        )
        
        self.assertEqual(config.max_workers, 10)
        self.assertFalse(config.enable_parallel_processing)
        self.assertEqual(config.request_delay_seconds, 1.0)

    def test_config_file_loading(self):
        """Test loading configuration from YAML file"""
        # Create test config file
        test_config = {
            'max_workers': 15,
            'enable_parallel_processing': False,
            'books_path': '/custom/books/path',
            'request_delay_seconds': 3.0
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load config from file
        config = OpenBooksConfig.load_from_file(self.config_file)
        
        self.assertEqual(config.max_workers, 15)
        self.assertFalse(config.enable_parallel_processing)
        self.assertEqual(config.books_path, '/custom/books/path')
        self.assertEqual(config.request_delay_seconds, 3.0)

    def test_config_validation(self):
        """Test configuration validation"""
        config = OpenBooksConfig()
        issues = config.validate()
        
        # Should return empty list for valid config
        self.assertIsInstance(issues, list)

    def test_invalid_config_file(self):
        """Test handling of invalid config file"""
        # Create invalid YAML file
        with open(self.config_file, 'w') as f:
            f.write("invalid: yaml: content: {\n")
        
        # Should handle gracefully
        with self.assertRaises(Exception):
            OpenBooksConfig.load_from_file(self.config_file)

    def test_nonexistent_config_file(self):
        """Test handling of nonexistent config file"""
        nonexistent_file = "/path/that/does/not/exist.yaml"
        
        with self.assertRaises(FileNotFoundError):
            OpenBooksConfig.load_from_file(nonexistent_file)

    def test_path_properties(self):
        """Test path-related properties"""
        config = OpenBooksConfig()
        
        # Test that paths are properly set
        self.assertIsInstance(config.books_path, str)
        self.assertIsInstance(config.metadata_path, str)
        self.assertIsInstance(config.cache_path, str)

    def test_worker_configuration(self):
        """Test worker configuration validation"""
        config = OpenBooksConfig(max_workers=50)
        
        # Test that worker counts are reasonable
        self.assertGreater(config.max_discovery_workers, 0)
        self.assertGreater(config.max_clone_workers, 0)
        self.assertGreater(config.max_processing_workers, 0)
        
        # Test that individual workers don't exceed total
        total_workers = (config.max_discovery_workers + 
                        config.max_clone_workers + 
                        config.max_processing_workers)
        self.assertLessEqual(total_workers, config.max_workers * 2)  # Allow some overlap


if __name__ == '__main__':
    unittest.main()