"""
Unit tests for core.repository_tracker module
"""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
import json
from pathlib import Path
import shutil
from datetime import datetime

from core.repository_tracker import RepositoryTracker, RepositoryRecord
from core.config import OpenBooksConfig


class TestRepositoryRecord(unittest.TestCase):
    """Test cases for RepositoryRecord dataclass"""

    def test_record_creation(self):
        """Test RepositoryRecord creation"""
        record = RepositoryRecord(
            repo_name="test-repo",
            clone_url="https://github.com/test/repo.git",
            local_path="/books/test-repo",
            org="test",
            subject="physics",
            discipline="Physics",
            educational_level="University",
            size_mb=50.5,
            cloned_at="2024-01-01T00:00:00",
            last_updated="2024-01-01T00:00:00",
            status="active",
            sha256_hash="abcd1234"
        )
        
        self.assertEqual(record.repo_name, "test-repo")
        self.assertEqual(record.clone_url, "https://github.com/test/repo.git")
        self.assertEqual(record.local_path, "/books/test-repo")
        self.assertEqual(record.status, "active")
        self.assertEqual(record.size_mb, 50.5)


class TestRepositoryTracker(unittest.TestCase):
    """Test cases for RepositoryTracker class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.config.metadata_path = os.path.join(self.temp_dir, 'metadata')
        self.config.books_path = os.path.join(self.temp_dir, 'books')
        os.makedirs(self.config.metadata_path, exist_ok=True)
        os.makedirs(self.config.books_path, exist_ok=True)
        self.tracker = RepositoryTracker(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_tracker_initialization(self):
        """Test RepositoryTracker initialization"""
        self.assertIsInstance(self.tracker.config, OpenBooksConfig)
        self.assertEqual(len(self.tracker.repositories), 0)
        self.assertTrue(self.tracker.inventory_path.parent.exists())

    def test_generate_repo_key(self):
        """Test repository key generation"""
        clone_url = "https://github.com/test/repo.git"
        local_path = "/books/test-repo"
        
        key1 = self.tracker._generate_repo_key(clone_url, local_path)
        key2 = self.tracker._generate_repo_key(clone_url, local_path)
        
        self.assertEqual(key1, key2)  # Should be deterministic
        self.assertEqual(len(key1), 16)  # Should be 16 characters

    def test_generate_url_hash(self):
        """Test URL hash generation"""
        clone_url = "https://github.com/test/repo.git"
        
        hash1 = self.tracker._generate_url_hash(clone_url)
        hash2 = self.tracker._generate_url_hash(clone_url)
        
        self.assertEqual(hash1, hash2)  # Should be deterministic
        self.assertEqual(len(hash1), 64)  # SHA256 hex string

    def test_add_repository(self):
        """Test adding repository to tracker"""
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test',
            'subject': 'physics',
            'discipline': 'Physics',
            'educational_level': 'University'
        }
        local_path = '/books/test-repo'
        size_mb = 50.5
        
        repo_key = self.tracker.add_repository(repo_info, local_path, size_mb)
        
        self.assertIn(repo_key, self.tracker.repositories)
        
        record = self.tracker.repositories[repo_key]
        self.assertEqual(record.repo_name, 'test-repo')
        self.assertEqual(record.clone_url, 'https://github.com/test/repo.git')
        self.assertEqual(record.local_path, local_path)
        self.assertEqual(record.size_mb, size_mb)
        self.assertEqual(record.status, 'active')

    def test_is_repository_tracked_new_repo(self):
        """Test checking if new repository is tracked"""
        clone_url = "https://github.com/test/repo.git"
        local_path = "/books/test-repo"
        
        is_tracked = self.tracker.is_repository_tracked(clone_url, local_path)
        
        self.assertFalse(is_tracked)

    def test_is_repository_tracked_existing_repo(self):
        """Test checking if existing repository is tracked"""
        # Add repository first
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        local_path = '/books/test-repo'
        
        # Create the local path
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        self.tracker.add_repository(repo_info, local_path)
        
        # Check if tracked
        is_tracked = self.tracker.is_repository_tracked(repo_info['clone_url'], local_path)
        
        self.assertTrue(is_tracked)

    def test_is_repository_tracked_by_path(self):
        """Test tracking by local path when path exists"""
        # Add repository first
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        local_path = os.path.join(self.temp_dir, 'test-repo')
        
        # Create the local path
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        self.tracker.add_repository(repo_info, local_path)
        
        # Check with different URL but same path
        different_url = 'https://github.com/other/repo.git'
        is_tracked = self.tracker.is_repository_tracked(different_url, local_path)
        
        self.assertTrue(is_tracked)

    def test_update_repository_status(self):
        """Test updating repository status"""
        # Add repository first
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        local_path = '/books/test-repo'
        repo_key = self.tracker.add_repository(repo_info, local_path)
        
        # Update status
        self.tracker.update_repository_status(repo_info['clone_url'], local_path, 'failed')
        
        record = self.tracker.repositories[repo_key]
        self.assertEqual(record.status, 'failed')

    def test_remove_repository(self):
        """Test removing repository from tracker"""
        # Add repository first
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        local_path = '/books/test-repo'
        repo_key = self.tracker.add_repository(repo_info, local_path)
        
        # Remove repository
        success = self.tracker.remove_repository(repo_info['clone_url'], local_path)
        
        self.assertTrue(success)
        self.assertNotIn(repo_key, self.tracker.repositories)

    def test_remove_nonexistent_repository(self):
        """Test removing non-existent repository"""
        clone_url = "https://github.com/nonexistent/repo.git"
        local_path = "/books/nonexistent"
        
        success = self.tracker.remove_repository(clone_url, local_path)
        
        self.assertFalse(success)

    def test_find_duplicate_repositories(self):
        """Test finding duplicate repositories"""
        # Add two repositories with same clone URL
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        
        self.tracker.add_repository(repo_info, '/books/test-repo-1')
        self.tracker.add_repository(repo_info, '/books/test-repo-2')
        
        duplicates = self.tracker.find_duplicate_repositories()
        
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(duplicates[0]['duplicates']), 1)
        self.assertEqual(duplicates[0]['clone_url'], repo_info['clone_url'])

    def test_find_nested_repositories(self):
        """Test finding nested repositories"""
        # Add parent repository
        parent_info = {
            'repo': 'parent-repo',
            'clone_url': 'https://github.com/test/parent.git',
            'org': 'test'
        }
        parent_path = '/books/parent-repo'
        self.tracker.add_repository(parent_info, parent_path)
        
        # Add nested repository
        nested_info = {
            'repo': 'nested-repo',
            'clone_url': 'https://github.com/test/nested.git',
            'org': 'test'
        }
        nested_path = '/books/parent-repo/nested-repo'
        
        # Create nested path to simulate existence
        Path(nested_path).mkdir(parents=True, exist_ok=True)
        self.tracker.add_repository(nested_info, nested_path)
        
        nested = self.tracker.find_nested_repositories()
        
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0]['parent_repo'], 'parent-repo')
        self.assertEqual(nested[0]['nested_repo'], 'nested-repo')

    def test_get_repository_stats(self):
        """Test getting repository statistics"""
        # Add some test repositories
        repos = [
            {
                'repo': 'physics-repo',
                'clone_url': 'https://github.com/test/physics.git',
                'org': 'test',
                'discipline': 'Physics',
                'educational_level': 'University'
            },
            {
                'repo': 'biology-repo',
                'clone_url': 'https://github.com/test/biology.git',
                'org': 'test',
                'discipline': 'Biology',
                'educational_level': 'HighSchool'
            }
        ]
        
        for i, repo in enumerate(repos):
            self.tracker.add_repository(repo, f'/books/repo-{i}', 50.0)
        
        stats = self.tracker.get_repository_stats()
        
        self.assertEqual(stats['total_repositories'], 2)
        self.assertEqual(stats['active_repositories'], 2)
        self.assertEqual(stats['total_size_mb'], 100.0)
        self.assertIn('Physics', stats['by_discipline'])
        self.assertIn('Biology', stats['by_discipline'])
        self.assertIn('University', stats['by_educational_level'])
        self.assertIn('HighSchool', stats['by_educational_level'])

    def test_scan_existing_repositories(self):
        """Test scanning existing repositories"""
        # Create test repository structure
        books_path = Path(self.config.books_path)
        
        # Create a git repository
        repo1_path = books_path / 'Textbooks' / 'Physics' / 'University' / 'test-repo1'
        repo1_path.mkdir(parents=True)
        (repo1_path / '.git').mkdir()
        (repo1_path / 'content.txt').write_text('test content')
        
        # Create another git repository
        repo2_path = books_path / 'test-repo2'
        repo2_path.mkdir()
        (repo2_path / '.git').mkdir()
        
        scan_result = self.tracker.scan_existing_repositories()
        
        self.assertEqual(scan_result['scanned'], 2)
        self.assertEqual(scan_result['added'], 2)
        self.assertEqual(len(self.tracker.repositories), 2)

    def test_calculate_directory_size(self):
        """Test calculating directory size"""
        # Create test directory with files
        test_dir = Path(self.temp_dir) / 'size-test'
        test_dir.mkdir()
        
        (test_dir / 'file1.txt').write_text('Hello World' * 100)
        (test_dir / 'file2.txt').write_text('Test Content' * 50)
        
        size = self.tracker._calculate_directory_size(test_dir)
        
        self.assertGreater(size, 0)

    def test_generate_cleanup_plan(self):
        """Test generating cleanup plan"""
        # Add duplicate repositories
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test'
        }
        
        self.tracker.add_repository(repo_info, '/books/test-repo-1', 50.0)
        self.tracker.add_repository(repo_info, '/books/test-repo-2', 30.0)
        
        # Add nested repository
        parent_path = '/books/parent'
        nested_path = '/books/parent/nested'
        Path(nested_path).mkdir(parents=True, exist_ok=True)
        
        parent_info = {'repo': 'parent', 'clone_url': 'https://github.com/test/parent.git', 'org': 'test'}
        nested_info = {'repo': 'nested', 'clone_url': 'https://github.com/test/nested.git', 'org': 'test'}
        
        self.tracker.add_repository(parent_info, parent_path)
        self.tracker.add_repository(nested_info, nested_path)
        
        cleanup_plan = self.tracker.generate_cleanup_plan()
        
        self.assertGreater(cleanup_plan['duplicate_repositories'], 0)
        self.assertGreater(cleanup_plan['nested_repositories'], 0)
        self.assertGreater(len(cleanup_plan['actions']), 0)

    def test_get_master_repository_list(self):
        """Test getting master repository list"""
        # Add some repositories
        repos = [
            {
                'repo': 'repo1',
                'clone_url': 'https://github.com/test/repo1.git',
                'org': 'test',
                'subject': 'physics'
            },
            {
                'repo': 'repo2',
                'clone_url': 'https://github.com/test/repo2.git',
                'org': 'test',
                'subject': 'biology'
            }
        ]
        
        for i, repo in enumerate(repos):
            self.tracker.add_repository(repo, f'/books/repo-{i}')
        
        # Mark one as failed
        self.tracker.update_repository_status(repos[1]['clone_url'], '/books/repo-1', 'failed')
        
        master_list = self.tracker.get_master_repository_list()
        
        # Should only include active repositories
        self.assertEqual(len(master_list), 1)
        self.assertEqual(master_list[0]['repo_name'], 'repo1')

    def test_save_and_load_inventory(self):
        """Test saving and loading inventory"""
        # Add a repository
        repo_info = {
            'repo': 'test-repo',
            'clone_url': 'https://github.com/test/repo.git',
            'org': 'test',
            'subject': 'physics'
        }
        local_path = '/books/test-repo'
        
        self.tracker.add_repository(repo_info, local_path, 50.0)
        
        # Create new tracker instance (should load existing inventory)
        new_tracker = RepositoryTracker(self.config)
        
        self.assertEqual(len(new_tracker.repositories), 1)
        
        # Check that repository data is preserved
        records = list(new_tracker.repositories.values())
        record = records[0]
        self.assertEqual(record.repo_name, 'test-repo')
        self.assertEqual(record.clone_url, 'https://github.com/test/repo.git')
        self.assertEqual(record.size_mb, 50.0)

    def test_load_inventory_error_handling(self):
        """Test inventory loading with corrupted file"""
        # Create corrupted inventory file
        with open(self.tracker.inventory_path, 'w') as f:
            f.write("invalid json content {")
        
        # Create new tracker (should handle error gracefully)
        new_tracker = RepositoryTracker(self.config)
        
        self.assertEqual(len(new_tracker.repositories), 0)

    def test_save_inventory_error_handling(self):
        """Test inventory saving error handling"""
        # Make inventory file read-only to cause save error
        self.tracker.inventory_path.touch()
        self.tracker.inventory_path.chmod(0o444)
        
        try:
            # This should log error but not raise exception
            self.tracker.save_inventory()
        except Exception:
            self.fail("save_inventory() should handle errors gracefully")
        finally:
            # Restore permissions for cleanup
            self.tracker.inventory_path.chmod(0o644)


if __name__ == '__main__':
    unittest.main()