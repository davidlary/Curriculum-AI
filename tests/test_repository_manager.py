"""
Unit tests for core.repository_manager module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import subprocess
from pathlib import Path
import shutil

from core.repository_manager import RepositoryManager
from core.config import OpenBooksConfig


class TestRepositoryManager(unittest.TestCase):
    """Test cases for RepositoryManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.config.books_path = self.temp_dir
        self.manager = RepositoryManager(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_manager_initialization(self):
        """Test RepositoryManager initialization"""
        self.assertIsInstance(self.manager.config, OpenBooksConfig)
        self.assertEqual(str(self.manager.books_path), self.temp_dir)
        self.assertTrue(hasattr(self.manager, 'tracker'))

    @patch('core.repository_manager.subprocess.run')
    def test_clone_repository_success(self, mock_run):
        """Test successful repository cloning"""
        # Mock successful clone
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        # Mock tracker methods
        with patch.object(self.manager.tracker, 'is_repository_tracked', return_value=False), \
             patch.object(self.manager.tracker, 'add_repository'), \
             patch.object(self.manager, '_validate_repository_content', return_value=True), \
             patch.object(self.manager, '_get_directory_size', return_value=1024000), \
             patch.object(self.manager, '_log_repository_info'):
            
            book_info = {
                'repo': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'subject': 'astronomy'
            }
            
            result = self.manager.clone_repository(book_info)
            self.assertTrue(result)

    @patch('core.repository_manager.subprocess.run')
    def test_clone_repository_failure(self, mock_run):
        """Test failed repository cloning"""
        # Mock failed clone
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Clone failed"
        mock_run.return_value = mock_result
        
        # Mock tracker methods
        with patch.object(self.manager.tracker, 'is_repository_tracked', return_value=False), \
             patch.object(self.manager.tracker, 'update_repository_status'), \
             patch.object(self.manager, '_validate_repository_content', return_value=True):
            
            book_info = {
                'repo': 'osbooks-test',
                'clone_url': 'https://github.com/openstax/osbooks-test.git',
                'subject': 'test'
            }
            
            result = self.manager.clone_repository(book_info)
            self.assertFalse(result)

    def test_clone_repository_validation_failure(self):
        """Test repository cloning with validation failure"""
        book_info = {
            'repo': 'non-openstax-book',
            'clone_url': 'https://github.com/other/non-openstax-book.git',
            'subject': 'test'
        }
        
        with patch.object(self.manager, '_validate_repository_content', return_value=False):
            result = self.manager.clone_repository(book_info)
            self.assertFalse(result)

    def test_clone_repository_already_tracked(self):
        """Test cloning repository that's already tracked"""
        book_info = {
            'repo': 'osbooks-astronomy',
            'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
            'subject': 'astronomy'
        }
        
        with patch.object(self.manager.tracker, 'is_repository_tracked', return_value=True), \
             patch.object(self.manager, '_validate_repository_content', return_value=True):
            
            result = self.manager.clone_repository(book_info)
            self.assertTrue(result)

    @patch('core.repository_manager.subprocess.run')
    def test_update_repository_success(self, mock_run):
        """Test successful repository update"""
        # Create test repository directory
        repo_path = Path(self.temp_dir) / 'test-repo'
        repo_path.mkdir()
        git_dir = repo_path / '.git'
        git_dir.mkdir()
        
        # Mock git commands
        mock_results = [
            Mock(returncode=0, stdout=""),  # git status
            Mock(returncode=0, stdout=""),  # git fetch
            Mock(returncode=0, stdout="1"),  # git rev-list (1 update available)
            Mock(returncode=0, stdout="")   # git pull
        ]
        mock_run.side_effect = mock_results
        
        with patch.object(self.manager, '_check_git_lfs', return_value=False):
            result = self.manager.update_repository(str(repo_path))
            self.assertTrue(result)

    @patch('core.repository_manager.subprocess.run')
    def test_update_repository_uncommitted_changes(self, mock_run):
        """Test repository update with uncommitted changes"""
        # Create test repository directory
        repo_path = Path(self.temp_dir) / 'test-repo'
        repo_path.mkdir()
        git_dir = repo_path / '.git'
        git_dir.mkdir()
        
        # Mock git status showing uncommitted changes
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "M modified_file.txt"
        mock_run.return_value = mock_result
        
        result = self.manager.update_repository(str(repo_path))
        self.assertFalse(result)

    def test_update_repository_not_git(self):
        """Test updating non-git directory"""
        # Create test directory without .git
        repo_path = Path(self.temp_dir) / 'not-git'
        repo_path.mkdir()
        
        result = self.manager.update_repository(str(repo_path))
        self.assertFalse(result)

    def test_get_git_env(self):
        """Test git environment configuration"""
        env = self.manager._get_git_env()
        
        self.assertIn('GIT_TERMINAL_PROMPT', env)
        self.assertEqual(env['GIT_TERMINAL_PROMPT'], '0')
        self.assertIn('GIT_ASKPASS', env)
        self.assertEqual(env['GIT_ASKPASS'], 'echo')

    @patch('core.repository_manager.subprocess.run')
    def test_check_git_lfs_available(self, mock_run):
        """Test checking Git LFS availability when available"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = self.manager._check_git_lfs()
        self.assertTrue(result)

    @patch('core.repository_manager.subprocess.run')
    def test_check_git_lfs_unavailable(self, mock_run):
        """Test checking Git LFS availability when unavailable"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        result = self.manager._check_git_lfs()
        self.assertFalse(result)

    def test_get_directory_size(self):
        """Test directory size calculation"""
        # Create test directory with files
        test_dir = Path(self.temp_dir) / 'size-test'
        test_dir.mkdir()
        
        # Create test files
        (test_dir / 'file1.txt').write_text('Hello World')
        (test_dir / 'file2.txt').write_text('Test Content')
        
        size = self.manager._get_directory_size(test_dir)
        self.assertGreater(size, 0)

    def test_get_repository_status(self):
        """Test getting repository status"""
        # Create test repository
        repo_path = Path(self.temp_dir) / 'status-test'
        repo_path.mkdir()
        git_dir = repo_path / '.git'
        git_dir.mkdir()
        
        # Create some test files
        (repo_path / 'test.txt').write_text('test content')
        
        with patch.object(self.manager, '_run_git_command') as mock_git:
            # Mock git status output
            mock_git.return_value = Mock(returncode=0, stdout="")
            
            status = self.manager.get_repository_status(str(repo_path))
            
            self.assertTrue(status['exists'])
            self.assertTrue(status['is_git_repo'])
            self.assertFalse(status['has_changes'])

    def test_list_repositories(self):
        """Test listing repositories"""
        # Create test repositories
        repo1 = Path(self.temp_dir) / 'repo1'
        repo1.mkdir()
        (repo1 / '.git').mkdir()
        
        repo2 = Path(self.temp_dir) / 'repo2'
        repo2.mkdir()
        (repo2 / '.git').mkdir()
        
        # Create non-git directory
        non_git = Path(self.temp_dir) / 'not-git'
        non_git.mkdir()
        
        with patch.object(self.manager, 'get_repository_status') as mock_status:
            mock_status.return_value = {
                'exists': True,
                'is_git_repo': True,
                'size_mb': 1.0,
                'file_count': 5
            }
            
            repos = self.manager.list_repositories()
            
            # Should find only git repositories
            self.assertEqual(len(repos), 2)
            repo_names = [r['name'] for r in repos]
            self.assertIn('repo1', repo_names)
            self.assertIn('repo2', repo_names)

    def test_extract_subject_from_repo_name(self):
        """Test subject extraction from repository names"""
        test_cases = [
            ('osbooks-astronomy', 'astronomy'),
            ('osbooks-biology-bundle', 'biology'),
            ('osbooks-college-physics', 'physics'),
            ('osbooks-chemistry-2e', 'chemistry'),
            ('cnxbook-psychology', 'psychology'),
            ('unknown-repo', 'unknown')
        ]
        
        for repo_name, expected_subject in test_cases:
            subject = self.manager._extract_subject_from_repo_name(repo_name)
            self.assertEqual(subject, expected_subject)

    def test_is_openstax_repository(self):
        """Test OpenStax repository detection"""
        # OpenStax repositories
        openstax_cases = [
            {'repo': 'osbooks-astronomy', 'org': 'openstax'},
            {'repo': 'derived-from-osbooks-chemistry', 'org': 'other'},
            {'repo': 'test-book', 'org': 'openstax', 'source': 'openstax.org'}
        ]
        
        for book_info in openstax_cases:
            result = self.manager._is_openstax_repository(book_info)
            self.assertTrue(result, f"Failed for {book_info}")
        
        # Non-OpenStax repositories
        non_openstax_cases = [
            {'repo': 'random-book', 'org': 'other'},
            {'repo': 'cnxbook-test', 'org': 'cnx'},
        ]
        
        for book_info in non_openstax_cases:
            result = self.manager._is_openstax_repository(book_info)
            self.assertFalse(result, f"Failed for {book_info}")

    def test_determine_educational_level(self):
        """Test educational level determination"""
        test_cases = [
            ({'repo': 'osbooks-high-school-physics'}, 'HighSchool'),
            ({'repo': 'osbooks-ap-biology'}, 'HighSchool'),
            ({'repo': 'osbooks-prealgebra'}, 'HighSchool'),
            ({'repo': 'osbooks-college-physics'}, 'University'),
            ({'repo': 'osbooks-university-chemistry'}, 'University'),
            ({'repo': 'advanced-signal-processing'}, 'Graduate'),
            ({'repo': 'graduate-mathematics'}, 'Graduate'),
            ({'repo': 'osbooks-biology'}, 'University')  # Default
        ]
        
        for book_info, expected_level in test_cases:
            level = self.manager._determine_educational_level(book_info)
            self.assertEqual(level, expected_level)

    def test_map_subject_to_discipline(self):
        """Test subject to discipline mapping"""
        test_cases = [
            ('physics', 'Physics'),
            ('biology', 'Biology'),
            ('calculus', 'Mathematics'),
            ('computer science', 'Engineering'),
            ('unknown subject', 'Unknown Subject')
        ]
        
        for subject, expected_discipline in test_cases:
            discipline = self.manager._map_subject_to_discipline(subject)
            self.assertEqual(discipline, expected_discipline)

    def test_validate_repository_content_openstax(self):
        """Test repository content validation for OpenStax"""
        # Valid OpenStax repositories
        valid_cases = [
            {
                'repo': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git'
            },
            {
                'repo': 'test-book',
                'clone_url': 'https://github.com/openstax/test-book.git'
            }
        ]
        
        for book_info in valid_cases:
            target_path = Path(self.temp_dir) / 'test'
            result = self.manager._validate_repository_content(book_info, target_path)
            self.assertTrue(result)

    def test_validate_repository_content_non_openstax(self):
        """Test repository content validation for non-OpenStax"""
        # Invalid non-OpenStax repositories
        invalid_cases = [
            {
                'repo': 'random-book',
                'clone_url': 'https://github.com/other/random-book.git'
            },
            {
                'repo': 'cnxbook-test',
                'clone_url': 'https://github.com/cnx/cnxbook-test.git'
            }
        ]
        
        for book_info in invalid_cases:
            target_path = Path(self.temp_dir) / 'test'
            result = self.manager._validate_repository_content(book_info, target_path)
            self.assertFalse(result)

    def test_convert_clone_url_format_ssh(self):
        """Test URL format conversion to SSH"""
        self.config.use_ssh_for_git = True
        manager = RepositoryManager(self.config)
        
        https_url = 'https://github.com/openstax/osbooks-astronomy.git'
        ssh_url = manager._convert_clone_url_format(https_url)
        
        self.assertEqual(ssh_url, 'git@github.com:openstax/osbooks-astronomy.git')

    def test_convert_clone_url_format_https(self):
        """Test URL format conversion to HTTPS"""
        self.config.use_ssh_for_git = False
        manager = RepositoryManager(self.config)
        
        ssh_url = 'git@github.com:openstax/osbooks-astronomy.git'
        https_url = manager._convert_clone_url_format(ssh_url)
        
        self.assertEqual(https_url, 'https://github.com/openstax/osbooks-astronomy.git')

    @patch('core.repository_manager.get_hierarchy')
    def test_get_openalex_target_path(self, mock_get_hierarchy):
        """Test OpenAlex target path generation"""
        # Mock OpenAlex hierarchy
        mock_hierarchy = Mock()
        mock_hierarchy.classify_subject.return_value = 'Physics'
        mock_hierarchy.get_directory_path.return_value = 'english/Physics/University'
        mock_get_hierarchy.return_value = mock_hierarchy
        
        self.config.use_openalex_disciplines = True
        self.config.directory_structure_mode = "openalex"
        manager = RepositoryManager(self.config)
        
        book_info = {
            'repo': 'osbooks-astronomy',
            'subject': 'astronomy'
        }
        
        target_path = manager._get_openalex_target_path(book_info)
        expected_path = Path(self.temp_dir) / 'english' / 'Physics' / 'University' / 'osbooks-astronomy'
        
        self.assertEqual(target_path, expected_path)

    def test_get_legacy_target_path(self):
        """Test legacy target path generation"""
        book_info = {
            'repo': 'osbooks-astronomy',
            'subject': 'Physics'
        }
        
        target_path = self.manager._get_legacy_target_path(book_info)
        expected_path = Path(self.temp_dir) / 'Textbooks' / 'Physics' / 'University' / 'osbooks-astronomy'
        
        self.assertEqual(target_path, expected_path)

    @patch('core.repository_manager.subprocess.run')
    def test_run_git_command(self, mock_run):
        """Test running git commands"""
        repo_path = Path(self.temp_dir) / 'test-repo'
        repo_path.mkdir()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = self.manager._run_git_command(['status'], repo_path, capture_output=True)
        
        self.assertEqual(result.returncode, 0)
        mock_run.assert_called_once()
        
        # Check that git was prepended to command
        called_args = mock_run.call_args[0][0]
        self.assertEqual(called_args[0], 'git')
        self.assertEqual(called_args[1], 'status')


if __name__ == '__main__':
    unittest.main()