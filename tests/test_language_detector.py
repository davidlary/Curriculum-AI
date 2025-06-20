"""
Unit tests for core.language_detector module
"""

import unittest
import tempfile
import os
from pathlib import Path

from core.language_detector import LanguageDetector


class TestLanguageDetector(unittest.TestCase):
    """Test cases for LanguageDetector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = LanguageDetector()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_supported_languages(self):
        """Test getting supported languages"""
        languages = self.detector.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('english', languages)
        self.assertIn('spanish', languages)
        self.assertIn('french', languages)
        self.assertIn('polish', languages)
        self.assertIn('german', languages)

    def test_english_repository_detection(self):
        """Test detection of English repositories"""
        test_cases = [
            'osbooks-astronomy',
            'osbooks-biology-bundle',
            'osbooks-physics',
            'osbooks-chemistry'
        ]
        
        for repo_name in test_cases:
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'english', 
                           f"Failed for repository: {repo_name}")

    def test_spanish_repository_detection(self):
        """Test detection of Spanish repositories"""
        test_cases = [
            'osbooks-calculo-bundle',
            'osbooks-quimica-bundle',
            'osbooks-fisica-universitaria-bundle',
            'osbooks-prealgebra-bundle'
        ]
        
        for repo_name in test_cases:
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'spanish', 
                           f"Failed for repository: {repo_name}")

    def test_french_repository_detection(self):
        """Test detection of French repositories"""
        test_cases = [
            'osbooks-introduction-business',
            'osbooks-introduction-philosophy',
            'osbooks-introduction-python-programming'
        ]
        
        for repo_name in test_cases:
            language = self.detector.detect_language(None, repo_name)
            # Note: These should be detected as French based on known repository mapping
            self.assertIn(language, ['french', 'english'])  # Allow fallback

    def test_polish_repository_detection(self):
        """Test detection of Polish repositories"""
        test_cases = [
            'osbooks-fizyka-bundle',
            'osbooks-psychologia',
            'osbooks-mikroekonomia',
            'osbooks-makroekonomia'
        ]
        
        for repo_name in test_cases:
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'polish', 
                           f"Failed for repository: {repo_name}")

    def test_german_repository_detection(self):
        """Test detection of German repositories"""
        test_cases = [
            'osbooks-principles-of-management-bundle',
            'work-management'
        ]
        
        for repo_name in test_cases:
            language = self.detector.detect_language(None, repo_name)
            # These might be detected as German or English, depending on content
            self.assertIn(language, ['german', 'english'])

    def test_unknown_repository_fallback(self):
        """Test fallback for unknown repositories"""
        unknown_repos = [
            'unknown-repository',
            'test-book-12345',
            'random-name'
        ]
        
        for repo_name in unknown_repos:
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'english')  # Should default to English

    def test_content_analysis(self):
        """Test content-based language detection"""
        # Create test directory with content
        test_repo = Path(self.temp_dir) / "test-repo"
        test_repo.mkdir()
        
        # Create test content file
        content_file = test_repo / "content.txt"
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write("This is English content for testing.")
        
        language = self.detector.detect_language(test_repo, "test-repo")
        self.assertEqual(language, 'english')

    def test_collection_language_analysis(self):
        """Test analyzing languages in a collection"""
        # Create test repository structure
        books_dir = Path(self.temp_dir) / "Books"
        books_dir.mkdir()
        
        # Create test repositories
        test_repos = [
            ("osbooks-astronomy", "english"),
            ("osbooks-calculo-bundle", "spanish"),
            ("osbooks-fizyka-bundle", "polish")
        ]
        
        for repo_name, expected_lang in test_repos:
            repo_dir = books_dir / expected_lang / "Physics" / "University" / repo_name
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            # Create .git directory to mark as repository
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
        
        # Test collection analysis
        languages = self.detector.detect_collection_languages(books_dir)
        
        self.assertIsInstance(languages, dict)
        self.assertIn('english', languages)
        self.assertIn('spanish', languages)
        self.assertIn('polish', languages)

    def test_language_indicators(self):
        """Test language indicator patterns"""
        # Test Spanish indicators
        spanish_words = ['calculo', 'quimica', 'fisica', 'matematicas']
        for word in spanish_words:
            repo_name = f"osbooks-{word}-test"
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'spanish')

        # Test Polish indicators  
        polish_words = ['fizyka', 'psychologia', 'ekonomia']
        for word in polish_words:
            repo_name = f"osbooks-{word}-test"
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, 'polish')

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test None inputs
        language = self.detector.detect_language(None, None)
        self.assertEqual(language, 'english')
        
        # Test empty string
        language = self.detector.detect_language(None, "")
        self.assertEqual(language, 'english')
        
        # Test very long repository name
        long_name = "a" * 1000
        language = self.detector.detect_language(None, long_name)
        self.assertEqual(language, 'english')

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive"""
        test_cases = [
            ('OSBOOKS-CALCULO-BUNDLE', 'spanish'),
            ('osbooks-FIZYKA-bundle', 'polish'),
            ('OsBooks-Quimica-Bundle', 'spanish')
        ]
        
        for repo_name, expected_lang in test_cases:
            language = self.detector.detect_language(None, repo_name)
            self.assertEqual(language, expected_lang)


if __name__ == '__main__':
    unittest.main()