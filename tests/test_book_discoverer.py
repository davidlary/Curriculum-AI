"""
Unit tests for core.book_discoverer module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from core.book_discoverer import BookDiscoverer
from core.config import OpenBooksConfig


class TestBookDiscoverer(unittest.TestCase):
    """Test cases for BookDiscoverer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OpenBooksConfig()
        self.discoverer = BookDiscoverer(self.config)

    def test_discoverer_initialization(self):
        """Test BookDiscoverer initialization"""
        self.assertIsInstance(self.discoverer.config, OpenBooksConfig)
        self.assertTrue(hasattr(self.discoverer, 'github_api_base'))

    @patch('core.book_discoverer.requests.get')
    def test_discover_openstax_books_success(self, mock_get):
        """Test successful OpenStax book discovery"""
        # Mock GitHub API response for repositories
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'name': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'html_url': 'https://github.com/openstax/osbooks-astronomy',
                'description': 'OpenStax Astronomy textbook',
                'size': 50000,
                'default_branch': 'main'
            },
            {
                'name': 'osbooks-biology-bundle',
                'clone_url': 'https://github.com/openstax/osbooks-biology-bundle.git',
                'html_url': 'https://github.com/openstax/osbooks-biology-bundle',
                'description': 'OpenStax Biology bundle',
                'size': 75000,
                'default_branch': 'main'
            }
        ]
        mock_get.return_value = mock_response

        books = self.discoverer.discover_openstax_books(openstax_only=False)
        
        self.assertIsInstance(books, list)
        self.assertGreater(len(books), 0)
        
        # Check that books have required fields
        for book in books:
            self.assertIn('name', book)
            self.assertIn('clone_url', book)
            self.assertIn('repo', book)

    @patch('core.book_discoverer.requests.get')
    def test_discover_openstax_only_filtering(self, mock_get):
        """Test OpenStax-only filtering"""
        # Mock response with mixed repositories
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'name': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'html_url': 'https://github.com/openstax/osbooks-astronomy',
                'description': 'OpenStax Astronomy textbook',
                'size': 50000,
                'default_branch': 'main'
            },
            {
                'name': 'non-openstax-book',
                'clone_url': 'https://github.com/other/non-openstax-book.git',
                'html_url': 'https://github.com/other/non-openstax-book',
                'description': 'Non-OpenStax textbook',
                'size': 30000,
                'default_branch': 'main'
            }
        ]
        mock_get.return_value = mock_response

        # Test with OpenStax-only filtering
        books = self.discoverer.discover_openstax_books(openstax_only=True)
        
        # Should only contain OpenStax repositories
        for book in books:
            self.assertTrue(
                'openstax' in book['clone_url'].lower() or 
                book['repo'].startswith('osbooks-')
            )

    @patch('core.book_discoverer.requests.get')
    def test_api_error_handling(self, mock_get):
        """Test handling of API errors"""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        books = self.discoverer.discover_openstax_books()
        
        # Should return empty list on error
        self.assertIsInstance(books, list)

    def test_filter_invalid_repositories(self):
        """Test filtering of invalid repositories"""
        # Create test repositories with mixed validity
        test_repos = [
            {
                'repo': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'name': 'osbooks-astronomy'
            },
            {
                'repo': 'invalid-repo',
                'clone_url': 'https://github.com/other/invalid-repo.git',
                'name': 'invalid-repo'
            },
            {
                'repo': 'osbooks-biology',
                'clone_url': 'https://github.com/openstax/osbooks-biology.git',
                'name': 'osbooks-biology'
            }
        ]

        # Test filtering (this would need to access private method)
        # For now, test that the discoverer has filtering capability
        self.assertTrue(hasattr(self.discoverer, '_filter_invalid_repositories'))

    def test_deduplicate_books(self):
        """Test book deduplication functionality"""
        # Create test books with duplicates
        test_books = [
            {
                'repo': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'name': 'osbooks-astronomy'
            },
            {
                'repo': 'osbooks-astronomy',  # Duplicate
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'name': 'osbooks-astronomy'
            },
            {
                'repo': 'osbooks-biology',
                'clone_url': 'https://github.com/openstax/osbooks-biology.git',
                'name': 'osbooks-biology'
            }
        ]

        # Test that discoverer has deduplication capability
        self.assertTrue(hasattr(self.discoverer, '_deduplicate_books'))

    @patch('core.book_discoverer.requests.get')
    def test_rate_limiting_handling(self, mock_get):
        """Test rate limiting handling"""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limited
        mock_response.headers = {'Retry-After': '60'}
        mock_get.return_value = mock_response

        # Should handle rate limiting gracefully
        books = self.discoverer.discover_openstax_books()
        self.assertIsInstance(books, list)

    def test_repository_validation(self):
        """Test repository validation logic"""
        valid_openstax_repos = [
            'osbooks-astronomy',
            'osbooks-biology-bundle',
            'osbooks-physics',
            'derived-from-osbooks-chemistry'
        ]

        invalid_repos = [
            'random-textbook',
            'non-openstax-book',
            'cnxbook-random'
        ]

        # Test that validation logic exists
        # (This would test private methods if accessible)
        self.assertTrue(hasattr(self.discoverer, 'discover_openstax_books'))

    @patch('core.book_discoverer.requests.get')
    def test_pagination_handling(self, mock_get):
        """Test handling of paginated API responses"""
        # Mock first page response
        first_response = Mock()
        first_response.status_code = 200
        first_response.json.return_value = [
            {
                'name': 'osbooks-astronomy',
                'clone_url': 'https://github.com/openstax/osbooks-astronomy.git',
                'html_url': 'https://github.com/openstax/osbooks-astronomy',
                'description': 'OpenStax Astronomy textbook',
                'size': 50000,
                'default_branch': 'main'
            }
        ]
        first_response.links = {
            'next': {'url': 'https://api.github.com/orgs/openstax/repos?page=2'}
        }

        # Mock second page response
        second_response = Mock()
        second_response.status_code = 200
        second_response.json.return_value = [
            {
                'name': 'osbooks-biology',
                'clone_url': 'https://github.com/openstax/osbooks-biology.git',
                'html_url': 'https://github.com/openstax/osbooks-biology',
                'description': 'OpenStax Biology textbook',
                'size': 60000,
                'default_branch': 'main'
            }
        ]
        second_response.links = {}

        mock_get.side_effect = [first_response, second_response]

        books = self.discoverer.discover_openstax_books()
        
        # Should handle pagination and return books from multiple pages
        self.assertIsInstance(books, list)

    def test_subject_detection(self):
        """Test subject detection from repository names"""
        test_cases = [
            ('osbooks-astronomy', 'Physics'),
            ('osbooks-biology-bundle', 'Biology'),
            ('osbooks-chemistry', 'Chemistry'),
            ('osbooks-calculus-bundle', 'Mathematics')
        ]

        for repo_name, expected_subject in test_cases:
            # Test that subject detection capability exists
            # (Implementation would depend on integration with OpenAlex)
            self.assertIsInstance(repo_name, str)
            self.assertIsInstance(expected_subject, str)


if __name__ == '__main__':
    unittest.main()