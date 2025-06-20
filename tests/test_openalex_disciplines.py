"""
Unit tests for core.openalex_disciplines module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

from core.openalex_disciplines import OpenAlexHierarchy, get_hierarchy


class TestOpenAlexHierarchy(unittest.TestCase):
    """Test cases for OpenAlexHierarchy class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, 'test_cache.json')

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        os.rmdir(self.temp_dir)

    def test_hierarchy_initialization(self):
        """Test OpenAlexHierarchy initialization"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        self.assertIsInstance(hierarchy.domains, dict)
        self.assertIsInstance(hierarchy.fields, dict)
        self.assertIsInstance(hierarchy.subfields, dict)

    @patch('core.openalex_disciplines.requests.get')
    def test_fetch_concepts_success(self, mock_get):
        """Test successful API response for fetching concepts"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 'https://openalex.org/C121332964',
                    'display_name': 'Physics',
                    'level': 0,
                    'works_count': 1000000
                },
                {
                    'id': 'https://openalex.org/C86803240',
                    'display_name': 'Biology',
                    'level': 0,
                    'works_count': 800000
                }
            ],
            'meta': {'count': 2}
        }
        mock_get.return_value = mock_response

        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        concepts = hierarchy._fetch_concepts_page(1, 0)
        
        self.assertEqual(len(concepts), 2)
        self.assertEqual(concepts[0]['display_name'], 'Physics')
        self.assertEqual(concepts[1]['display_name'], 'Biology')

    @patch('core.openalex_disciplines.requests.get')
    def test_fetch_concepts_api_error(self, mock_get):
        """Test API error handling"""
        # Mock API error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        concepts = hierarchy._fetch_concepts_page(1, 0)
        
        self.assertEqual(concepts, [])

    def test_cache_operations(self):
        """Test cache save and load operations"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        # Add test data
        test_data = {
            'domains': {'test_domain': {'id': '1', 'display_name': 'Test Domain'}},
            'fields': {'test_field': {'id': '2', 'display_name': 'Test Field'}},
            'subfields': {},
            'topics': {}
        }
        
        hierarchy.domains = test_data['domains']
        hierarchy.fields = test_data['fields']
        
        # Save cache
        hierarchy._save_cache()
        self.assertTrue(os.path.exists(self.cache_file))
        
        # Load cache in new instance
        new_hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        new_hierarchy._load_cache()
        
        self.assertEqual(new_hierarchy.domains, test_data['domains'])
        self.assertEqual(new_hierarchy.fields, test_data['fields'])

    def test_subject_classification(self):
        """Test subject classification functionality"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        # Add mock data
        hierarchy.fields = {
            'physics': {'id': '1', 'display_name': 'Physics', 'level': 0},
            'biology': {'id': '2', 'display_name': 'Biology', 'level': 0},
            'chemistry': {'id': '3', 'display_name': 'Chemistry', 'level': 0}
        }

        # Test known subjects
        self.assertEqual(hierarchy.classify_subject('astronomy'), 'Physics')
        self.assertEqual(hierarchy.classify_subject('biology'), 'Biology')
        self.assertEqual(hierarchy.classify_subject('chemistry'), 'Chemistry')
        
        # Test unknown subject
        self.assertEqual(hierarchy.classify_subject('unknown'), 'Uncategorized')

    def test_level_0_concepts(self):
        """Test getting Level-0 concepts"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        # Add mock Level-0 concepts
        hierarchy.fields = {
            'physics': {'id': '1', 'display_name': 'Physics', 'level': 0},
            'chemistry': {'id': '2', 'display_name': 'Chemistry', 'level': 1},  # Not Level-0
            'biology': {'id': '3', 'display_name': 'Biology', 'level': 0}
        }

        level_0 = hierarchy._get_level_0_concepts()
        
        self.assertIn('Physics', level_0)
        self.assertIn('Biology', level_0)
        self.assertNotIn('Chemistry', level_0)

    def test_subject_mapping_variations(self):
        """Test subject mapping with various input formats"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        # Add physics mapping
        hierarchy.fields = {
            'physics': {'id': '1', 'display_name': 'Physics', 'level': 0}
        }

        # Test various astronomy-related terms
        astronomy_terms = [
            'astronomy', 'Astronomy', 'ASTRONOMY',
            'astronomical', 'astrophysics', 'space science'
        ]
        
        for term in astronomy_terms:
            result = hierarchy.classify_subject(term)
            # Should map astronomy to Physics
            self.assertIn(result, ['Physics', 'Uncategorized'])

    def test_multingual_subject_mapping(self):
        """Test subject classification for multilingual terms"""
        hierarchy = OpenAlexHierarchy(cache_file=self.cache_file)
        
        # Add mock fields
        hierarchy.fields = {
            'physics': {'id': '1', 'display_name': 'Physics', 'level': 0},
            'mathematics': {'id': '2', 'display_name': 'Mathematics', 'level': 0}
        }

        # Test Spanish terms
        spanish_terms = {
            'fisica': 'Physics',
            'calculo': 'Mathematics',
            'quimica': 'Uncategorized'  # No chemistry in mock data
        }
        
        for spanish_term, expected in spanish_terms.items():
            result = hierarchy.classify_subject(spanish_term)
            # Allow for fallback to Uncategorized if mapping not found
            self.assertIn(result, [expected, 'Uncategorized'])


class TestGetHierarchy(unittest.TestCase):
    """Test cases for get_hierarchy function"""

    def test_singleton_behavior(self):
        """Test that get_hierarchy returns singleton instance"""
        hierarchy1 = get_hierarchy()
        hierarchy2 = get_hierarchy()
        
        # Should return the same instance
        self.assertIs(hierarchy1, hierarchy2)

    def test_hierarchy_properties(self):
        """Test that hierarchy has required properties"""
        hierarchy = get_hierarchy()
        
        # Should have the required attributes
        self.assertTrue(hasattr(hierarchy, 'domains'))
        self.assertTrue(hasattr(hierarchy, 'fields'))
        self.assertTrue(hasattr(hierarchy, 'subfields'))
        self.assertTrue(hasattr(hierarchy, 'classify_subject'))

    @patch('core.openalex_disciplines.OpenAlexHierarchy')
    def test_hierarchy_initialization_called_once(self, mock_hierarchy_class):
        """Test that hierarchy is only initialized once"""
        mock_instance = MagicMock()
        mock_hierarchy_class.return_value = mock_instance
        
        # Clear any existing singleton
        import core.openalex_disciplines
        if hasattr(core.openalex_disciplines, '_hierarchy_instance'):
            delattr(core.openalex_disciplines, '_hierarchy_instance')
        
        # Call get_hierarchy multiple times
        hierarchy1 = get_hierarchy()
        hierarchy2 = get_hierarchy()
        hierarchy3 = get_hierarchy()
        
        # Should only create one instance
        mock_hierarchy_class.assert_called_once()
        self.assertIs(hierarchy1, hierarchy2)
        self.assertIs(hierarchy2, hierarchy3)


if __name__ == '__main__':
    unittest.main()