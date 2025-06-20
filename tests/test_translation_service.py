"""
Unit tests for Translation Service module.

Goal: Test translation functionality for converting educational content
to English with discipline-specific terminology standardization.
"""

import pytest
from core.translation_service import TranslationService


class TestTranslationService:
    """Test TranslationService functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = TranslationService()
    
    def test_translator_initialization(self):
        """Goal: Test translation service initialization."""
        assert hasattr(self.translator, 'discipline_translations')
        assert hasattr(self.translator, 'common_academic_terms')
        assert len(self.translator.discipline_translations) > 0
        assert len(self.translator.common_academic_terms) > 0
    
    def test_get_supported_languages(self):
        """Goal: Test getting list of supported languages."""
        languages = self.translator.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert 'spanish' in languages
        assert 'french' in languages
        assert 'german' in languages
    
    def test_get_supported_disciplines(self):
        """Goal: Test getting supported disciplines for a language."""
        disciplines = self.translator.get_supported_disciplines('spanish')
        
        assert isinstance(disciplines, list)
        assert len(disciplines) > 0
        assert 'Physics' in disciplines
        assert 'Mathematics' in disciplines
        assert 'Chemistry' in disciplines
        assert 'Biology' in disciplines
    
    def test_translate_english_passthrough(self):
        """Goal: Test that English text is passed through with standardization."""
        text = "quantum mechanics"
        result = self.translator.translate_to_english(text, "english", "Physics")
        
        assert result == "Quantum Mechanics"  # Should be standardized
    
    def test_translate_spanish_physics(self):
        """Goal: Test translation from Spanish to English for Physics."""
        test_cases = [
            ("mecánica cuántica", "Quantum Mechanics"),
            ("termodinámica", "Thermodynamics"),
            ("electromagnetismo", "Electromagnetism"),
            ("óptica", "Optics"),
            ("energía", "Energy"),
            ("fuerza", "Force")
        ]
        
        for spanish_text, expected_english in test_cases:
            result = self.translator.translate_to_english(spanish_text, "spanish", "Physics")
            assert result == expected_english
    
    def test_translate_french_physics(self):
        """Goal: Test translation from French to English for Physics."""
        test_cases = [
            ("mécanique quantique", "Quantum Mechanics"),
            ("thermodynamique", "Thermodynamics"),
            ("électromagnétisme", "Electromagnetism"),
            ("optique", "Optics"),
            ("énergie", "Energy")
        ]
        
        for french_text, expected_english in test_cases:
            result = self.translator.translate_to_english(french_text, "french", "Physics")
            assert result == expected_english
    
    def test_translate_german_physics(self):
        """Goal: Test translation from German to English for Physics."""
        test_cases = [
            ("quantenmechanik", "Quantum Mechanics"),
            ("thermodynamik", "Thermodynamics"),
            ("elektromagnetismus", "Electromagnetism"),
            ("optik", "Optics"),
            ("energie", "Energy")
        ]
        
        for german_text, expected_english in test_cases:
            result = self.translator.translate_to_english(german_text, "german", "Physics")
            assert result == expected_english
    
    def test_translate_spanish_mathematics(self):
        """Goal: Test translation from Spanish to English for Mathematics."""
        test_cases = [
            ("cálculo", "Calculus"),
            ("álgebra lineal", "Linear Algebra"),
            ("ecuaciones diferenciales", "Differential Equations"),
            ("estadística", "Statistics"),
            ("geometría", "Geometry")
        ]
        
        for spanish_text, expected_english in test_cases:
            result = self.translator.translate_to_english(spanish_text, "spanish", "Mathematics")
            assert result == expected_english
    
    def test_translate_common_academic_terms(self):
        """Goal: Test translation of common academic terms."""
        test_cases = [
            ("capítulo", "spanish", "Chapter"),
            ("sección", "spanish", "Section"), 
            ("introducción", "spanish", "Introduction"),
            ("chapitre", "french", "Chapter"),
            ("section", "french", "Section"),
            ("kapitel", "german", "Chapter"),
            ("abschnitt", "german", "Section")
        ]
        
        for term, language, expected in test_cases:
            result = self.translator.translate_to_english(term, language, "Physics")
            assert result == expected
    
    def test_translate_unknown_term(self):
        """Goal: Test handling of unknown terms."""
        # Should return title-cased version of input
        result = self.translator.translate_to_english("término desconocido", "spanish", "Physics")
        assert result == "Término Desconocido"
    
    def test_translate_unsupported_language(self):
        """Goal: Test handling of unsupported languages."""
        # Should return title-cased version of input
        result = self.translator.translate_to_english("unknown text", "unsupported_lang", "Physics")
        assert result == "Unknown Text"
    
    def test_apply_translations_case_insensitive(self):
        """Goal: Test that translations work regardless of case."""
        # Test with different cases
        test_cases = [
            "mecánica cuántica",  # lowercase
            "Mecánica Cuántica",  # title case
            "MECÁNICA CUÁNTICA",  # uppercase
            "Mecánica CUÁNTICA"   # mixed case
        ]
        
        for text in test_cases:
            result = self.translator.translate_to_english(text, "spanish", "Physics")
            assert result == "Quantum Mechanics"
    
    def test_translate_multiple_terms(self):
        """Goal: Test translation of text with multiple terms."""
        text = "introducción a la mecánica cuántica"
        result = self.translator.translate_to_english(text, "spanish", "Physics")
        
        # Should translate both "introducción" and "mecánica cuántica"
        assert "Introduction" in result
        assert "Quantum Mechanics" in result
    
    def test_standardize_english_terminology(self):
        """Goal: Test English terminology standardization."""
        test_cases = [
            ("newtons laws", "Physics", "Newton's Laws"),
            ("maxwells equations", "Physics", "Maxwell's Equations"),
            ("organic chemistry", "Chemistry", "Organic Chemistry"),
            ("linear algebra", "Mathematics", "Linear Algebra"),
            ("molecular biology", "Biology", "Molecular Biology")
        ]
        
        for text, discipline, expected in test_cases:
            result = self.translator._standardize_english_terminology(text, discipline)
            assert result == expected
    
    def test_get_translations_for_language(self):
        """Goal: Test getting combined translations for language and discipline."""
        translations = self.translator._get_translations_for_language("spanish", "Physics")
        
        assert isinstance(translations, dict)
        assert len(translations) > 0
        
        # Should contain both common terms and physics-specific terms
        assert "capítulo" in translations  # Common term
        assert "mecánica" in translations  # Physics-specific term
        assert translations["capítulo"] == "Chapter"
        assert translations["mecánica"] == "Mechanics"
    
    def test_word_boundary_matching(self):
        """Goal: Test that translation respects word boundaries."""
        # "mecánica" should translate, but "mecánicax" should not
        result1 = self.translator.translate_to_english("mecánica", "spanish", "Physics")
        result2 = self.translator.translate_to_english("mecánicax", "spanish", "Physics")
        
        assert result1 == "Mechanics"
        assert result2 == "Mecánicax"  # Should not translate partial match
    
    def test_longest_match_first(self):
        """Goal: Test that longer terms are matched before shorter ones."""
        # "mecánica cuántica" should be translated as a whole, not as separate words
        result = self.translator.translate_to_english("mecánica cuántica", "spanish", "Physics")
        assert result == "Quantum Mechanics"
        
        # Test that it doesn't translate as "Mechanics Quantum" 
        assert result != "Mechanics Quantum"


if __name__ == '__main__':
    pytest.main([__file__])