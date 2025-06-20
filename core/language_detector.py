"""
Language detection module for OpenStax textbooks.

This module provides comprehensive language detection for OpenStax repositories
using multiple strategies: repository name analysis, content inspection, and
metadata examination.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects language of OpenStax textbook repositories."""
    
    def __init__(self):
        """Initialize language detector with comprehensive mapping."""
        
        # Repository name language indicators
        self.repo_name_indicators = {
            # Spanish indicators
            'spanish': [
                'fisica', 'matematicas', 'quimica', 'biologia', 'psicologia',
                'economia', 'estadistica', 'calculo', 'algebra', 'geometria',
                'universitaria', 'introduccion', 'principios', 'conceptos',
                'anatomia', 'fisiologia', 'microbiologia', 'sociologia',
                'filosofia', 'historia', 'gobierno', 'empresarial', 'contabilidad'
            ],
            # Polish indicators
            'polish': [
                'fizyka', 'matematyka', 'chemia', 'biologia', 'psychologia',
                'ekonomia', 'statystyka', 'socjologia', 'filozofia', 'historia',
                'makroekonomia', 'mikroekonomia', 'rachunkowosc', 'zarzadzanie'
            ],
            # French indicators
            'french': [
                'physique', 'mathematiques', 'chimie', 'biologie', 'psychologie',
                'economie', 'statistiques', 'sociologie', 'philosophie', 'histoire',
                'comptabilite', 'gestion', 'introduction', 'principes'
            ],
            # German indicators
            'german': [
                'physik', 'mathematik', 'chemie', 'biologie', 'psychologie',
                'wirtschaft', 'statistik', 'soziologie', 'philosophie', 'geschichte',
                'buchhaltung', 'management', 'einfuhrung', 'grundlagen'
            ],
            # Italian indicators
            'italian': [
                'fisica', 'matematica', 'chimica', 'biologia', 'psicologia',
                'economia', 'statistica', 'sociologia', 'filosofia', 'storia',
                'contabilita', 'gestione', 'introduzione', 'principi'
            ],
            # Portuguese indicators
            'portuguese': [
                'fisica', 'matematica', 'quimica', 'biologia', 'psicologia',
                'economia', 'estatistica', 'sociologia', 'filosofia', 'historia',
                'contabilidade', 'gestao', 'introducao', 'principios'
            ]
        }
        
        # Content-based language indicators (for file content analysis)
        self.content_indicators = {
            'spanish': [
                'capítulo', 'sección', 'introducción', 'objetivos', 'resumen',
                'ejercicios', 'problemas', 'soluciones', 'bibliografía', 'índice',
                'contenido', 'apéndice', 'referencias', 'tabla', 'figura',
                'ecuación', 'ejemplo', 'definición', 'teorema', 'proposición'
            ],
            'polish': [
                'rozdział', 'sekcja', 'wprowadzenie', 'cele', 'podsumowanie',
                'ćwiczenia', 'problemy', 'rozwiązania', 'bibliografia', 'indeks',
                'zawartość', 'dodatek', 'referencje', 'tabela', 'rysunek',
                'równanie', 'przykład', 'definicja', 'twierdzenie', 'propozycja'
            ],
            'french': [
                'chapitre', 'section', 'introduction', 'objectifs', 'résumé',
                'exercices', 'problèmes', 'solutions', 'bibliographie', 'index',
                'contenu', 'annexe', 'références', 'tableau', 'figure',
                'équation', 'exemple', 'définition', 'théorème', 'proposition'
            ],
            'german': [
                'kapitel', 'abschnitt', 'einführung', 'ziele', 'zusammenfassung',
                'übungen', 'probleme', 'lösungen', 'bibliographie', 'index',
                'inhalt', 'anhang', 'referenzen', 'tabelle', 'abbildung',
                'gleichung', 'beispiel', 'definition', 'theorem', 'satz'
            ],
            'italian': [
                'capitolo', 'sezione', 'introduzione', 'obiettivi', 'riassunto',
                'esercizi', 'problemi', 'soluzioni', 'bibliografia', 'indice',
                'contenuto', 'appendice', 'riferimenti', 'tabella', 'figura',
                'equazione', 'esempio', 'definizione', 'teorema', 'proposizione'
            ],
            'portuguese': [
                'capítulo', 'seção', 'introdução', 'objetivos', 'resumo',
                'exercícios', 'problemas', 'soluções', 'bibliografia', 'índice',
                'conteúdo', 'apêndice', 'referências', 'tabela', 'figura',
                'equação', 'exemplo', 'definição', 'teorema', 'proposição'
            ],
            'english': [
                'chapter', 'section', 'introduction', 'objectives', 'summary',
                'exercises', 'problems', 'solutions', 'bibliography', 'index',
                'content', 'appendix', 'references', 'table', 'figure',
                'equation', 'example', 'definition', 'theorem', 'proposition'
            ]
        }
        
        # Known OpenStax multilingual repositories
        self.known_repos = {
            'spanish': [
                'osbooks-fisica-universitaria-bundle',
                'osbooks-quimica-bundle', 
                'osbooks-matematicas-bundle',
                'osbooks-biologia-bundle',
                'osbooks-economia-bundle',
                'osbooks-estadistica-bundle'
            ],
            'polish': [
                'osbooks-fizyka-bundle',
                'osbooks-makroekonomia-bundle',
                'osbooks-mikroekonomia-bundle',
                'osbooks-psychologia-bundle'
            ]
        }
        
        # File extensions to analyze for content detection
        self.content_file_extensions = {'.md', '.txt', '.xml', '.html', '.tex', '.cnxml'}
        
    def detect_language(self, repo_path: Path, repo_name: str = None) -> str:
        """
        Detect language of a repository using multiple strategies.
        
        Args:
            repo_path: Path to the repository
            repo_name: Repository name (optional, will be extracted from path)
            
        Returns:
            Detected language name (e.g., 'english', 'spanish', 'polish')
        """
        if repo_name is None:
            repo_name = repo_path.name
        
        logger.debug(f"Detecting language for repository: {repo_name}")
        
        # Strategy 1: Check known repositories
        detected_lang = self._check_known_repositories(repo_name)
        if detected_lang:
            logger.debug(f"Language detected from known repos: {detected_lang}")
            return detected_lang
        
        # Strategy 2: Repository name analysis
        detected_lang = self._analyze_repository_name(repo_name)
        if detected_lang:
            logger.debug(f"Language detected from repo name: {detected_lang}")
            return detected_lang
        
        # Strategy 3: Content analysis (if repository exists locally)
        if repo_path.exists():
            detected_lang = self._analyze_content(repo_path)
            if detected_lang:
                logger.debug(f"Language detected from content: {detected_lang}")
                return detected_lang
        
        # Default to English for OpenStax repositories
        logger.debug(f"No specific language detected, defaulting to English for: {repo_name}")
        return 'english'
    
    def _check_known_repositories(self, repo_name: str) -> Optional[str]:
        """Check if repository is in known multilingual repository list."""
        repo_name_lower = repo_name.lower()
        
        for language, repos in self.known_repos.items():
            if any(known_repo.lower() in repo_name_lower for known_repo in repos):
                return language
        
        return None
    
    def _analyze_repository_name(self, repo_name: str) -> Optional[str]:
        """Analyze repository name for language indicators."""
        repo_name_lower = repo_name.lower()
        
        # Remove common prefixes to focus on subject/content words
        for prefix in ['osbooks-', 'cnxbook-', 'derived-from-osbooks-']:
            if repo_name_lower.startswith(prefix):
                repo_name_lower = repo_name_lower[len(prefix):]
                break
        
        # Score each language based on indicators found
        language_scores = {}
        
        for language, indicators in self.repo_name_indicators.items():
            score = 0
            for indicator in indicators:
                if indicator in repo_name_lower:
                    # Longer matches get higher scores
                    score += len(indicator)
            
            if score > 0:
                language_scores[language] = score
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores.keys(), key=lambda k: language_scores[k])
            logger.debug(f"Repository name analysis scores: {language_scores}, best: {best_language}")
            return best_language
        
        return None
    
    def _analyze_content(self, repo_path: Path, max_files: int = 10) -> Optional[str]:
        """Analyze repository content for language indicators."""
        try:
            language_scores = {lang: 0 for lang in self.content_indicators.keys()}
            files_analyzed = 0
            
            # Look for content files to analyze
            for file_path in repo_path.rglob('*'):
                if files_analyzed >= max_files:
                    break
                
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.content_file_extensions and
                    file_path.stat().st_size < 1024 * 1024):  # Skip files larger than 1MB
                    
                    try:
                        content = self._read_file_sample(file_path)
                        if content:
                            self._score_content(content, language_scores)
                            files_analyzed += 1
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
            
            # Find language with highest score
            if any(score > 0 for score in language_scores.values()):
                best_language = max(language_scores.keys(), key=lambda k: language_scores[k])
                logger.debug(f"Content analysis scores: {language_scores}, best: {best_language}")
                return best_language
            
        except Exception as e:
            logger.debug(f"Error in content analysis for {repo_path}: {e}")
        
        return None
    
    def _read_file_sample(self, file_path: Path, max_chars: int = 5000) -> Optional[str]:
        """Read a sample of file content for analysis."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                return content.lower()
        except Exception:
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    content = f.read(max_chars)
                    return content.lower()
            except Exception:
                return None
    
    def _score_content(self, content: str, language_scores: Dict[str, int]) -> None:
        """Score content for each language based on indicator frequency."""
        for language, indicators in self.content_indicators.items():
            for indicator in indicators:
                # Count occurrences of each indicator
                count = len(re.findall(r'\b' + re.escape(indicator) + r'\b', content))
                if count > 0:
                    # Weight by indicator length and frequency
                    language_scores[language] += count * len(indicator)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        languages = set()
        languages.update(self.repo_name_indicators.keys())
        languages.update(self.content_indicators.keys())
        languages.update(self.known_repos.keys())
        languages.add('english')  # Always include English as default
        return sorted(list(languages))
    
    def detect_collection_languages(self, books_path: Path) -> Dict[str, List[str]]:
        """
        Detect languages for all repositories in the collection.
        
        Args:
            books_path: Path to Books directory
            
        Returns:
            Dictionary mapping languages to lists of repository names
        """
        language_repos = {}
        
        if not books_path.exists():
            logger.warning(f"Books directory does not exist: {books_path}")
            return language_repos
        
        # Find all repositories
        for git_dir in books_path.rglob(".git"):
            if git_dir.is_dir():
                repo_path = git_dir.parent
                repo_name = repo_path.name
                
                detected_language = self.detect_language(repo_path, repo_name)
                
                if detected_language not in language_repos:
                    language_repos[detected_language] = []
                
                language_repos[detected_language].append(repo_name)
                logger.info(f"Repository {repo_name} detected as {detected_language}")
        
        return language_repos