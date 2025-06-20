#!/usr/bin/env python3
"""
discipline_discoverer.py - Discover disciplines and map to authority sources

This module identifies disciplines present in the book collection and maps them
to appropriate authority sources for validation and conflict resolution.
"""

import logging
import json
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

from .data_models import TOCEntry, SubtopicEntry
from .authority_based_conflict_resolver import AuthoritySource, SourceType

logger = logging.getLogger(__name__)


@dataclass
class DisciplineMetadata:
    """
    Goal: Store comprehensive metadata about discovered disciplines.
    
    Contains information about discipline identification, authority sources,
    and validation parameters for curriculum generation.
    """
    name: str
    confidence_score: float
    primary_keywords: List[str]
    secondary_keywords: List[str]
    authority_sources: List[AuthoritySource]
    book_sources: List[str]
    topic_count: int
    complexity_level: str  # Introductory, Intermediate, Advanced
    interdisciplinary_connections: List[str]
    standards_frameworks: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'confidence_score': self.confidence_score,
            'primary_keywords': self.primary_keywords,
            'secondary_keywords': self.secondary_keywords,
            'authority_sources': [source.name for source in self.authority_sources],
            'book_sources': self.book_sources,
            'topic_count': self.topic_count,
            'complexity_level': self.complexity_level,
            'interdisciplinary_connections': self.interdisciplinary_connections,
            'standards_frameworks': self.standards_frameworks
        }


@dataclass 
class DisciplineMapping:
    """
    Goal: Map books and topics to their primary and secondary disciplines.
    
    Enables cross-disciplinary analysis and proper authority source selection.
    """
    book_title: str
    primary_discipline: str
    secondary_disciplines: List[str]
    confidence_scores: Dict[str, float]
    topic_distribution: Dict[str, int]  # Topics per discipline
    authority_recommendations: Dict[str, List[str]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'book_title': self.book_title,
            'primary_discipline': self.primary_discipline,
            'secondary_disciplines': self.secondary_disciplines,
            'confidence_scores': self.confidence_scores,
            'topic_distribution': self.topic_distribution,
            'authority_recommendations': self.authority_recommendations
        }


class DisciplineDiscoverer:
    """
    Goal: Discover disciplines and map to authority sources for validation.
    
    Identifies disciplines present in book collection, maps them to appropriate
    authority sources, and provides confidence scoring for validation.
    """
    
    def __init__(self):
        self.discipline_keywords = self._load_discipline_keywords()
        self.authority_mappings = self._load_authority_mappings()
        self.interdisciplinary_patterns = self._load_interdisciplinary_patterns()
        self.standards_mappings = self._load_standards_mappings()
        self.discovered_disciplines: Dict[str, DisciplineMetadata] = {}
        self.book_mappings: List[DisciplineMapping] = []
        
    def discover_disciplines_with_authority_mapping(self, book_list_data: Dict, 
                                                   toc_entries: List[TOCEntry]) -> Dict[str, DisciplineMetadata]:
        """
        Goal: Discover all disciplines and create comprehensive authority mappings.
        
        Analyzes book collection and TOC entries to identify disciplines with
        confidence scoring and authority source recommendations.
        """
        logger.info(f"Discovering disciplines from {len(book_list_data)} books and {len(toc_entries)} TOC entries")
        
        # Step 1: Analyze book titles and descriptions for discipline indicators
        book_disciplines = self._analyze_books_for_disciplines(book_list_data)
        
        # Step 2: Analyze TOC entries for discipline-specific content
        toc_disciplines = self._analyze_toc_for_disciplines(toc_entries)
        
        # Step 3: Combine and validate discipline discoveries
        combined_disciplines = self._combine_discipline_discoveries(book_disciplines, toc_disciplines)
        
        # Step 4: Map to authority sources
        for discipline_name, metadata in combined_disciplines.items():
            authority_sources = self._map_discipline_to_authorities(discipline_name, metadata)
            metadata.authority_sources = authority_sources
        
        # Step 5: Identify interdisciplinary connections
        self._identify_interdisciplinary_connections(combined_disciplines)
        
        # Step 6: Map to standards frameworks
        self._map_to_standards_frameworks(combined_disciplines)
        
        # Step 7: Create book-to-discipline mappings
        self._create_book_discipline_mappings(book_list_data, toc_entries, combined_disciplines)
        
        self.discovered_disciplines = combined_disciplines
        
        logger.info(f"Discovered {len(combined_disciplines)} disciplines with authority mappings")
        return combined_disciplines
    
    def _analyze_books_for_disciplines(self, book_list_data: Dict) -> Dict[str, DisciplineMetadata]:
        """Analyze book titles and metadata for discipline identification."""
        discipline_scores = defaultdict(float)
        discipline_keywords_found = defaultdict(set)
        discipline_books = defaultdict(list)
        
        for book_key, book_info in book_list_data.items():
            if not isinstance(book_info, dict):
                continue
                
            # Analyze book title and description
            text_to_analyze = f"{book_info.get('title', '')} {book_info.get('description', '')}".lower()
            
            for discipline, keywords in self.discipline_keywords.items():
                score = 0.0
                found_keywords = set()
                
                # Check primary keywords (higher weight)
                for keyword in keywords.get('primary', []):
                    if keyword.lower() in text_to_analyze:
                        score += 2.0
                        found_keywords.add(keyword)
                
                # Check secondary keywords (lower weight)
                for keyword in keywords.get('secondary', []):
                    if keyword.lower() in text_to_analyze:
                        score += 1.0
                        found_keywords.add(keyword)
                
                if score > 0:
                    discipline_scores[discipline] += score
                    discipline_keywords_found[discipline].update(found_keywords)
                    discipline_books[discipline].append(book_info.get('title', book_key))
        
        # Create discipline metadata
        discovered = {}
        for discipline, total_score in discipline_scores.items():
            if total_score >= 2.0:  # Minimum threshold
                confidence = min(total_score / 10.0, 1.0)  # Normalize to 0-1
                
                keywords_found = discipline_keywords_found[discipline]
                primary_keywords = [k for k in keywords_found if k in self.discipline_keywords[discipline]['primary']]
                secondary_keywords = [k for k in keywords_found if k in self.discipline_keywords[discipline]['secondary']]
                
                metadata = DisciplineMetadata(
                    name=discipline,
                    confidence_score=confidence,
                    primary_keywords=primary_keywords,
                    secondary_keywords=secondary_keywords,
                    authority_sources=[],  # Will be populated later
                    book_sources=discipline_books[discipline][:10],  # Limit to top 10
                    topic_count=0,  # Will be updated with TOC analysis
                    complexity_level=self._assess_complexity_level(discipline_books[discipline]),
                    interdisciplinary_connections=[],  # Will be populated later
                    standards_frameworks=[]  # Will be populated later
                )
                
                discovered[discipline] = metadata
        
        return discovered
    
    def _analyze_toc_for_disciplines(self, toc_entries: List[TOCEntry]) -> Dict[str, DisciplineMetadata]:
        """Analyze TOC entries for discipline-specific content patterns."""
        discipline_topic_counts = defaultdict(int)
        discipline_keywords_found = defaultdict(set)
        discipline_sources = defaultdict(set)
        
        for toc_entry in toc_entries:
            text_to_analyze = f"{toc_entry.title} {getattr(toc_entry, 'description', '') or ''}".lower()
            
            for discipline, keywords in self.discipline_keywords.items():
                matches = 0
                found_keywords = set()
                
                # Check for keyword matches
                for keyword_list in [keywords.get('primary', []), keywords.get('secondary', [])]:
                    for keyword in keyword_list:
                        if keyword.lower() in text_to_analyze:
                            matches += 1
                            found_keywords.add(keyword)
                
                if matches > 0:
                    discipline_topic_counts[discipline] += 1
                    discipline_keywords_found[discipline].update(found_keywords)
                    discipline_sources[discipline].add(toc_entry.book_source)
        
        # Create discipline metadata from TOC analysis
        discovered = {}
        for discipline, topic_count in discipline_topic_counts.items():
            if topic_count >= 5:  # Minimum topic threshold
                keywords_found = discipline_keywords_found[discipline]
                primary_keywords = [k for k in keywords_found if k in self.discipline_keywords[discipline]['primary']]
                secondary_keywords = [k for k in keywords_found if k in self.discipline_keywords[discipline]['secondary']]
                
                confidence = min(topic_count / 50.0, 1.0)  # Normalize based on topic count
                
                metadata = DisciplineMetadata(
                    name=discipline,
                    confidence_score=confidence,
                    primary_keywords=primary_keywords,
                    secondary_keywords=secondary_keywords,
                    authority_sources=[],
                    book_sources=list(discipline_sources[discipline])[:10],
                    topic_count=topic_count,
                    complexity_level=self._assess_toc_complexity_level(toc_entries, discipline),
                    interdisciplinary_connections=[],
                    standards_frameworks=[]
                )
                
                discovered[discipline] = metadata
        
        return discovered
    
    def _combine_discipline_discoveries(self, book_disciplines: Dict[str, DisciplineMetadata],
                                     toc_disciplines: Dict[str, DisciplineMetadata]) -> Dict[str, DisciplineMetadata]:
        """Combine discipline discoveries from books and TOC entries."""
        combined = {}
        
        # Get all unique disciplines
        all_disciplines = set(book_disciplines.keys()) | set(toc_disciplines.keys())
        
        for discipline in all_disciplines:
            book_meta = book_disciplines.get(discipline)
            toc_meta = toc_disciplines.get(discipline)
            
            if book_meta and toc_meta:
                # Combine both sources
                combined_confidence = (book_meta.confidence_score + toc_meta.confidence_score) / 2
                combined_keywords = list(set(book_meta.primary_keywords + toc_meta.primary_keywords))
                combined_secondary = list(set(book_meta.secondary_keywords + toc_meta.secondary_keywords))
                combined_sources = list(set(book_meta.book_sources + toc_meta.book_sources))
                
                metadata = DisciplineMetadata(
                    name=discipline,
                    confidence_score=combined_confidence,
                    primary_keywords=combined_keywords,
                    secondary_keywords=combined_secondary,
                    authority_sources=[],
                    book_sources=combined_sources,
                    topic_count=toc_meta.topic_count,
                    complexity_level=self._combine_complexity_levels(book_meta.complexity_level, toc_meta.complexity_level),
                    interdisciplinary_connections=[],
                    standards_frameworks=[]
                )
                
                combined[discipline] = metadata
                
            elif book_meta:
                # Only from books
                combined[discipline] = book_meta
            elif toc_meta:
                # Only from TOC
                combined[discipline] = toc_meta
        
        # Filter out low-confidence disciplines
        filtered = {name: meta for name, meta in combined.items() if meta.confidence_score >= 0.3}
        
        return filtered
    
    def _map_discipline_to_authorities(self, discipline: str, metadata: DisciplineMetadata) -> List[AuthoritySource]:
        """Map discipline to appropriate authority sources."""
        authority_sources = []
        
        # Get predefined authority mappings
        mappings = self.authority_mappings.get(discipline, {})
        
        for authority_name, details in mappings.items():
            authority_source = AuthoritySource(
                name=authority_name,
                source_type=SourceType(details['type']),
                discipline=discipline,
                authority_score=details['score'],
                reliability_factors=details.get('reliability_factors', {}),
                url=details.get('url', '')
            )
            authority_sources.append(authority_source)
        
        # Add universal authorities if not already present
        universal_authorities = self.authority_mappings.get('universal', {})
        for authority_name, details in universal_authorities.items():
            if not any(source.name == authority_name for source in authority_sources):
                authority_source = AuthoritySource(
                    name=authority_name,
                    source_type=SourceType(details['type']),
                    discipline='universal',
                    authority_score=details['score'] * 0.8,  # Lower score for universal
                    reliability_factors=details.get('reliability_factors', {}),
                    url=details.get('url', '')
                )
                authority_sources.append(authority_source)
        
        # Sort by authority score
        authority_sources.sort(key=lambda x: x.authority_score, reverse=True)
        
        return authority_sources
    
    def _identify_interdisciplinary_connections(self, disciplines: Dict[str, DisciplineMetadata]) -> None:
        """Identify interdisciplinary connections between discovered disciplines."""
        
        for discipline_name, metadata in disciplines.items():
            connections = []
            
            # Check predefined interdisciplinary patterns
            patterns = self.interdisciplinary_patterns.get(discipline_name, {})
            
            for connected_discipline in patterns.get('strong_connections', []):
                if connected_discipline in disciplines:
                    connections.append(connected_discipline)
            
            # Check for keyword overlap indicating connections
            for other_discipline, other_metadata in disciplines.items():
                if other_discipline != discipline_name:
                    overlap_score = self._calculate_keyword_overlap(metadata, other_metadata)
                    if overlap_score >= 0.3:  # Threshold for interdisciplinary connection
                        if other_discipline not in connections:
                            connections.append(other_discipline)
            
            metadata.interdisciplinary_connections = connections
    
    def _map_to_standards_frameworks(self, disciplines: Dict[str, DisciplineMetadata]) -> None:
        """Map disciplines to relevant standards frameworks."""
        
        for discipline_name, metadata in disciplines.items():
            frameworks = []
            
            # Get predefined standards mappings
            standards = self.standards_mappings.get(discipline_name, [])
            frameworks.extend(standards)
            
            # Add universal standards
            universal_standards = self.standards_mappings.get('universal', [])
            frameworks.extend(universal_standards)
            
            # Remove duplicates while preserving order
            unique_frameworks = []
            for framework in frameworks:
                if framework not in unique_frameworks:
                    unique_frameworks.append(framework)
            
            metadata.standards_frameworks = unique_frameworks
    
    def _create_book_discipline_mappings(self, book_list_data: Dict, toc_entries: List[TOCEntry],
                                       disciplines: Dict[str, DisciplineMetadata]) -> None:
        """Create mappings from books to their disciplines."""
        
        self.book_mappings = []
        
        # Group TOC entries by book
        toc_by_book = defaultdict(list)
        for toc_entry in toc_entries:
            toc_by_book[toc_entry.book_source].append(toc_entry)
        
        for book_key, book_info in book_list_data.items():
            if not isinstance(book_info, dict):
                continue
                
            book_title = book_info.get('title', book_key)
            book_toc = toc_by_book.get(book_title, [])
            
            # Analyze book for discipline assignment
            discipline_scores = self._analyze_book_for_discipline_scores(book_info, book_toc, disciplines)
            
            if discipline_scores:
                # Primary discipline (highest score)
                primary_discipline = max(discipline_scores.items(), key=lambda x: x[1])[0]
                
                # Secondary disciplines (significant scores but not primary)
                secondary_disciplines = [disc for disc, score in discipline_scores.items() 
                                       if disc != primary_discipline and score >= 0.2]
                
                # Topic distribution
                topic_distribution = self._calculate_topic_distribution(book_toc, disciplines)
                
                # Authority recommendations
                authority_recommendations = {}
                for discipline in [primary_discipline] + secondary_disciplines:
                    if discipline in disciplines:
                        authorities = [source.name for source in disciplines[discipline].authority_sources[:3]]
                        authority_recommendations[discipline] = authorities
                
                mapping = DisciplineMapping(
                    book_title=book_title,
                    primary_discipline=primary_discipline,
                    secondary_disciplines=secondary_disciplines,
                    confidence_scores=discipline_scores,
                    topic_distribution=topic_distribution,
                    authority_recommendations=authority_recommendations
                )
                
                self.book_mappings.append(mapping)
    
    def _analyze_book_for_discipline_scores(self, book_info: Dict, book_toc: List[TOCEntry],
                                          disciplines: Dict[str, DisciplineMetadata]) -> Dict[str, float]:
        """Analyze a book to determine discipline scores."""
        scores = {}
        
        book_text = f"{book_info.get('title', '')} {book_info.get('description', '')}".lower()
        
        for discipline_name, metadata in disciplines.items():
            score = 0.0
            
            # Check book title/description
            for keyword in metadata.primary_keywords:
                if keyword.lower() in book_text:
                    score += 0.3
            
            for keyword in metadata.secondary_keywords:
                if keyword.lower() in book_text:
                    score += 0.1
            
            # Check TOC entries
            toc_matches = 0
            for toc_entry in book_toc:
                toc_text = f"{toc_entry.title} {getattr(toc_entry, 'description', '') or ''}".lower()
                for keyword in metadata.primary_keywords + metadata.secondary_keywords:
                    if keyword.lower() in toc_text:
                        toc_matches += 1
                        break
            
            if book_toc:
                toc_score = min(toc_matches / len(book_toc), 0.7)  # Cap at 0.7
                score += toc_score
            
            if score > 0:
                scores[discipline_name] = min(score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _calculate_topic_distribution(self, book_toc: List[TOCEntry], 
                                    disciplines: Dict[str, DisciplineMetadata]) -> Dict[str, int]:
        """Calculate how topics are distributed across disciplines."""
        distribution = defaultdict(int)
        
        for toc_entry in book_toc:
            toc_text = f"{toc_entry.title} {getattr(toc_entry, 'description', '') or ''}".lower()
            
            for discipline_name, metadata in disciplines.items():
                matches = 0
                for keyword in metadata.primary_keywords + metadata.secondary_keywords:
                    if keyword.lower() in toc_text:
                        matches += 1
                
                if matches > 0:
                    distribution[discipline_name] += 1
        
        return dict(distribution)
    
    def _calculate_keyword_overlap(self, meta1: DisciplineMetadata, meta2: DisciplineMetadata) -> float:
        """Calculate keyword overlap between two discipline metadata objects."""
        keywords1 = set(meta1.primary_keywords + meta1.secondary_keywords)
        keywords2 = set(meta2.primary_keywords + meta2.secondary_keywords)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_complexity_level(self, book_sources: List[str]) -> str:
        """Assess complexity level based on book sources."""
        complexity_indicators = {
            'introductory': ['intro', 'basic', 'fundamental', 'elementary', 'first', 'beginning'],
            'intermediate': ['intermediate', 'general', 'principles', 'concepts'],
            'advanced': ['advanced', 'graduate', 'research', 'specialized', 'professional']
        }
        
        level_scores = defaultdict(int)
        
        for book_title in book_sources:
            title_lower = book_title.lower()
            for level, indicators in complexity_indicators.items():
                for indicator in indicators:
                    if indicator in title_lower:
                        level_scores[level] += 1
        
        if level_scores:
            return max(level_scores.items(), key=lambda x: x[1])[0]
        return 'intermediate'  # Default
    
    def _assess_toc_complexity_level(self, toc_entries: List[TOCEntry], discipline: str) -> str:
        """Assess complexity level based on TOC entry patterns."""
        advanced_patterns = ['research', 'graduate', 'advanced', 'specialized', 'current', 'cutting-edge']
        intro_patterns = ['introduction', 'basic', 'fundamental', 'overview', 'elementary']
        
        advanced_count = 0
        intro_count = 0
        total_count = 0
        
        for toc_entry in toc_entries:
            # Only consider entries relevant to this discipline
            text = f"{toc_entry.title} {getattr(toc_entry, 'description', '') or ''}".lower()
            keywords = self.discipline_keywords.get(discipline, {})
            relevant = False
            
            for keyword_list in [keywords.get('primary', []), keywords.get('secondary', [])]:
                if any(keyword.lower() in text for keyword in keyword_list):
                    relevant = True
                    break
            
            if relevant:
                total_count += 1
                
                if any(pattern in text for pattern in advanced_patterns):
                    advanced_count += 1
                elif any(pattern in text for pattern in intro_patterns):
                    intro_count += 1
        
        if total_count == 0:
            return 'intermediate'
        
        advanced_ratio = advanced_count / total_count
        intro_ratio = intro_count / total_count
        
        if advanced_ratio > 0.3:
            return 'advanced'
        elif intro_ratio > 0.4:
            return 'introductory'
        else:
            return 'intermediate'
    
    def _combine_complexity_levels(self, book_level: str, toc_level: str) -> str:
        """Combine complexity assessments from books and TOC."""
        level_hierarchy = {'introductory': 1, 'intermediate': 2, 'advanced': 3}
        
        book_rank = level_hierarchy.get(book_level, 2)
        toc_rank = level_hierarchy.get(toc_level, 2)
        
        # Take the higher complexity level
        combined_rank = max(book_rank, toc_rank)
        
        rank_to_level = {1: 'introductory', 2: 'intermediate', 3: 'advanced'}
        return rank_to_level[combined_rank]
    
    def get_discipline_discovery_report(self) -> Dict:
        """Generate comprehensive discipline discovery report."""
        return {
            'discovered_disciplines': {name: meta.to_dict() for name, meta in self.discovered_disciplines.items()},
            'total_disciplines': len(self.discovered_disciplines),
            'book_mappings': [mapping.to_dict() for mapping in self.book_mappings],
            'interdisciplinary_analysis': self._generate_interdisciplinary_analysis(),
            'authority_coverage': self._generate_authority_coverage_analysis(),
            'standards_coverage': self._generate_standards_coverage_analysis()
        }
    
    def _generate_interdisciplinary_analysis(self) -> Dict:
        """Generate analysis of interdisciplinary connections."""
        connection_counts = defaultdict(int)
        
        for metadata in self.discovered_disciplines.values():
            for connection in metadata.interdisciplinary_connections:
                connection_counts[connection] += 1
        
        most_connected = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'most_connected_disciplines': most_connected,
            'total_connections': sum(len(meta.interdisciplinary_connections) 
                                   for meta in self.discovered_disciplines.values()),
            'average_connections': sum(len(meta.interdisciplinary_connections) 
                                     for meta in self.discovered_disciplines.values()) / 
                                   len(self.discovered_disciplines) if self.discovered_disciplines else 0
        }
    
    def _generate_authority_coverage_analysis(self) -> Dict:
        """Generate analysis of authority source coverage."""
        authority_counts = defaultdict(int)
        discipline_authority_counts = defaultdict(int)
        
        for discipline, metadata in self.discovered_disciplines.items():
            discipline_authority_counts[discipline] = len(metadata.authority_sources)
            for source in metadata.authority_sources:
                authority_counts[source.name] += 1
        
        return {
            'total_unique_authorities': len(authority_counts),
            'most_used_authorities': sorted(authority_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'disciplines_with_authorities': len([d for d, count in discipline_authority_counts.items() if count > 0]),
            'average_authorities_per_discipline': sum(discipline_authority_counts.values()) / 
                                                len(discipline_authority_counts) if discipline_authority_counts else 0
        }
    
    def _generate_standards_coverage_analysis(self) -> Dict:
        """Generate analysis of standards framework coverage."""
        standards_counts = defaultdict(int)
        
        for metadata in self.discovered_disciplines.values():
            for framework in metadata.standards_frameworks:
                standards_counts[framework] += 1
        
        return {
            'total_standards_frameworks': len(standards_counts),
            'most_common_frameworks': sorted(standards_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'disciplines_with_standards': len([meta for meta in self.discovered_disciplines.values() 
                                             if meta.standards_frameworks])
        }
    
    def _load_discipline_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load comprehensive discipline keyword mappings."""
        return {
            'Physics': {
                'primary': [
                    'physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'quantum',
                    'relativity', 'optics', 'waves', 'force', 'energy', 'momentum'
                ],
                'secondary': [
                    'kinematics', 'dynamics', 'statics', 'fluid', 'acoustics', 'particle',
                    'field', 'electromagnetic', 'nuclear', 'atomic', 'solid state'
                ]
            },
            'Chemistry': {
                'primary': [
                    'chemistry', 'chemical', 'molecule', 'atom', 'reaction', 'bond',
                    'organic', 'inorganic', 'analytical', 'physical chemistry'
                ],
                'secondary': [
                    'stoichiometry', 'equilibrium', 'kinetics', 'thermochemistry',
                    'electrochemistry', 'spectroscopy', 'polymer', 'catalyst'
                ]
            },
            'Biology': {
                'primary': [
                    'biology', 'biological', 'life', 'cell', 'organism', 'genetics',
                    'evolution', 'ecology', 'physiology', 'biochemistry'
                ],
                'secondary': [
                    'molecular biology', 'microbiology', 'botany', 'zoology',
                    'anatomy', 'neuroscience', 'immunology', 'developmental'
                ]
            },
            'Medicine': {
                'primary': [
                    'medicine', 'medical', 'clinical', 'health', 'disease', 'therapy',
                    'diagnosis', 'treatment', 'patient', 'healthcare'
                ],
                'secondary': [
                    'pathology', 'pharmacology', 'surgery', 'radiology',
                    'cardiology', 'neurology', 'oncology', 'pediatrics'
                ]
            },
            'Psychology': {
                'primary': [
                    'psychology', 'psychological', 'behavior', 'cognition', 'mental',
                    'brain', 'mind', 'learning', 'memory', 'perception'
                ],
                'secondary': [
                    'cognitive', 'social psychology', 'developmental', 'abnormal',
                    'personality', 'emotion', 'motivation', 'consciousness'
                ]
            },
            'Mathematics': {
                'primary': [
                    'mathematics', 'mathematical', 'calculus', 'algebra', 'geometry',
                    'statistics', 'probability', 'analysis', 'topology'
                ],
                'secondary': [
                    'differential', 'integral', 'linear algebra', 'discrete',
                    'number theory', 'graph theory', 'optimization'
                ]
            },
            'Computer Science': {
                'primary': [
                    'computer science', 'programming', 'algorithm', 'software',
                    'data structure', 'computation', 'artificial intelligence'
                ],
                'secondary': [
                    'machine learning', 'database', 'network', 'security',
                    'graphics', 'human-computer interaction', 'distributed'
                ]
            },
            'Engineering': {
                'primary': [
                    'engineering', 'design', 'system', 'control', 'signal',
                    'mechanical', 'electrical', 'civil', 'chemical engineering'
                ],
                'secondary': [
                    'robotics', 'automation', 'manufacturing', 'materials',
                    'structural', 'fluid mechanics', 'heat transfer'
                ]
            }
        }
    
    def _load_authority_mappings(self) -> Dict[str, Dict[str, Dict]]:
        """Load authority source mappings for each discipline."""
        return {
            'Physics': {
                'NIST': {
                    'type': 'Government_Agency',
                    'score': 1.0,
                    'url': 'https://www.nist.gov/',
                    'reliability_factors': {'government_backing': 1.0, 'standards_authority': 1.0}
                },
                'AIP': {
                    'type': 'Professional_Organization',
                    'score': 0.95,
                    'url': 'https://www.aip.org/',
                    'reliability_factors': {'professional_standards': 1.0, 'peer_review': 0.9}
                },
                'Physical Review': {
                    'type': 'Peer_Reviewed_Journal',
                    'score': 0.92,
                    'url': 'https://journals.aps.org/',
                    'reliability_factors': {'peer_review': 1.0, 'impact_factor': 0.9}
                }
            },
            'Chemistry': {
                'IUPAC': {
                    'type': 'International_Standard',
                    'score': 1.0,
                    'url': 'https://iupac.org/',
                    'reliability_factors': {'international_standard': 1.0, 'nomenclature_authority': 1.0}
                },
                'ACS': {
                    'type': 'Professional_Organization',
                    'score': 0.98,
                    'url': 'https://www.acs.org/',
                    'reliability_factors': {'professional_standards': 1.0, 'research_quality': 0.95}
                }
            },
            'Biology': {
                'WHO': {
                    'type': 'Government_Agency',
                    'score': 1.0,
                    'url': 'https://www.who.int/',
                    'reliability_factors': {'global_authority': 1.0, 'health_expertise': 1.0}
                },
                'NIH': {
                    'type': 'Government_Agency',
                    'score': 0.98,
                    'url': 'https://www.nih.gov/',
                    'reliability_factors': {'research_funding': 1.0, 'medical_authority': 0.95}
                }
            },
            'Medicine': {
                'WHO': {
                    'type': 'Government_Agency',
                    'score': 1.0,
                    'url': 'https://www.who.int/',
                    'reliability_factors': {'global_health_authority': 1.0}
                },
                'FDA': {
                    'type': 'Government_Agency',
                    'score': 0.98,
                    'url': 'https://www.fda.gov/',
                    'reliability_factors': {'regulatory_authority': 1.0, 'drug_approval': 1.0}
                }
            },
            'Psychology': {
                'APA': {
                    'type': 'Professional_Organization',
                    'score': 1.0,
                    'url': 'https://www.apa.org/',
                    'reliability_factors': {'professional_standards': 1.0, 'ethical_guidelines': 1.0}
                }
            },
            'universal': {
                'ISO': {
                    'type': 'International_Standard',
                    'score': 0.95,
                    'url': 'https://www.iso.org/',
                    'reliability_factors': {'international_standards': 1.0}
                },
                'UNESCO': {
                    'type': 'Government_Agency',
                    'score': 0.90,
                    'url': 'https://www.unesco.org/',
                    'reliability_factors': {'educational_authority': 1.0}
                }
            }
        }
    
    def _load_interdisciplinary_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load interdisciplinary connection patterns."""
        return {
            'Physics': {
                'strong_connections': ['Mathematics', 'Chemistry', 'Engineering'],
                'weak_connections': ['Biology', 'Computer Science']
            },
            'Chemistry': {
                'strong_connections': ['Physics', 'Biology', 'Medicine'],
                'weak_connections': ['Mathematics', 'Engineering']
            },
            'Biology': {
                'strong_connections': ['Chemistry', 'Medicine', 'Psychology'],
                'weak_connections': ['Physics', 'Mathematics']
            },
            'Medicine': {
                'strong_connections': ['Biology', 'Chemistry', 'Psychology'],
                'weak_connections': ['Physics']
            },
            'Psychology': {
                'strong_connections': ['Biology', 'Medicine'],
                'weak_connections': ['Mathematics', 'Computer Science']
            },
            'Mathematics': {
                'strong_connections': ['Physics', 'Computer Science', 'Engineering'],
                'weak_connections': ['Chemistry', 'Biology']
            },
            'Computer Science': {
                'strong_connections': ['Mathematics', 'Engineering'],
                'weak_connections': ['Psychology', 'Physics']
            },
            'Engineering': {
                'strong_connections': ['Physics', 'Mathematics', 'Computer Science'],
                'weak_connections': ['Chemistry']
            }
        }
    
    def _load_standards_mappings(self) -> Dict[str, List[str]]:
        """Load standards framework mappings for disciplines."""
        return {
            'Physics': ['NGSS', 'AP_Physics', 'IB_Physics', 'AAPT_Physics', 'MCAT'],
            'Chemistry': ['NGSS', 'AP_Chemistry', 'IB_Chemistry', 'ACS_Chemistry', 'MCAT'],
            'Biology': ['NGSS', 'AP_Biology', 'IB_Biology', 'MCAT'],
            'Medicine': ['MCAT', 'WHO_Health', 'AAMC_Medical'],
            'Psychology': ['AP_Psychology', 'APA_Psychology', 'MCAT'],
            'Mathematics': ['AP_Mathematics', 'IB_Mathematics', 'Common_Core'],
            'Computer Science': ['AP_Computer_Science', 'ACM_Computing'],
            'Engineering': ['ABET_Engineering', 'IEEE_Standards'],
            'universal': ['Educational_Standards', 'International_Baccalaureate']
        }