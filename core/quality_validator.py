#!/usr/bin/env python3
"""
Quality Validation System
Compares LLM-enhanced TOC normalization quality vs previous methods.

This system provides:
1. Pedagogical structure validation
2. Topic coverage analysis
3. Coherence and organization metrics
4. Academic level progression validation
5. Content diversity assessment
6. Cross-method quality comparison
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Represents quality metrics for a curriculum."""
    total_topics: int
    unique_topics: int
    duplicate_ratio: float
    academic_level_distribution: Dict[str, int]
    level_progression_score: float
    topic_coherence_score: float
    coverage_breadth_score: float
    pedagogical_ordering_score: float
    content_diversity_score: float
    structural_consistency_score: float
    overall_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComparisonResult:
    """Results from comparing two normalization methods."""
    llm_enhanced_metrics: QualityMetrics
    traditional_metrics: QualityMetrics
    improvement_percentages: Dict[str, float]
    strengths_llm: List[str]
    strengths_traditional: List[str]
    overall_winner: str
    confidence_score: float
    detailed_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class QualityValidator:
    """Validates and compares curriculum quality between different normalization methods."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("Cache/QualityValidation")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Academic level hierarchy for progression scoring
        self.level_hierarchy = {
            'high_school': 1,
            'undergraduate': 2,
            'graduate': 3,
            'professional': 4
        }
        
        # Physics domain knowledge for coherence scoring
        self.physics_topics_hierarchy = {
            'mechanics': ['motion', 'force', 'energy', 'momentum', 'rotation'],
            'thermodynamics': ['heat', 'temperature', 'entropy', 'gas laws'],
            'electromagnetism': ['electric field', 'magnetic field', 'circuits', 'waves'],
            'quantum physics': ['quantum mechanics', 'atomic physics', 'particle physics'],
            'optics': ['light', 'reflection', 'refraction', 'interference'],
            'modern physics': ['relativity', 'nuclear physics', 'cosmology']
        }
        
        logger.info("QualityValidator initialized")
    
    def validate_curriculum_quality(self, curriculum_data: Dict[str, Any], 
                                  method_name: str = "unknown") -> QualityMetrics:
        """
        Validate quality of a normalized curriculum.
        
        Args:
            curriculum_data: Normalized curriculum data
            method_name: Name of the normalization method
            
        Returns:
            QualityMetrics with comprehensive quality assessment
        """
        logger.info(f"Validating curriculum quality for method: {method_name}")
        
        topics = curriculum_data.get('normalized_topics', [])
        if not topics:
            logger.warning("No topics found in curriculum data")
            return self._create_empty_metrics()
        
        # Basic metrics
        total_topics = len(topics)
        unique_topics = len(set(topic.get('title', '').lower() for topic in topics))
        duplicate_ratio = 1.0 - (unique_topics / total_topics) if total_topics > 0 else 0.0
        
        # Academic level distribution
        level_distribution = Counter(topic.get('academic_level', 'unknown') for topic in topics)
        
        # Calculate individual quality scores
        level_progression_score = self._calculate_level_progression_score(topics)
        topic_coherence_score = self._calculate_topic_coherence_score(topics)
        coverage_breadth_score = self._calculate_coverage_breadth_score(topics)
        pedagogical_ordering_score = self._calculate_pedagogical_ordering_score(topics)
        content_diversity_score = self._calculate_content_diversity_score(topics)
        structural_consistency_score = self._calculate_structural_consistency_score(topics)
        
        # Overall quality score (weighted average)
        weights = {
            'level_progression': 0.20,
            'topic_coherence': 0.25,
            'coverage_breadth': 0.15,
            'pedagogical_ordering': 0.20,
            'content_diversity': 0.10,
            'structural_consistency': 0.10
        }
        
        overall_quality_score = (
            level_progression_score * weights['level_progression'] +
            topic_coherence_score * weights['topic_coherence'] +
            coverage_breadth_score * weights['coverage_breadth'] +
            pedagogical_ordering_score * weights['pedagogical_ordering'] +
            content_diversity_score * weights['content_diversity'] +
            structural_consistency_score * weights['structural_consistency']
        )
        
        metrics = QualityMetrics(
            total_topics=total_topics,
            unique_topics=unique_topics,
            duplicate_ratio=duplicate_ratio,
            academic_level_distribution=dict(level_distribution),
            level_progression_score=level_progression_score,
            topic_coherence_score=topic_coherence_score,
            coverage_breadth_score=coverage_breadth_score,
            pedagogical_ordering_score=pedagogical_ordering_score,
            content_diversity_score=content_diversity_score,
            structural_consistency_score=structural_consistency_score,
            overall_quality_score=overall_quality_score
        )
        
        logger.info(f"Quality validation completed. Overall score: {overall_quality_score:.3f}")
        return metrics
    
    def compare_methods(self, llm_enhanced_data: Dict[str, Any], 
                       traditional_data: Dict[str, Any]) -> ComparisonResult:
        """
        Compare quality between LLM-enhanced and traditional normalization methods.
        
        Args:
            llm_enhanced_data: Results from LLM-enhanced normalization
            traditional_data: Results from traditional normalization
            
        Returns:
            ComparisonResult with detailed comparison analysis
        """
        logger.info("Starting quality comparison between methods")
        
        # Validate both methods
        llm_metrics = self.validate_curriculum_quality(llm_enhanced_data, "llm_enhanced")
        traditional_metrics = self.validate_curriculum_quality(traditional_data, "traditional")
        
        # Calculate improvement percentages
        improvement_percentages = self._calculate_improvements(llm_metrics, traditional_metrics)
        
        # Identify strengths of each method
        strengths_llm, strengths_traditional = self._identify_method_strengths(
            llm_metrics, traditional_metrics
        )
        
        # Determine overall winner
        overall_winner, confidence_score = self._determine_winner(llm_metrics, traditional_metrics)
        
        # Detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            llm_enhanced_data, traditional_data, llm_metrics, traditional_metrics
        )
        
        result = ComparisonResult(
            llm_enhanced_metrics=llm_metrics,
            traditional_metrics=traditional_metrics,
            improvement_percentages=improvement_percentages,
            strengths_llm=strengths_llm,
            strengths_traditional=strengths_traditional,
            overall_winner=overall_winner,
            confidence_score=confidence_score,
            detailed_analysis=detailed_analysis
        )
        
        logger.info(f"Quality comparison completed. Winner: {overall_winner} (confidence: {confidence_score:.3f})")
        return result
    
    def _calculate_level_progression_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate how well topics progress through academic levels."""
        if not topics:
            return 0.0
        
        # Count topics per level
        level_counts = Counter(topic.get('academic_level', 'unknown') for topic in topics)
        
        # Check if all major levels are represented
        major_levels = ['high_school', 'undergraduate', 'graduate']
        covered_levels = set(level_counts.keys()).intersection(major_levels)
        coverage_score = len(covered_levels) / len(major_levels)
        
        # Check for balanced distribution (not too skewed)
        total_topics = sum(level_counts.values())
        if total_topics == 0:
            return 0.0
        
        proportions = [count / total_topics for count in level_counts.values()]
        balance_score = 1.0 - np.std(proportions) if proportions else 0.0
        
        # Combine scores
        return (coverage_score * 0.7 + balance_score * 0.3)
    
    def _calculate_topic_coherence_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate how coherent and related the topics are within the domain."""
        if not topics:
            return 0.0
        
        topic_titles = [topic.get('title', '').lower() for topic in topics]
        
        # Physics domain coherence
        physics_score = self._calculate_physics_domain_coherence(topic_titles)
        
        # Semantic coherence using TF-IDF similarity
        semantic_score = self._calculate_semantic_coherence(topic_titles)
        
        # Hierarchical coherence (topics build on each other)
        hierarchical_score = self._calculate_hierarchical_coherence(topics)
        
        # Weighted average
        return (physics_score * 0.4 + semantic_score * 0.3 + hierarchical_score * 0.3)
    
    def _calculate_physics_domain_coherence(self, topic_titles: List[str]) -> float:
        """Calculate coherence within physics domain knowledge."""
        if not topic_titles:
            return 0.0
        
        # Count topics in each physics subdomain
        subdomain_coverage = defaultdict(int)
        total_matches = 0
        
        for title in topic_titles:
            for subdomain, keywords in self.physics_topics_hierarchy.items():
                if any(keyword in title for keyword in keywords):
                    subdomain_coverage[subdomain] += 1
                    total_matches += 1
        
        if total_matches == 0:
            return 0.5  # Neutral score if no clear physics topics found
        
        # Score based on coverage breadth and depth
        covered_subdomains = len(subdomain_coverage)
        max_subdomains = len(self.physics_topics_hierarchy)
        breadth_score = covered_subdomains / max_subdomains
        
        # Depth score (average topics per covered subdomain)
        avg_depth = total_matches / covered_subdomains if covered_subdomains > 0 else 0
        depth_score = min(avg_depth / 5.0, 1.0)  # Normalize to 0-1
        
        return (breadth_score * 0.6 + depth_score * 0.4)
    
    def _calculate_semantic_coherence(self, topic_titles: List[str]) -> float:
        """Calculate semantic coherence using TF-IDF similarity."""
        if len(topic_titles) < 2:
            return 1.0
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(topic_titles)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Get average similarity (excluding diagonal)
            n = similarity_matrix.shape[0]
            total_similarity = similarity_matrix.sum() - n  # Exclude diagonal
            avg_similarity = total_similarity / (n * (n - 1)) if n > 1 else 0
            
            return max(0.0, avg_similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating semantic coherence: {e}")
            return 0.5
    
    def _calculate_hierarchical_coherence(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate how well topics follow a hierarchical learning progression."""
        if not topics:
            return 0.0
        
        # Group topics by level and check for logical progression
        levels_order = ['high_school', 'undergraduate', 'graduate']
        topics_by_level = defaultdict(list)
        
        for topic in topics:
            level = topic.get('academic_level', 'unknown')
            if level in levels_order:
                topics_by_level[level].append(topic.get('title', ''))
        
        # Check for prerequisite-like progression
        progression_score = 0.0
        total_comparisons = 0
        
        for i in range(len(levels_order) - 1):
            current_level = levels_order[i]
            next_level = levels_order[i + 1]
            
            current_topics = topics_by_level[current_level]
            next_topics = topics_by_level[next_level]
            
            if current_topics and next_topics:
                # Simple heuristic: later levels should have more advanced terminology
                current_complexity = self._calculate_topic_complexity(current_topics)
                next_complexity = self._calculate_topic_complexity(next_topics)
                
                if next_complexity >= current_complexity:
                    progression_score += 1.0
                total_comparisons += 1
        
        return progression_score / total_comparisons if total_comparisons > 0 else 0.5
    
    def _calculate_topic_complexity(self, topic_titles: List[str]) -> float:
        """Calculate average complexity of topic titles."""
        if not topic_titles:
            return 0.0
        
        complexity_indicators = [
            'advanced', 'quantum', 'relativistic', 'nuclear', 'molecular',
            'electromagnetic', 'thermodynamic', 'statistical', 'theoretical'
        ]
        
        total_complexity = 0
        for title in topic_titles:
            title_lower = title.lower()
            complexity = sum(1 for indicator in complexity_indicators if indicator in title_lower)
            complexity += len(title.split()) * 0.1  # Longer titles tend to be more complex
            total_complexity += complexity
        
        return total_complexity / len(topic_titles)
    
    def _calculate_coverage_breadth_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate how broadly the curriculum covers the subject domain."""
        if not topics:
            return 0.0
        
        topic_titles = [topic.get('title', '').lower() for topic in topics]
        
        # Check coverage of major physics areas
        major_areas = {
            'mechanics': ['motion', 'force', 'energy', 'momentum', 'mechanics'],
            'thermodynamics': ['heat', 'temperature', 'thermal', 'thermodynamics'],
            'electromagnetism': ['electric', 'magnetic', 'electromagnetic', 'circuit'],
            'waves_optics': ['wave', 'light', 'optic', 'sound', 'interference'],
            'modern_physics': ['quantum', 'atomic', 'nuclear', 'relativity', 'particle']
        }
        
        covered_areas = 0
        for area, keywords in major_areas.items():
            if any(any(keyword in title for keyword in keywords) for title in topic_titles):
                covered_areas += 1
        
        return covered_areas / len(major_areas)
    
    def _calculate_pedagogical_ordering_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate how well topics are ordered for learning."""
        if not topics:
            return 0.0
        
        # Check if fundamental topics appear before advanced ones
        fundamental_keywords = ['introduction', 'basic', 'fundamental', 'overview', 'definition']
        advanced_keywords = ['advanced', 'complex', 'detailed', 'analysis', 'application']
        
        fundamental_positions = []
        advanced_positions = []
        
        for i, topic in enumerate(topics):
            title = topic.get('title', '').lower()
            if any(keyword in title for keyword in fundamental_keywords):
                fundamental_positions.append(i)
            elif any(keyword in title for keyword in advanced_keywords):
                advanced_positions.append(i)
        
        if not fundamental_positions or not advanced_positions:
            return 0.5  # Neutral if no clear fundamental/advanced topics
        
        # Check if fundamentals generally come before advanced
        avg_fundamental = np.mean(fundamental_positions)
        avg_advanced = np.mean(advanced_positions)
        
        if avg_advanced > avg_fundamental:
            return 1.0
        else:
            return max(0.0, 1.0 - (avg_fundamental - avg_advanced) / len(topics))
    
    def _calculate_content_diversity_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate diversity of content and perspectives."""
        if not topics:
            return 0.0
        
        # Check diversity in topic types
        topic_titles = [topic.get('title', '').lower() for topic in topics]
        
        # Different types of content
        content_types = {
            'theoretical': ['theory', 'principle', 'law', 'concept'],
            'practical': ['experiment', 'application', 'problem', 'example'],
            'mathematical': ['equation', 'calculation', 'formula', 'mathematics'],
            'historical': ['history', 'historical', 'development', 'discovery'],
            'conceptual': ['understanding', 'concept', 'idea', 'framework']
        }
        
        type_coverage = 0
        for content_type, keywords in content_types.items():
            if any(any(keyword in title for keyword in keywords) for title in topic_titles):
                type_coverage += 1
        
        diversity_score = type_coverage / len(content_types)
        
        # Check for source diversity if available
        source_books = set(topic.get('original_book', 'unknown') for topic in topics)
        source_diversity = min(len(source_books) / 5.0, 1.0)  # Normalize to max 5 sources
        
        return (diversity_score * 0.7 + source_diversity * 0.3)
    
    def _calculate_structural_consistency_score(self, topics: List[Dict[str, Any]]) -> float:
        """Calculate consistency in topic structure and formatting."""
        if not topics:
            return 0.0
        
        # Check consistency in required fields
        required_fields = ['title', 'level', 'academic_level', 'original_book']
        field_completeness = []
        
        for field in required_fields:
            complete_count = sum(1 for topic in topics if topic.get(field))
            completeness = complete_count / len(topics)
            field_completeness.append(completeness)
        
        avg_completeness = np.mean(field_completeness)
        
        # Check title format consistency
        title_lengths = [len(topic.get('title', '')) for topic in topics]
        title_consistency = 1.0 - (np.std(title_lengths) / np.mean(title_lengths)) if title_lengths else 0.0
        title_consistency = max(0.0, min(1.0, title_consistency))
        
        return (avg_completeness * 0.8 + title_consistency * 0.2)
    
    def _calculate_improvements(self, llm_metrics: QualityMetrics, 
                              traditional_metrics: QualityMetrics) -> Dict[str, float]:
        """Calculate percentage improvements of LLM method over traditional."""
        improvements = {}
        
        metrics_to_compare = [
            'level_progression_score', 'topic_coherence_score', 'coverage_breadth_score',
            'pedagogical_ordering_score', 'content_diversity_score', 
            'structural_consistency_score', 'overall_quality_score'
        ]
        
        for metric in metrics_to_compare:
            llm_value = getattr(llm_metrics, metric)
            traditional_value = getattr(traditional_metrics, metric)
            
            if traditional_value > 0:
                improvement = ((llm_value - traditional_value) / traditional_value) * 100
            else:
                improvement = 100.0 if llm_value > 0 else 0.0
            
            improvements[metric] = improvement
        
        return improvements
    
    def _identify_method_strengths(self, llm_metrics: QualityMetrics, 
                                 traditional_metrics: QualityMetrics) -> Tuple[List[str], List[str]]:
        """Identify strengths of each method."""
        strengths_llm = []
        strengths_traditional = []
        
        comparisons = [
            ('level_progression_score', 'Academic Level Progression'),
            ('topic_coherence_score', 'Topic Coherence'),
            ('coverage_breadth_score', 'Subject Coverage'),
            ('pedagogical_ordering_score', 'Pedagogical Ordering'),
            ('content_diversity_score', 'Content Diversity'),
            ('structural_consistency_score', 'Structural Consistency')
        ]
        
        for metric, description in comparisons:
            llm_value = getattr(llm_metrics, metric)
            traditional_value = getattr(traditional_metrics, metric)
            
            if llm_value > traditional_value + 0.05:  # Threshold for significance
                strengths_llm.append(description)
            elif traditional_value > llm_value + 0.05:
                strengths_traditional.append(description)
        
        return strengths_llm, strengths_traditional
    
    def _determine_winner(self, llm_metrics: QualityMetrics, 
                         traditional_metrics: QualityMetrics) -> Tuple[str, float]:
        """Determine overall winner and confidence score."""
        llm_score = llm_metrics.overall_quality_score
        traditional_score = traditional_metrics.overall_quality_score
        
        if llm_score > traditional_score:
            winner = "LLM Enhanced"
            confidence = min((llm_score - traditional_score) * 2, 1.0)
        elif traditional_score > llm_score:
            winner = "Traditional"
            confidence = min((traditional_score - llm_score) * 2, 1.0)
        else:
            winner = "Tie"
            confidence = 0.0
        
        return winner, confidence
    
    def _generate_detailed_analysis(self, llm_data: Dict[str, Any], 
                                  traditional_data: Dict[str, Any],
                                  llm_metrics: QualityMetrics, 
                                  traditional_metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate detailed analysis of the comparison."""
        
        analysis = {
            'topic_count_comparison': {
                'llm_enhanced': len(llm_data.get('normalized_topics', [])),
                'traditional': len(traditional_data.get('normalized_topics', [])),
            },
            'duplicate_analysis': {
                'llm_enhanced_duplicates': llm_metrics.duplicate_ratio,
                'traditional_duplicates': traditional_metrics.duplicate_ratio,
                'duplicate_reduction': traditional_metrics.duplicate_ratio - llm_metrics.duplicate_ratio
            },
            'academic_level_distribution': {
                'llm_enhanced': llm_metrics.academic_level_distribution,
                'traditional': traditional_metrics.academic_level_distribution
            },
            'quality_breakdown': {
                'llm_enhanced': {
                    'level_progression': llm_metrics.level_progression_score,
                    'topic_coherence': llm_metrics.topic_coherence_score,
                    'coverage_breadth': llm_metrics.coverage_breadth_score,
                    'pedagogical_ordering': llm_metrics.pedagogical_ordering_score,
                    'content_diversity': llm_metrics.content_diversity_score,
                    'structural_consistency': llm_metrics.structural_consistency_score
                },
                'traditional': {
                    'level_progression': traditional_metrics.level_progression_score,
                    'topic_coherence': traditional_metrics.topic_coherence_score,
                    'coverage_breadth': traditional_metrics.coverage_breadth_score,
                    'pedagogical_ordering': traditional_metrics.pedagogical_ordering_score,
                    'content_diversity': traditional_metrics.content_diversity_score,
                    'structural_consistency': traditional_metrics.structural_consistency_score
                }
            }
        }
        
        return analysis
    
    def _create_empty_metrics(self) -> QualityMetrics:
        """Create empty quality metrics for error cases."""
        return QualityMetrics(
            total_topics=0,
            unique_topics=0,
            duplicate_ratio=0.0,
            academic_level_distribution={},
            level_progression_score=0.0,
            topic_coherence_score=0.0,
            coverage_breadth_score=0.0,
            pedagogical_ordering_score=0.0,
            content_diversity_score=0.0,
            structural_consistency_score=0.0,
            overall_quality_score=0.0
        )
    
    def save_comparison_result(self, result: ComparisonResult, 
                              discipline: str, language: str) -> Path:
        """Save comparison result to cache."""
        cache_file = self.cache_dir / f"{discipline}_{language}_quality_comparison.json"
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'discipline': discipline,
            'language': language,
            'comparison_result': result.to_dict()
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality comparison results saved to: {cache_file}")
        return cache_file
    
    def load_comparison_result(self, discipline: str, language: str) -> Optional[ComparisonResult]:
        """Load cached comparison result."""
        cache_file = self.cache_dir / f"{discipline}_{language}_quality_comparison.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result_data = data['comparison_result']
            
            # Reconstruct ComparisonResult
            llm_metrics = QualityMetrics(**result_data['llm_enhanced_metrics'])
            traditional_metrics = QualityMetrics(**result_data['traditional_metrics'])
            
            result = ComparisonResult(
                llm_enhanced_metrics=llm_metrics,
                traditional_metrics=traditional_metrics,
                improvement_percentages=result_data['improvement_percentages'],
                strengths_llm=result_data['strengths_llm'],
                strengths_traditional=result_data['strengths_traditional'],
                overall_winner=result_data['overall_winner'],
                confidence_score=result_data['confidence_score'],
                detailed_analysis=result_data['detailed_analysis']
            )
            
            logger.info(f"Loaded cached quality comparison for {discipline}/{language}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading cached comparison result: {e}")
            return None