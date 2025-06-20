#!/usr/bin/env python3
"""
data_models.py - Enhanced data models for curriculum synthesis system

This module provides comprehensive data models for the curriculum synthesis system
with scientific validation, authority tracking, and cross-disciplinary management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class EducationalLevel(Enum):
    """Educational levels for curriculum subtopics."""
    HS_FOUND = "HS-Found"  # High School Foundations
    HS_ADV = "HS-Adv"      # High School Advanced
    UG_INTRO = "UG-Intro"  # Undergraduate Introductory
    UG_ADV = "UG-Adv"      # Undergraduate Advanced
    GRAD_INTRO = "Grad-Intro"  # Graduate Introductory
    GRAD_ADV = "Grad-Adv"      # Graduate Advanced


class BloomLevel(Enum):
    """Bloom's Taxonomy levels for cognitive complexity."""
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


class QuestionType(Enum):
    """Types of questions/assessments for subtopics."""
    COMPUTATIONAL = "Computational"  # Mathematical problem-solving
    CONCEPTUAL = "Conceptual"        # Theory and principle understanding
    GRAPHICAL = "Graphical"          # Visual representation and interpretation
    EXPERIMENTAL = "Experimental"    # Laboratory and observational methods


class AlertSeverity(Enum):
    """Severity levels for alerts and violations."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class SubtopicEntry:
    """
    Goal: Represent curriculum subtopic with comprehensive metadata and scientific validation.
    
    Each subtopic includes educational metadata, authority tracking, scientific validation,
    cross-disciplinary links, and discipline-specific contexts at individual subtopic level.
    """
    id: str
    discipline: str
    category: str  # Main branch (Level 2 in 6-level hierarchy)
    subtopic: str  # Fine-grained topic
    level: EducationalLevel
    bloom: BloomLevel
    standards_links: List[str]
    prerequisites: List[str]
    learning_objectives: List[str]
    textbook_references: List[Dict[str, str]]
    question_types: List[QuestionType]
    hierarchy_level: int  # 1-6 for 6-level hierarchy
    parent_topics: List[str]
    child_topics: List[str]
    
    # Individual subtopic-level discipline-specific context with validation
    discipline_specific_context: str
    discipline_specific_learning_objectives: List[str]
    discipline_specific_applications: List[str]
    discipline_specific_prerequisites: List[str]
    
    # Scientific validation and authority tracking
    scientific_principles_validated: bool = False
    authority_source: str = ""  # Most authoritative source for this topic
    authority_confidence: float = 0.0  # Confidence in authority ranking (0-1)
    scientific_principle_conflicts: List[str] = field(default_factory=list)
    
    # Cross-disciplinary management
    cross_disciplinary_links: Dict[str, str] = field(default_factory=dict)  # {discipline: context_in_that_discipline}
    conceptual_consistency_validated: bool = False
    
    # Pedagogical extensions (non-duplicated)
    common_misconceptions: List[str] = field(default_factory=list)
    key_equations: List[str] = field(default_factory=list)
    typical_examples: List[str] = field(default_factory=list)
    experimental_methods: List[str] = field(default_factory=list)
    
    # Additional metadata
    mcat_relevance: bool = False
    gap_fill_source: Optional[str] = None  # Source if auto-added to fill gaps
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'discipline': self.discipline,
            'category': self.category,
            'subtopic': self.subtopic,
            'level': self.level.value if isinstance(self.level, EducationalLevel) else self.level,
            'bloom': self.bloom.value if isinstance(self.bloom, BloomLevel) else self.bloom,
            'standards_links': self.standards_links,
            'prerequisites': self.prerequisites,
            'learning_objectives': self.learning_objectives,
            'textbook_references': self.textbook_references,
            'question_types': [qt.value if isinstance(qt, QuestionType) else qt for qt in self.question_types],
            'hierarchy_level': self.hierarchy_level,
            'parent_topics': self.parent_topics,
            'child_topics': self.child_topics,
            'discipline_specific_context': self.discipline_specific_context,
            'discipline_specific_learning_objectives': self.discipline_specific_learning_objectives,
            'discipline_specific_applications': self.discipline_specific_applications,
            'discipline_specific_prerequisites': self.discipline_specific_prerequisites,
            'scientific_principles_validated': self.scientific_principles_validated,
            'authority_source': self.authority_source,
            'authority_confidence': self.authority_confidence,
            'scientific_principle_conflicts': self.scientific_principle_conflicts,
            'cross_disciplinary_links': self.cross_disciplinary_links,
            'conceptual_consistency_validated': self.conceptual_consistency_validated,
            'common_misconceptions': self.common_misconceptions,
            'key_equations': self.key_equations,
            'typical_examples': self.typical_examples,
            'experimental_methods': self.experimental_methods,
            'mcat_relevance': self.mcat_relevance,
            'gap_fill_source': self.gap_fill_source,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ScientificViolation:
    """
    Goal: Track violations of fundamental scientific principles in discipline contexts.
    
    Identifies conflicts between discipline-specific contexts and established scientific
    principles, enabling authority-based resolution.
    """
    subtopic_id: str
    discipline: str
    violation_type: str  # Contradiction, Inconsistency, Factual_Error, Nomenclature_Error
    conflicting_principle: str
    description: str
    severity: AlertSeverity
    resolution_recommendation: str
    authoritative_sources: List[str]
    confidence_score: float = 0.0  # Confidence in violation detection (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'subtopic_id': self.subtopic_id,
            'discipline': self.discipline,
            'violation_type': self.violation_type,
            'conflicting_principle': self.conflicting_principle,
            'description': self.description,
            'severity': self.severity.value if isinstance(self.severity, AlertSeverity) else self.severity,
            'resolution_recommendation': self.resolution_recommendation,
            'authoritative_sources': self.authoritative_sources,
            'confidence_score': self.confidence_score
        }


@dataclass
class DualPriorityAlert:
    """
    Goal: Alert system prioritizing both standards importance and missing topic count.
    
    Combines standards framework importance with missing topic impact to create
    composite priority scores for actionable alert prioritization.
    """
    discipline: str
    alert_type: str  # Coverage, Consistency, Scientific_Violation, Gap
    standards_importance_score: float  # 0-1 based on standards criticality
    missing_topics_count: int
    composite_priority_score: float  # Calculated from both factors
    severity: AlertSeverity
    recommendations: List[str]
    affected_standards: List[str]
    specific_missing_topics: List[str] = field(default_factory=list)
    textbook_recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'discipline': self.discipline,
            'alert_type': self.alert_type,
            'standards_importance_score': self.standards_importance_score,
            'missing_topics_count': self.missing_topics_count,
            'composite_priority_score': self.composite_priority_score,
            'severity': self.severity.value if isinstance(self.severity, AlertSeverity) else self.severity,
            'recommendations': self.recommendations,
            'affected_standards': self.affected_standards,
            'specific_missing_topics': self.specific_missing_topics,
            'textbook_recommendations': self.textbook_recommendations
        }


@dataclass
class ConfigurableCoverageThreshold:
    """
    Goal: Store configurable coverage thresholds per standards framework.
    
    Enables framework-specific coverage requirements instead of uniform thresholds,
    allowing higher standards for critical frameworks like MCAT.
    """
    framework_name: str
    threshold: float  # 0-1 coverage requirement
    justification: str
    importance_weight: float = 1.0  # For dual-priority calculations
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'framework_name': self.framework_name,
            'threshold': self.threshold,
            'justification': self.justification,
            'importance_weight': self.importance_weight,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ConsistencyValidationResult:
    """
    Goal: Track conceptual consistency validation across disciplines.
    
    Documents cross-disciplinary consistency validation results and authority-based
    conflict resolution decisions.
    """
    topic: str
    disciplines: List[str]
    consistency_score: float  # 0-1 consistency across disciplines
    conflicts_detected: List[str]
    resolution_recommendations: List[str]
    validation_passed: bool
    authority_resolution_applied: bool = False
    authoritative_source_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'topic': self.topic,
            'disciplines': self.disciplines,
            'consistency_score': self.consistency_score,
            'conflicts_detected': self.conflicts_detected,
            'resolution_recommendations': self.resolution_recommendations,
            'validation_passed': self.validation_passed,
            'authority_resolution_applied': self.authority_resolution_applied,
            'authoritative_source_used': self.authoritative_source_used
        }


@dataclass
class BookMetadata:
    """
    Goal: Store comprehensive book metadata for difficulty ranking and analysis.
    
    Contains metadata for ranking books from introductory to advanced and
    tracking authority sources for conflict resolution.
    """
    title: str
    language: str
    level: str
    discipline: str
    file_path: str
    difficulty_score: float = 0.0
    audience_indicators: List[str] = field(default_factory=list)
    toc_complexity: Dict[str, float] = field(default_factory=dict)
    authority_score: float = 0.0  # Authority ranking for conflict resolution
    author_credentials: List[str] = field(default_factory=list)
    publication_info: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'language': self.language,
            'level': self.level,
            'discipline': self.discipline,
            'file_path': self.file_path,
            'difficulty_score': self.difficulty_score,
            'audience_indicators': self.audience_indicators,
            'toc_complexity': self.toc_complexity,
            'authority_score': self.authority_score,
            'author_credentials': self.author_credentials,
            'publication_info': self.publication_info
        }


@dataclass
class TOCEntry:
    """
    Goal: Represent table of contents entry with hierarchical structure.
    
    Preserves 6-level hierarchy and enables prerequisite dependency tracking.
    """
    title: str
    level: int  # 1-6 hierarchy level
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    book_source: str = ""
    page_number: Optional[int] = None
    section_number: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'title': self.title,
            'level': self.level,
            'parent': self.parent,
            'children': self.children,
            'book_source': self.book_source,
            'page_number': self.page_number,
            'section_number': self.section_number
        }


@dataclass
class StandardsFramework:
    """
    Goal: Represent educational standards framework with authority tracking.
    
    Stores standards framework information with authority scores for
    dual-priority alert generation and conflict resolution.
    """
    name: str
    discipline: str
    outcomes: List[str]
    authority_score: float = 1.0
    coverage_threshold: float = 0.95
    importance_weight: float = 1.0
    source_url: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'discipline': self.discipline,
            'outcomes': self.outcomes,
            'authority_score': self.authority_score,
            'coverage_threshold': self.coverage_threshold,
            'importance_weight': self.importance_weight,
            'source_url': self.source_url,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ValidationReport:
    """
    Goal: Comprehensive validation report with all validation results.
    
    Aggregates coverage validation, scientific validation, consistency validation,
    and alert generation results into unified report.
    """
    discipline: str
    total_subtopics: int
    coverage_results: Dict[str, float]  # Framework -> coverage percentage
    scientific_violations: List[ScientificViolation]
    consistency_results: List[ConsistencyValidationResult]
    dual_priority_alerts: List[DualPriorityAlert]
    overall_completeness_score: float
    generation_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'discipline': self.discipline,
            'total_subtopics': self.total_subtopics,
            'coverage_results': self.coverage_results,
            'scientific_violations': [v.to_dict() for v in self.scientific_violations],
            'consistency_results': [c.to_dict() for c in self.consistency_results],
            'dual_priority_alerts': [a.to_dict() for a in self.dual_priority_alerts],
            'overall_completeness_score': self.overall_completeness_score,
            'generation_timestamp': self.generation_timestamp.isoformat()
        }


@dataclass
class DisciplineInfo:
    """
    Goal: Store comprehensive discipline information for processing.
    
    Contains all metadata needed for discipline-specific curriculum generation
    including cross-disciplinary relationships and authority sources.
    """
    name: str
    available_books: List[BookMetadata]
    languages: List[str]
    levels: List[str]
    mcat_relevant: bool = False
    cross_disciplinary_topics: List[str] = field(default_factory=list)
    authority_sources: Dict[str, float] = field(default_factory=dict)  # Source -> authority score
    standards_frameworks: List[StandardsFramework] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'available_books': [book.to_dict() for book in self.available_books],
            'languages': self.languages,
            'levels': self.levels,
            'mcat_relevant': self.mcat_relevant,
            'cross_disciplinary_topics': self.cross_disciplinary_topics,
            'authority_sources': self.authority_sources,
            'standards_frameworks': [sf.to_dict() for sf in self.standards_frameworks]
        }


# Utility functions for data model operations
def create_subtopic_id(discipline: str, category: str, subtopic: str) -> str:
    """
    Goal: Generate unique subtopic ID for cross-referencing.
    
    Creates consistent, unique identifiers for subtopics to enable
    cross-disciplinary linking and validation tracking.
    """
    return f"{discipline.lower().replace(' ', '_')}_{category.lower().replace(' ', '_')}_{subtopic.lower().replace(' ', '_')}"


def calculate_composite_priority(standards_score: float, topic_count: int, 
                               standards_weight: float = 0.7, 
                               count_weight: float = 0.3) -> float:
    """
    Goal: Calculate composite priority score for dual-priority alerts.
    
    Combines standards importance and missing topic count into unified
    priority score for alert ranking and resource allocation.
    """
    # Normalize topic count to 0-1 scale (assuming max 100 missing topics)
    normalized_count = min(topic_count / 100.0, 1.0)
    
    # Calculate weighted composite score
    composite = (standards_score * standards_weight) + (normalized_count * count_weight)
    return min(composite, 1.0)


def determine_alert_severity(composite_score: float) -> AlertSeverity:
    """
    Goal: Determine alert severity from composite priority score.
    
    Converts numeric priority scores into actionable severity levels
    for alert triage and response prioritization.
    """
    if composite_score >= 0.8:
        return AlertSeverity.CRITICAL
    elif composite_score >= 0.6:
        return AlertSeverity.HIGH
    elif composite_score >= 0.4:
        return AlertSeverity.MEDIUM
    else:
        return AlertSeverity.LOW