#!/usr/bin/env python3
"""
enhanced_toc_extractor.py - Enhanced TOC extraction with scientific validation

This module extracts table of contents information from books and validates
it against scientific principles, ensuring accuracy and consistency.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import difflib

from .data_models import TOCEntry, ScientificViolation, AlertSeverity
from .scientific_principle_validator import ScientificPrincipleValidator

logger = logging.getLogger(__name__)


@dataclass
class TOCValidationResult:
    """
    Goal: Store comprehensive TOC validation results.
    
    Contains validation outcomes, detected issues, and recommendations
    for TOC quality and scientific accuracy.
    """
    toc_entry: TOCEntry
    is_valid: bool
    scientific_violations: List[ScientificViolation]
    quality_score: float  # 0-1 scale
    validation_notes: List[str]
    recommended_improvements: List[str]
    authority_alignment_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'toc_entry_id': self.toc_entry.id if hasattr(self.toc_entry, 'id') else None,
            'toc_title': self.toc_entry.title,
            'is_valid': self.is_valid,
            'scientific_violations': [v.__dict__ for v in self.scientific_violations],
            'quality_score': self.quality_score,
            'validation_notes': self.validation_notes,
            'recommended_improvements': self.recommended_improvements,
            'authority_alignment_score': self.authority_alignment_score
        }


@dataclass
class TOCExtractionMetrics:
    """
    Goal: Track comprehensive metrics for TOC extraction quality.
    
    Provides detailed analytics on extraction success, validation results,
    and quality indicators for continuous improvement.
    """
    total_entries_processed: int
    successfully_extracted: int
    scientifically_validated: int
    quality_distribution: Dict[str, int]  # High, Medium, Low
    common_violations: Dict[str, int]
    extraction_confidence_scores: List[float]
    discipline_coverage: Dict[str, int]
    
    def get_success_rate(self) -> float:
        """Calculate overall extraction success rate."""
        return self.successfully_extracted / self.total_entries_processed if self.total_entries_processed > 0 else 0.0
    
    def get_validation_rate(self) -> float:
        """Calculate scientific validation success rate."""
        return self.scientifically_validated / self.successfully_extracted if self.successfully_extracted > 0 else 0.0


class EnhancedTOCExtractor:
    """
    Goal: Extract and validate table of contents with scientific accuracy.
    
    Provides comprehensive TOC extraction with scientific principle validation,
    quality assessment, and authority source verification.
    """
    
    def __init__(self):
        self.scientific_validator = ScientificPrincipleValidator()
        self.quality_thresholds = self._load_quality_thresholds()
        self.extraction_patterns = self._load_extraction_patterns()
        self.discipline_validators = self._load_discipline_validators()
        self.authority_terminology = self._load_authority_terminology()
        self.extraction_metrics = TOCExtractionMetrics(
            total_entries_processed=0,
            successfully_extracted=0,
            scientifically_validated=0,
            quality_distribution={'High': 0, 'Medium': 0, 'Low': 0},
            common_violations={},
            extraction_confidence_scores=[],
            discipline_coverage={}
        )
    
    def extract_and_validate_toc_entries(self, book_list_data: Dict, 
                                       discipline_mappings: Dict) -> Tuple[List[TOCEntry], List[TOCValidationResult]]:
        """
        Goal: Extract TOC entries with comprehensive scientific validation.
        
        Processes book collection to extract TOC information and validates
        against scientific principles, terminology standards, and quality metrics.
        """
        logger.info(f"Starting enhanced TOC extraction for {len(book_list_data)} books")
        
        all_toc_entries = []
        all_validation_results = []
        
        for book_key, book_info in book_list_data.items():
            if not isinstance(book_info, dict):
                continue
            
            # Extract raw TOC entries
            raw_toc_entries = self._extract_raw_toc_entries(book_key, book_info)
            
            # Enhance TOC entries with metadata
            enhanced_entries = self._enhance_toc_entries(raw_toc_entries, book_info, discipline_mappings)
            
            # Validate each TOC entry
            validation_results = []
            for entry in enhanced_entries:
                validation_result = self._validate_toc_entry_comprehensive(entry, discipline_mappings)
                validation_results.append(validation_result)
                
                # Update metrics
                self.extraction_metrics.total_entries_processed += 1
                if validation_result.is_valid:
                    self.extraction_metrics.successfully_extracted += 1
                if not validation_result.scientific_violations:
                    self.extraction_metrics.scientifically_validated += 1
                
                # Track quality distribution
                quality_level = self._determine_quality_level(validation_result.quality_score)
                self.extraction_metrics.quality_distribution[quality_level] += 1
                
                # Track violations
                for violation in validation_result.scientific_violations:
                    violation_type = violation.violation_type
                    self.extraction_metrics.common_violations[violation_type] = \
                        self.extraction_metrics.common_violations.get(violation_type, 0) + 1
            
            all_toc_entries.extend(enhanced_entries)
            all_validation_results.extend(validation_results)
        
        # Apply post-processing improvements
        improved_entries, improved_validations = self._apply_post_processing_improvements(
            all_toc_entries, all_validation_results, discipline_mappings
        )
        
        logger.info(f"Enhanced TOC extraction complete: {len(improved_entries)} entries, "
                   f"{self.extraction_metrics.get_success_rate():.1%} success rate, "
                   f"{self.extraction_metrics.get_validation_rate():.1%} validation rate")
        
        return improved_entries, improved_validations
    
    def _extract_raw_toc_entries(self, book_key: str, book_info: Dict) -> List[TOCEntry]:
        """Extract raw TOC entries from book information."""
        toc_entries = []
        
        # Try to extract from different possible TOC formats
        toc_data = book_info.get('table_of_contents', None)
        if not toc_data:
            toc_data = book_info.get('contents', None)
        if not toc_data:
            toc_data = book_info.get('chapters', None)
        
        if isinstance(toc_data, list):
            # List format
            for i, item in enumerate(toc_data):
                entry = self._parse_toc_item(item, book_key, book_info, i)
                if entry:
                    toc_entries.append(entry)
                    
        elif isinstance(toc_data, dict):
            # Dictionary format
            for key, value in toc_data.items():
                entry = self._parse_toc_dict_item(key, value, book_key, book_info)
                if entry:
                    toc_entries.append(entry)
                    
        elif isinstance(toc_data, str):
            # String format - need to parse
            parsed_entries = self._parse_toc_string(toc_data, book_key, book_info)
            toc_entries.extend(parsed_entries)
        
        # If no explicit TOC, try to infer from title/description
        if not toc_entries:
            inferred_entries = self._infer_toc_from_metadata(book_key, book_info)
            toc_entries.extend(inferred_entries)
        
        return toc_entries
    
    def _parse_toc_item(self, item, book_key: str, book_info: Dict, index: int) -> Optional[TOCEntry]:
        """Parse individual TOC item from list format."""
        try:
            if isinstance(item, str):
                # Simple string format
                title = item.strip()
                if not title:
                    return None
                
                return TOCEntry(
                    title=title,
                    level=self._infer_level_from_title(title),
                    parent=None,
                    children=[],
                    book_source=book_info.get('title', book_key),
                    page_number=None,
                    section_number=str(index + 1)
                )
                
            elif isinstance(item, dict):
                # Dictionary format with metadata
                title = item.get('title', item.get('name', ''))
                if not title:
                    return None
                
                return TOCEntry(
                    title=title.strip(),
                    level=item.get('level', self._infer_level_from_title(title)),
                    parent=item.get('parent', None),
                    children=item.get('subsections', []),
                    book_source=book_info.get('title', book_key),
                    page_number=item.get('page', item.get('page_number', None)),
                    section_number=item.get('section', str(index + 1))
                )
        
        except Exception as e:
            logger.warning(f"Error parsing TOC item {item}: {e}")
            return None
    
    def _parse_toc_dict_item(self, key: str, value, book_key: str, book_info: Dict) -> Optional[TOCEntry]:
        """Parse TOC item from dictionary format."""
        try:
            title = key.strip()
            if not title:
                return None
            
            description = None
            subsections = []
            
            if isinstance(value, str):
                description = value
            elif isinstance(value, dict):
                description = value.get('description', value.get('summary', None))
                subsections = value.get('subsections', [])
            elif isinstance(value, list):
                subsections = value
            
            return TOCEntry(
                title=title,
                level=self._infer_level_from_title(title),
                parent=None,
                children=[],
                book_source=book_info.get('title', book_key),
                page_number=None,
                section_number=""
            )
            
        except Exception as e:
            logger.warning(f"Error parsing TOC dict item {key}: {e}")
            return None
    
    def _parse_toc_string(self, toc_string: str, book_key: str, book_info: Dict) -> List[TOCEntry]:
        """Parse TOC from string format."""
        entries = []
        
        # Split by common delimiters
        lines = re.split(r'\\n|\\r\\n|\\r|\n|\r', toc_string)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Remove common TOC formatting
            clean_line = re.sub(r'^(chapter|section|part|unit)\\s*\\d+[:\\-\\.]?\\s*', '', line, flags=re.IGNORECASE)
            clean_line = re.sub(r'\\.*\\d+$', '', clean_line)  # Remove page numbers
            clean_line = clean_line.strip()
            
            if clean_line and len(clean_line) > 3:  # Minimum reasonable title length
                entry = TOCEntry(
                    title=clean_line,
                    level=self._infer_level_from_formatting(line),
                    parent=None,
                    children=[],
                    book_source=book_info.get('title', book_key),
                    page_number=self._extract_page_number(line),
                    section_number=self._extract_section_number(line)
                )
                entries.append(entry)
        
        return entries
    
    def _infer_toc_from_metadata(self, book_key: str, book_info: Dict) -> List[TOCEntry]:
        """Infer TOC entries from book metadata when explicit TOC is not available."""
        entries = []
        
        title = book_info.get('title', '')
        description = book_info.get('description', '')
        
        # Try to identify chapter/section patterns in description
        if description:
            chapter_matches = re.findall(r'(chapter|section|part)\\s*\\d+[:\\-\\.]?\\s*([^\\n\\r\\.;]{10,80})', 
                                       description, flags=re.IGNORECASE)
            
            for i, (section_type, section_title) in enumerate(chapter_matches):
                entry = TOCEntry(
                    title=section_title.strip(),
                    level=1 if section_type.lower() == 'chapter' else 2,
                    parent=None,
                    children=[],
                    book_source=book_info.get('title', book_key),
                    page_number=None,
                    section_number=str(i + 1)
                )
                entries.append(entry)
        
        # If still no entries, create a generic entry from the book title
        if not entries and title:
            generic_entry = TOCEntry(
                title=title,
                level=0,
                parent=None,
                children=[],
                book_source=title,
                page_number=None,
                section_number="1"
            )
            entries.append(generic_entry)
        
        return entries
    
    def _enhance_toc_entries(self, toc_entries: List[TOCEntry], book_info: Dict, 
                           discipline_mappings: Dict) -> List[TOCEntry]:
        """Enhance TOC entries with additional metadata and validation."""
        enhanced_entries = []
        
        book_title = book_info.get('title', '')
        book_disciplines = self._identify_book_disciplines(book_title, discipline_mappings)
        
        for entry in toc_entries:
            # Enhanced keyword extraction
            enhanced_keywords = self._enhanced_keyword_extraction(entry, book_disciplines)
            # Note: Keywords enhancement would be applied if TOCEntry supported keywords field
            
            # Note: Complexity estimation would be applied if TOCEntry supported estimated_complexity field
            
            # Note: Discipline context would be applied if TOCEntry supported discipline_context field
            
            # Note: Terminology validation would be applied if TOCEntry supported terminology_validated field
            
            enhanced_entries.append(entry)
        
        return enhanced_entries
    
    def _validate_toc_entry_comprehensive(self, entry: TOCEntry, 
                                        discipline_mappings: Dict) -> TOCValidationResult:
        """Perform comprehensive validation of TOC entry."""
        validation_notes = []
        recommended_improvements = []
        scientific_violations = []
        
        # Determine disciplines for this entry
        entry_disciplines = self._determine_entry_disciplines(entry, discipline_mappings)
        
        # Scientific principle validation
        for discipline in entry_disciplines:
            violations = self.scientific_validator.validate_scientific_consistency(
                [self._convert_toc_to_subtopic(entry, discipline)], discipline
            )
            scientific_violations.extend(violations)
        
        # Quality assessment
        quality_score = self._assess_entry_quality(entry, entry_disciplines)
        
        # Authority terminology alignment
        authority_alignment_score = self._assess_authority_alignment(entry, entry_disciplines)
        
        # Generate validation notes
        if quality_score < 0.5:
            validation_notes.append("Low quality score - consider improving title clarity")
            recommended_improvements.append("Revise title for better clarity and specificity")
        
        if authority_alignment_score < 0.7:
            validation_notes.append("Poor alignment with authority terminology standards")
            recommended_improvements.append("Use standard scientific terminology")
        
        if scientific_violations:
            validation_notes.append(f"Found {len(scientific_violations)} scientific violations")
            recommended_improvements.append("Review content for scientific accuracy")
        
        # Determine overall validity
        is_valid = (quality_score >= 0.5 and 
                   authority_alignment_score >= 0.6 and 
                   len(scientific_violations) == 0)
        
        return TOCValidationResult(
            toc_entry=entry,
            is_valid=is_valid,
            scientific_violations=scientific_violations,
            quality_score=quality_score,
            validation_notes=validation_notes,
            recommended_improvements=recommended_improvements,
            authority_alignment_score=authority_alignment_score
        )
    
    def _apply_post_processing_improvements(self, toc_entries: List[TOCEntry], 
                                          validation_results: List[TOCValidationResult],
                                          discipline_mappings: Dict) -> Tuple[List[TOCEntry], List[TOCValidationResult]]:
        """Apply post-processing improvements based on validation results."""
        improved_entries = []
        improved_validations = []
        
        for entry, validation in zip(toc_entries, validation_results):
            improved_entry = entry
            
            # Apply automatic improvements for common issues
            if validation.quality_score < 0.5:
                improved_entry = self._improve_entry_quality(entry)
            
            if validation.authority_alignment_score < 0.7:
                improved_entry = self._improve_authority_alignment(improved_entry, discipline_mappings)
            
            # Re-validate if improvements were made
            if improved_entry != entry:
                improved_validation = self._validate_toc_entry_comprehensive(improved_entry, discipline_mappings)
                improved_validations.append(improved_validation)
            else:
                improved_validations.append(validation)
            
            improved_entries.append(improved_entry)
        
        return improved_entries, improved_validations
    
    def _improve_entry_quality(self, entry: TOCEntry) -> TOCEntry:
        """Apply automatic quality improvements to TOC entry."""
        improved_entry = entry
        
        # Improve title formatting
        improved_title = self._improve_title_formatting(entry.title)
        if improved_title != entry.title:
            improved_entry.title = improved_title
        
        # Add description if missing and can be inferred
        if not getattr(entry, "description", ""):
            inferred_description = self._infer_description(entry)
            if inferred_description:
                improved_entry.description = inferred_description
        
        # Improve keyword extraction
        improved_keywords = self._improve_keyword_extraction(entry)
        # Note: Keywords would be set if TOCEntry supported keywords field
        
        return improved_entry
    
    def _improve_authority_alignment(self, entry: TOCEntry, discipline_mappings: Dict) -> TOCEntry:
        """Improve alignment with authority terminology standards."""
        improved_entry = entry
        
        # Get disciplines for this entry
        entry_disciplines = self._determine_entry_disciplines(entry, discipline_mappings)
        
        # Apply terminology corrections
        for discipline in entry_disciplines:
            corrected_title = self._apply_terminology_corrections(entry.title, discipline)
            if corrected_title != entry.title:
                improved_entry.title = corrected_title
            
            if getattr(entry, "description", ""):
                corrected_description = self._apply_terminology_corrections(getattr(entry, "description", ""), discipline)
                if corrected_description != getattr(entry, "description", ""):
                    improved_entry.description = corrected_description
        
        return improved_entry
    
    def get_extraction_report(self) -> Dict:
        """Generate comprehensive extraction and validation report."""
        return {
            'extraction_metrics': {
                'total_processed': self.extraction_metrics.total_entries_processed,
                'successfully_extracted': self.extraction_metrics.successfully_extracted,
                'scientifically_validated': self.extraction_metrics.scientifically_validated,
                'success_rate': self.extraction_metrics.get_success_rate(),
                'validation_rate': self.extraction_metrics.get_validation_rate()
            },
            'quality_distribution': self.extraction_metrics.quality_distribution,
            'common_violations': dict(sorted(self.extraction_metrics.common_violations.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]),
            'discipline_coverage': self.extraction_metrics.discipline_coverage,
            'recommendations': self._generate_extraction_recommendations()
        }
    
    def _generate_extraction_recommendations(self) -> List[str]:
        """Generate recommendations for improving extraction quality."""
        recommendations = []
        
        success_rate = self.extraction_metrics.get_success_rate()
        validation_rate = self.extraction_metrics.get_validation_rate()
        
        if success_rate < 0.8:
            recommendations.append("Consider improving TOC parsing algorithms for better extraction success")
        
        if validation_rate < 0.9:
            recommendations.append("Enhance scientific validation procedures to reduce violations")
        
        # Analyze common violations
        if self.extraction_metrics.common_violations:
            most_common = max(self.extraction_metrics.common_violations.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on reducing {most_common[0]} violations (most common)")
        
        # Analyze quality distribution
        low_quality_count = self.extraction_metrics.quality_distribution.get('Low', 0)
        total_quality = sum(self.extraction_metrics.quality_distribution.values())
        
        if total_quality > 0 and low_quality_count / total_quality > 0.2:
            recommendations.append("Improve automatic quality enhancement procedures")
        
        return recommendations
    
    # Helper methods for TOC processing
    def _infer_level_from_title(self, title: str) -> int:
        """Infer hierarchical level from title patterns."""
        title_lower = title.lower()
        
        # Level indicators
        if any(indicator in title_lower for indicator in ['chapter', 'part', 'unit']):
            return 1
        elif any(indicator in title_lower for indicator in ['section', 'lesson']):
            return 2
        elif any(indicator in title_lower for indicator in ['subsection', 'topic']):
            return 3
        else:
            # Use indentation or numbering patterns
            if re.match(r'^\\s{4,}', title):  # Indented
                return 3
            elif re.match(r'^\\s{2,}', title):
                return 2
            else:
                return 1
    
    def _infer_level_from_formatting(self, line: str) -> int:
        """Infer level from line formatting."""
        # Count leading whitespace
        leading_spaces = len(line) - len(line.lstrip())
        
        if leading_spaces >= 8:
            return 3
        elif leading_spaces >= 4:
            return 2
        else:
            return 1
    
    def _extract_keywords_from_title(self, title: str) -> List[str]:
        """Extract relevant keywords from title."""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Clean and split title
        clean_title = re.sub(r'[^\\w\\s]', ' ', title.lower())
        words = [word.strip() for word in clean_title.split() if word.strip()]
        
        # Filter meaningful words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _estimate_complexity_from_title(self, title: str) -> float:
        """Estimate complexity from title content."""
        title_lower = title.lower()
        
        complexity_indicators = {
            'basic': -0.2, 'introduction': -0.2, 'elementary': -0.2, 'fundamental': -0.1,
            'intermediate': 0.0, 'general': 0.0,
            'advanced': 0.3, 'specialized': 0.3, 'graduate': 0.4, 'research': 0.5,
            'theory': 0.2, 'theoretical': 0.2, 'mathematical': 0.2, 'quantum': 0.3
        }
        
        base_complexity = 0.5  # Default intermediate level
        
        for indicator, adjustment in complexity_indicators.items():
            if indicator in title_lower:
                base_complexity += adjustment
        
        return max(0.0, min(1.0, base_complexity))
    
    def _extract_page_number(self, line: str) -> Optional[int]:
        """Extract page number from TOC line."""
        # Look for numbers at the end of the line
        match = re.search(r'\\b(\\d+)\\s*$', line)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_section_number(self, line: str) -> Optional[str]:
        """Extract section number from TOC line."""
        # Look for section numbering patterns
        patterns = [
            r'^(\\d+(?:\\.\\d+)*)',  # 1.2.3 format
            r'^([IVXLCDM]+)',        # Roman numerals
            r'^([A-Z])',             # Single letters
            r'(chapter|section)\\s*(\\d+)', # Chapter/Section numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line.strip(), re.IGNORECASE)
            if match:
                return match.group(1) if len(pattern.split('(')) == 2 else match.group(2)
        
        return None
    
    def _enhanced_keyword_extraction(self, entry: TOCEntry, disciplines: List[str]) -> List[str]:
        """Enhanced keyword extraction considering discipline context."""
        base_keywords = getattr(entry, 'keywords', []) or []
        
        # Add discipline-specific terminology
        enhanced_keywords = set(base_keywords)
        
        text_to_analyze = f"{entry.title} {getattr(entry, 'description', '') or ''}".lower()
        
        for discipline in disciplines:
            discipline_terms = self.discipline_validators.get(discipline, {}).get('key_terms', [])
            for term in discipline_terms:
                if term.lower() in text_to_analyze:
                    enhanced_keywords.add(term)
        
        return list(enhanced_keywords)[:15]  # Limit to 15 keywords
    
    def _enhanced_complexity_estimation(self, entry: TOCEntry, disciplines: List[str]) -> float:
        """Enhanced complexity estimation using discipline-specific indicators."""
        base_complexity = getattr(entry, 'estimated_complexity', 0.5) or 0.5
        
        # Adjust based on discipline-specific complexity indicators
        text = f"{entry.title} {getattr(entry, "description", "") or ''}".lower()
        
        for discipline in disciplines:
            complexity_adjustments = self.discipline_validators.get(discipline, {}).get('complexity_indicators', {})
            
            for indicator, adjustment in complexity_adjustments.items():
                if indicator in text:
                    base_complexity += adjustment
        
        return max(0.0, min(1.0, base_complexity))
    
    def _generate_discipline_context(self, entry: TOCEntry, disciplines: List[str]) -> str:
        """Generate discipline-specific context for TOC entry."""
        if not disciplines:
            return ""
        
        primary_discipline = disciplines[0]
        return f"This topic is primarily relevant to {primary_discipline} and relates to {', '.join(disciplines[1:]) if len(disciplines) > 1 else 'fundamental concepts'}."
    
    def _validate_scientific_terminology(self, entry: TOCEntry, disciplines: List[str]) -> bool:
        """Validate scientific terminology in TOC entry."""
        text = f"{entry.title} {getattr(entry, "description", "") or ''}".lower()
        
        for discipline in disciplines:
            terminology_standards = self.authority_terminology.get(discipline, {})
            
            # Check for incorrect terminology
            for incorrect, correct in terminology_standards.get('corrections', {}).items():
                if incorrect.lower() in text:
                    return False  # Found incorrect terminology
        
        return True  # No terminology issues found
    
    def _identify_book_disciplines(self, book_title: str, discipline_mappings: Dict) -> List[str]:
        """Identify disciplines relevant to a book."""
        disciplines = []
        
        title_lower = book_title.lower()
        
        # Check against known discipline mappings
        for mapping in discipline_mappings.values():
            if hasattr(mapping, 'book_title') and mapping.book_title.lower() == title_lower:
                disciplines.append(mapping.primary_discipline)
                disciplines.extend(mapping.secondary_disciplines)
                break
        
        # Fallback: infer from title keywords
        if not disciplines:
            discipline_keywords = {
                'Physics': ['physics', 'mechanics', 'quantum', 'electromagnetism'],
                'Chemistry': ['chemistry', 'chemical', 'molecular', 'organic'],
                'Biology': ['biology', 'biological', 'life', 'cell', 'genetics'],
                'Mathematics': ['mathematics', 'calculus', 'algebra', 'geometry']
            }
            
            for discipline, keywords in discipline_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    disciplines.append(discipline)
        
        return disciplines[:3]  # Limit to 3 disciplines
    
    def _determine_entry_disciplines(self, entry: TOCEntry, discipline_mappings: Dict) -> List[str]:
        """Determine disciplines relevant to a specific TOC entry."""
        # First try to get from book mapping
        book_disciplines = self._identify_book_disciplines(entry.book_source, discipline_mappings)
        
        # Then check entry-specific indicators
        entry_text = f"{entry.title} {getattr(entry, "description", "") or ''}".lower()
        entry_disciplines = []
        
        for discipline, validator_info in self.discipline_validators.items():
            key_terms = validator_info.get('key_terms', [])
            if any(term.lower() in entry_text for term in key_terms):
                entry_disciplines.append(discipline)
        
        # Combine and deduplicate
        all_disciplines = book_disciplines + entry_disciplines
        unique_disciplines = []
        for disc in all_disciplines:
            if disc not in unique_disciplines:
                unique_disciplines.append(disc)
        
        return unique_disciplines[:2]  # Limit to 2 most relevant disciplines
    
    def _convert_toc_to_subtopic(self, entry: TOCEntry, discipline: str):
        """Convert TOC entry to SubtopicEntry for validation."""
        # This is a simplified conversion for validation purposes
        from .data_models import SubtopicEntry, EducationalLevel, BloomLevel
        
        return SubtopicEntry(
            id=f"toc_{hash(entry.title) % 10000}",
            discipline=discipline,
            category="General",
            subtopic=entry.title,
            level=EducationalLevel.UG_INTRO,
            bloom=BloomLevel.UNDERSTAND,
            standards_links=[],
            prerequisites=[],
            learning_objectives=[f"Understand {entry.title}"],
            textbook_references=[],
            question_types=[],
            hierarchy_level=3,
            parent_topics=[],
            child_topics=[],
            discipline_specific_context=getattr(entry, "description", "") or "",
            discipline_specific_learning_objectives=[],
            discipline_specific_applications=[],
            discipline_specific_prerequisites=[],
            scientific_principles_validated=False,
            authority_source="TOC_Entry",
            authority_confidence=0.5,
            scientific_principle_conflicts=[],
            cross_disciplinary_links={},
            conceptual_consistency_validated=False,
            common_misconceptions=[],
            key_equations=[],
            typical_examples=[],
            experimental_methods=[]
        )
    
    def _assess_entry_quality(self, entry: TOCEntry, disciplines: List[str]) -> float:
        """Assess the overall quality of a TOC entry."""
        quality_score = 0.0
        
        # Title quality (40% of score)
        title_score = self._assess_title_quality(entry.title)
        quality_score += title_score * 0.4
        
        # Description quality (20% of score)
        if getattr(entry, "description", ""):
            desc_score = self._assess_description_quality(getattr(entry, "description", ""))
            quality_score += desc_score * 0.2
        
        # Keyword relevance (20% of score)
        keyword_score = self._assess_keyword_quality(getattr(entry, 'keywords', []), disciplines)
        quality_score += keyword_score * 0.2
        
        # Completeness (20% of score)
        completeness_score = self._assess_completeness(entry)
        quality_score += completeness_score * 0.2
        
        return min(1.0, quality_score)
    
    def _assess_title_quality(self, title: str) -> float:
        """Assess the quality of a TOC entry title."""
        if not title:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness
        length = len(title.split())
        if 2 <= length <= 8:  # Optimal length
            score += 0.2
        elif length == 1 or length > 12:  # Too short or too long
            score -= 0.2
        
        # Clarity indicators
        if not re.search(r'[^\\w\\s]', title):  # No special characters
            score += 0.1
        
        # Specificity
        vague_terms = ['introduction', 'overview', 'general', 'basic', 'miscellaneous']
        if not any(term in title.lower() for term in vague_terms):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_description_quality(self, description: str) -> float:
        """Assess the quality of a TOC entry description."""
        if not description:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(description.split())
        if 10 <= word_count <= 50:  # Good length
            score += 0.3
        elif word_count < 5 or word_count > 100:  # Too short or too long
            score -= 0.2
        
        # Information density
        if len(set(description.lower().split())) / len(description.split()) > 0.7:  # Low repetition
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_keyword_quality(self, keywords: List[str], disciplines: List[str]) -> float:
        """Assess the quality and relevance of extracted keywords."""
        if not keywords:
            return 0.0
        
        score = 0.5  # Base score
        
        # Quantity appropriateness
        if 3 <= len(keywords) <= 10:
            score += 0.2
        
        # Discipline relevance
        if disciplines:
            relevant_count = 0
            for discipline in disciplines:
                discipline_terms = self.discipline_validators.get(discipline, {}).get('key_terms', [])
                relevant_count += len(set(keywords).intersection(set(discipline_terms)))
            
            if relevant_count > 0:
                score += min(0.3, relevant_count * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _assess_completeness(self, entry: TOCEntry) -> float:
        """Assess the completeness of TOC entry metadata."""
        score = 0.0
        
        # Required fields
        if entry.title:
            score += 0.4
        if entry.book_source:
            score += 0.2
        
        # Optional but valuable fields
        if getattr(entry, "description", ""):
            score += 0.2
        if getattr(entry, 'keywords', []):
            score += 0.1
        if entry.level is not None:
            score += 0.1
        
        return score
    
    def _assess_authority_alignment(self, entry: TOCEntry, disciplines: List[str]) -> float:
        """Assess alignment with authority terminology standards."""
        if not disciplines:
            return 0.5  # Neutral score if no disciplines identified
        
        text = f"{entry.title} {getattr(entry, "description", "") or ''}".lower()
        total_score = 0.0
        
        for discipline in disciplines:
            discipline_score = 0.5  # Base score
            
            terminology_standards = self.authority_terminology.get(discipline, {})
            
            # Check for preferred terminology
            preferred_terms = terminology_standards.get('preferred', [])
            for term in preferred_terms:
                if term.lower() in text:
                    discipline_score += 0.1
            
            # Check for deprecated terminology
            deprecated_terms = terminology_standards.get('deprecated', [])
            for term in deprecated_terms:
                if term.lower() in text:
                    discipline_score -= 0.2
            
            total_score += max(0.0, min(1.0, discipline_score))
        
        return total_score / len(disciplines) if disciplines else 0.5
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """Determine quality level from score."""
        if quality_score >= 0.8:
            return 'High'
        elif quality_score >= 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _improve_title_formatting(self, title: str) -> str:
        """Improve title formatting."""
        # Capitalize properly
        improved = title.title()
        
        # Fix common formatting issues
        improved = re.sub(r'\\s+', ' ', improved)  # Multiple spaces
        improved = improved.strip()
        
        return improved
    
    def _infer_description(self, entry: TOCEntry) -> Optional[str]:
        """Infer description from title and context."""
        title_words = entry.title.lower().split()
        
        # Simple description generation based on title keywords
        if any(word in title_words for word in ['introduction', 'overview']):
            return f"An introductory overview of {entry.title.lower()}"
        elif any(word in title_words for word in ['advanced', 'specialized']):
            return f"Advanced concepts and applications in {entry.title.lower()}"
        else:
            return f"Study of {entry.title.lower()}"
    
    def _improve_keyword_extraction(self, entry: TOCEntry) -> List[str]:
        """Improve keyword extraction with enhanced algorithms."""
        text = f"{entry.title} {getattr(entry, "description", "") or ''}"
        
        # Use more sophisticated keyword extraction
        words = re.findall(r'\\b[a-zA-Z]{3,}\\b', text.lower())
        
        # Remove stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        for keyword in keywords:
            if keyword not in unique_keywords:
                unique_keywords.append(keyword)
        
        return unique_keywords[:10]
    
    def _apply_terminology_corrections(self, text: str, discipline: str) -> str:
        """Apply terminology corrections for a specific discipline."""
        corrections = self.authority_terminology.get(discipline, {}).get('corrections', {})
        
        corrected_text = text
        for incorrect, correct in corrections.items():
            corrected_text = re.sub(r'\\b' + re.escape(incorrect) + r'\\b', correct, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """Load quality assessment thresholds."""
        return {
            'minimum_title_quality': 0.5,
            'minimum_overall_quality': 0.6,
            'minimum_authority_alignment': 0.7,
            'maximum_violation_tolerance': 0
        }
    
    def _load_extraction_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for TOC extraction."""
        return {
            'chapter_patterns': [
                r'chapter\\s+(\\d+)',
                r'(\\d+)\\s*\\.',
                r'part\\s+(\\d+)',
                r'unit\\s+(\\d+)'
            ],
            'section_patterns': [
                r'section\\s+(\\d+)',
                r'(\\d+\\.\\d+)',
                r'lesson\\s+(\\d+)'
            ]
        }
    
    def _load_discipline_validators(self) -> Dict[str, Dict]:
        """Load discipline-specific validation information."""
        return {
            'Physics': {
                'key_terms': ['force', 'energy', 'momentum', 'field', 'wave', 'particle', 'quantum', 'relativity'],
                'complexity_indicators': {
                    'quantum': 0.3, 'relativity': 0.3, 'field theory': 0.4,
                    'mechanics': 0.1, 'thermodynamics': 0.1
                }
            },
            'Chemistry': {
                'key_terms': ['molecule', 'atom', 'bond', 'reaction', 'element', 'compound', 'organic', 'inorganic'],
                'complexity_indicators': {
                    'quantum chemistry': 0.4, 'spectroscopy': 0.2,
                    'general chemistry': -0.1, 'basic': -0.2
                }
            },
            'Biology': {
                'key_terms': ['cell', 'organism', 'gene', 'protein', 'evolution', 'ecology', 'physiology'],
                'complexity_indicators': {
                    'molecular biology': 0.2, 'biochemistry': 0.2,
                    'general biology': -0.1, 'introduction': -0.2
                }
            }
        }
    
    def _load_authority_terminology(self) -> Dict[str, Dict]:
        """Load authority-approved terminology standards."""
        return {
            'Physics': {
                'preferred': ['mass', 'acceleration', 'velocity', 'momentum'],
                'deprecated': ['weight'],  # when referring to mass
                'corrections': {
                    'weight': 'mass',  # In physics contexts
                    'centrifugal force': 'centripetal force'
                }
            },
            'Chemistry': {
                'preferred': ['molecule', 'molar mass', 'concentration'],
                'deprecated': ['molecular weight'],
                'corrections': {
                    'molecular weight': 'molar mass'
                }
            },
            'Biology': {
                'preferred': ['organism', 'species', 'population'],
                'deprecated': [],
                'corrections': {}
            }
        }