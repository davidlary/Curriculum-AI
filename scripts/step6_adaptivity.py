#!/usr/bin/env python3
"""
Step 6: Adaptivity Module
Creates adaptive learning pathways and personalized curriculum recommendations.

This module takes the sequenced curriculum and adds adaptivity features:
1. Multiple learning pathways for different learning styles
2. Prerequisite-based adaptive recommendations
3. Difficulty progression analysis
4. Learning objective mapping
5. Assessment strategy suggestions
6. Personalized pacing recommendations
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory structure
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Curriculum"

@dataclass
class LearningPathway:
    """Represents an adaptive learning pathway."""
    pathway_id: str
    name: str
    description: str
    target_audience: str
    learning_style: str
    difficulty_progression: str
    estimated_duration_weeks: float
    prerequisite_intensity: str
    topic_sequence: List[str]
    checkpoints: List[Dict]
    adaptive_branching: Dict[str, List[str]]

@dataclass
class AdaptiveRecommendation:
    """Represents an adaptive recommendation for a learner."""
    topic_id: str
    recommendation_type: str  # 'prerequisite', 'remediation', 'acceleration', 'enrichment'
    confidence_score: float
    reasoning: str
    suggested_resources: List[str]
    estimated_time: float
    difficulty_adjustment: float

@dataclass
class AdaptiveCurriculum:
    """Complete adaptive curriculum with multiple pathways."""
    base_curriculum_topics: List[str]
    learning_pathways: List[LearningPathway]
    adaptive_recommendations: Dict[str, List[AdaptiveRecommendation]]
    assessment_strategies: Dict[str, Dict]
    personalization_rules: Dict[str, Any]
    quality_metrics: Dict[str, float]

class AdaptivitySystem:
    """
    Creates adaptive learning pathways and personalized recommendations
    based on the sequenced curriculum.
    """
    
    def __init__(self):
        self.learning_styles = [
            'visual', 'auditory', 'kinesthetic', 'reading_writing', 
            'logical', 'social', 'solitary'
        ]
        
        self.difficulty_progressions = [
            'gentle', 'standard', 'accelerated', 'mastery_based'
        ]
        
        self.pathway_types = [
            'comprehensive', 'core_only', 'exam_focused', 'research_oriented',
            'practical_applications', 'theoretical_deep_dive'
        ]
        
        logger.info("AdaptivitySystem initialized")

    def create_adaptive_curriculum(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """
        Create adaptive curriculum with multiple learning pathways and 
        personalized recommendations.
        """
        start_time = time.time()
        logger.info(f"Creating adaptive curriculum for {discipline} in {language}")
        
        # Load sequenced curriculum from Step 5
        sequence_file = OUTPUT_DIR / f"{discipline}_{language}_curriculum_sequenced.json"
        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequenced curriculum not found: {sequence_file}")
        
        with open(sequence_file, 'r', encoding='utf-8') as f:
            sequenced_data = json.load(f)
        
        curriculum_units = sequenced_data.get('curriculum_units', [])
        logger.info(f"Processing {len(curriculum_units)} curriculum units for adaptivity")
        
        # Create multiple learning pathways
        learning_pathways = self._create_learning_pathways(curriculum_units)
        
        # Generate adaptive recommendations
        adaptive_recommendations = self._generate_adaptive_recommendations(curriculum_units)
        
        # Create assessment strategies
        assessment_strategies = self._create_assessment_strategies(curriculum_units)
        
        # Define personalization rules
        personalization_rules = self._define_personalization_rules()
        
        # Calculate quality metrics
        quality_metrics = self._calculate_adaptivity_quality_metrics(
            learning_pathways, adaptive_recommendations
        )
        
        # Create adaptive curriculum
        adaptive_curriculum = AdaptiveCurriculum(
            base_curriculum_topics=[unit['unit_id'] for unit in curriculum_units],
            learning_pathways=learning_pathways,
            adaptive_recommendations=adaptive_recommendations,
            assessment_strategies=assessment_strategies,
            personalization_rules=personalization_rules,
            quality_metrics=quality_metrics
        )
        
        processing_time = time.time() - start_time
        
        result = {
            'discipline': discipline,
            'language': language,
            'adaptivity_timestamp': datetime.now().isoformat(),
            'base_curriculum_topics': adaptive_curriculum.base_curriculum_topics,
            'learning_pathways': [asdict(pathway) for pathway in learning_pathways],
            'adaptive_recommendations': {
                topic_id: [asdict(rec) for rec in recs] 
                for topic_id, recs in adaptive_recommendations.items()
            },
            'assessment_strategies': assessment_strategies,
            'personalization_rules': personalization_rules,
            'quality_metrics': quality_metrics,
            'metrics': {
                'total_pathways': len(learning_pathways),
                'total_recommendations': sum(len(recs) for recs in adaptive_recommendations.values()),
                'processing_time': processing_time,
                'pathway_diversity_score': self._calculate_pathway_diversity(learning_pathways),
                'adaptivity_coverage': len(adaptive_recommendations) / len(curriculum_units) if curriculum_units else 0
            }
        }
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_adaptive_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Adaptive curriculum completed: {len(learning_pathways)} pathways, {sum(len(recs) for recs in adaptive_recommendations.values())} recommendations")
        return result

    def _create_learning_pathways(self, curriculum_units: List[Dict]) -> List[LearningPathway]:
        """Create multiple learning pathways for different learning styles and goals."""
        pathways = []
        
        # Separate core and elective topics
        core_topics = [unit for unit in curriculum_units if unit.get('is_core', True)]
        elective_topics = [unit for unit in curriculum_units if not unit.get('is_core', True)]
        
        # Create pathway 1: Comprehensive (all topics)
        comprehensive_pathway = LearningPathway(
            pathway_id="comprehensive",
            name="Comprehensive Physics Mastery",
            description="Complete coverage of all physics topics from high school to advanced undergraduate level",
            target_audience="Students seeking comprehensive physics education",
            learning_style="balanced",
            difficulty_progression="standard",
            estimated_duration_weeks=52.0,
            prerequisite_intensity="high",
            topic_sequence=[unit['unit_id'] for unit in curriculum_units],
            checkpoints=self._create_checkpoints(curriculum_units, 10),
            adaptive_branching=self._create_adaptive_branching(curriculum_units)
        )
        pathways.append(comprehensive_pathway)
        
        # Create pathway 2: Core Only (essential topics)
        core_pathway = LearningPathway(
            pathway_id="core_essential",
            name="Essential Physics Core",
            description="Core physics concepts essential for foundational understanding",
            target_audience="Students needing strong physics foundation",
            learning_style="logical",
            difficulty_progression="gentle",
            estimated_duration_weeks=30.0,
            prerequisite_intensity="medium",
            topic_sequence=[unit['unit_id'] for unit in core_topics],
            checkpoints=self._create_checkpoints(core_topics, 6),
            adaptive_branching=self._create_adaptive_branching(core_topics)
        )
        pathways.append(core_pathway)
        
        # Create pathway 3: Exam Focused (high-yield topics)
        exam_topics = self._select_exam_focused_topics(curriculum_units)
        exam_pathway = LearningPathway(
            pathway_id="exam_focused",
            name="Exam Preparation Track",
            description="High-yield topics optimized for physics exams and assessments",
            target_audience="Students preparing for physics exams",
            learning_style="reading_writing",
            difficulty_progression="accelerated",
            estimated_duration_weeks=20.0,
            prerequisite_intensity="low",
            topic_sequence=[unit['unit_id'] for unit in exam_topics],
            checkpoints=self._create_checkpoints(exam_topics, 5),
            adaptive_branching=self._create_adaptive_branching(exam_topics)
        )
        pathways.append(exam_pathway)
        
        # Create pathway 4: Visual Learners
        visual_topics = self._optimize_for_visual_learning(curriculum_units)
        visual_pathway = LearningPathway(
            pathway_id="visual_optimized",
            name="Visual Physics Journey",
            description="Physics curriculum optimized for visual and kinesthetic learners",
            target_audience="Visual and kinesthetic learners",
            learning_style="visual",
            difficulty_progression="gentle",
            estimated_duration_weeks=40.0,
            prerequisite_intensity="medium",
            topic_sequence=[unit['unit_id'] for unit in visual_topics],
            checkpoints=self._create_checkpoints(visual_topics, 8),
            adaptive_branching=self._create_adaptive_branching(visual_topics)
        )
        pathways.append(visual_pathway)
        
        # Create pathway 5: Accelerated Track
        accelerated_topics = self._create_accelerated_sequence(curriculum_units)
        accelerated_pathway = LearningPathway(
            pathway_id="accelerated",
            name="Accelerated Physics Track",
            description="Fast-paced curriculum for advanced students",
            target_audience="Advanced students with strong math background",
            learning_style="logical",
            difficulty_progression="accelerated",
            estimated_duration_weeks=25.0,
            prerequisite_intensity="high",
            topic_sequence=[unit['unit_id'] for unit in accelerated_topics],
            checkpoints=self._create_checkpoints(accelerated_topics, 6),
            adaptive_branching=self._create_adaptive_branching(accelerated_topics)
        )
        pathways.append(accelerated_pathway)
        
        return pathways

    def _create_checkpoints(self, topics: List[Dict], num_checkpoints: int) -> List[Dict]:
        """Create assessment checkpoints throughout the pathway."""
        checkpoints = []
        if not topics:
            return checkpoints
            
        checkpoint_interval = max(1, len(topics) // num_checkpoints)
        
        for i in range(0, len(topics), checkpoint_interval):
            checkpoint_topics = topics[i:i+checkpoint_interval]
            checkpoint = {
                'checkpoint_id': f"checkpoint_{i//checkpoint_interval + 1}",
                'position': i,
                'topics_covered': [topic['unit_id'] for topic in checkpoint_topics],
                'assessment_type': 'mixed',
                'estimated_duration_hours': 2.0,
                'mastery_threshold': 0.8,
                'remediation_topics': self._identify_remediation_topics(checkpoint_topics)
            }
            checkpoints.append(checkpoint)
        
        return checkpoints

    def _create_adaptive_branching(self, topics: List[Dict]) -> Dict[str, List[str]]:
        """Create adaptive branching options based on performance."""
        branching = {}
        
        for i, topic in enumerate(topics):
            topic_id = topic['unit_id']
            branches = []
            
            # Add remediation branch (easier topics)
            if i > 0:
                branches.append(topics[max(0, i-2)]['unit_id'])  # Go back for review
            
            # Add acceleration branch (skip ahead)
            if i < len(topics) - 2:
                branches.append(topics[min(len(topics)-1, i+2)]['unit_id'])  # Skip ahead
            
            # Add enrichment branch (related electives)
            domain = topic.get('domain', 'general')
            related_topics = [t['unit_id'] for t in topics 
                            if t.get('domain') == domain and t['unit_id'] != topic_id]
            if related_topics:
                branches.extend(related_topics[:2])  # Add up to 2 related topics
            
            if branches:
                branching[topic_id] = list(set(branches))  # Remove duplicates
        
        return branching

    def _select_exam_focused_topics(self, curriculum_units: List[Dict]) -> List[Dict]:
        """Select high-yield topics for exam preparation."""
        # Prioritize topics that are:
        # 1. Core topics
        # 2. Frequently tested domains
        # 3. Have clear learning objectives
        
        high_yield_domains = [
            'kinematics', 'dynamics', 'energy', 'electricity', 'magnetism', 
            'waves', 'optics', 'thermodynamics'
        ]
        
        exam_topics = []
        for unit in curriculum_units:
            if (unit.get('is_core', True) or 
                unit.get('domain', '').lower() in high_yield_domains or
                len(unit.get('learning_objectives', [])) >= 3):
                exam_topics.append(unit)
        
        return exam_topics[:int(len(curriculum_units) * 0.6)]  # Top 60% for exam focus

    def _optimize_for_visual_learning(self, curriculum_units: List[Dict]) -> List[Dict]:
        """Optimize topic sequence for visual and kinesthetic learners."""
        # Prioritize topics that benefit from visualization
        visual_friendly_domains = [
            'kinematics', 'waves', 'optics', 'electricity', 'magnetism',
            'rotation', 'oscillations'
        ]
        
        visual_topics = []
        other_topics = []
        
        for unit in curriculum_units:
            if unit.get('domain', '').lower() in visual_friendly_domains:
                visual_topics.append(unit)
            else:
                other_topics.append(unit)
        
        # Interleave visual-friendly topics with others, prioritizing visual
        optimized_sequence = []
        v_idx = o_idx = 0
        
        while v_idx < len(visual_topics) or o_idx < len(other_topics):
            # Add 2 visual topics for every 1 other topic
            for _ in range(2):
                if v_idx < len(visual_topics):
                    optimized_sequence.append(visual_topics[v_idx])
                    v_idx += 1
            
            if o_idx < len(other_topics):
                optimized_sequence.append(other_topics[o_idx])
                o_idx += 1
        
        return optimized_sequence

    def _create_accelerated_sequence(self, curriculum_units: List[Dict]) -> List[Dict]:
        """Create accelerated sequence by removing some intermediate steps."""
        # Keep only essential topics and high-level concepts
        accelerated = []
        
        for unit in curriculum_units:
            # Include if:
            # 1. It's a core topic
            # 2. It has many prerequisites (central concept)
            # 3. It's high difficulty
            
            is_essential = (
                unit.get('is_core', True) or
                len(unit.get('prerequisites', [])) >= 2 or
                unit.get('difficulty', 1) >= 3
            )
            
            if is_essential:
                accelerated.append(unit)
        
        return accelerated

    def _generate_adaptive_recommendations(self, curriculum_units: List[Dict]) -> Dict[str, List[AdaptiveRecommendation]]:
        """Generate adaptive recommendations for each topic."""
        recommendations = defaultdict(list)
        
        for unit in curriculum_units:
            topic_id = unit['unit_id']
            
            # Prerequisite recommendations
            if unit.get('prerequisites'):
                prereq_rec = AdaptiveRecommendation(
                    topic_id=topic_id,
                    recommendation_type='prerequisite',
                    confidence_score=0.9,
                    reasoning=f"Strong prerequisites required for {unit.get('title', 'this topic')}",
                    suggested_resources=['prerequisite_review', 'concept_map'],
                    estimated_time=2.0,
                    difficulty_adjustment=-0.5
                )
                recommendations[topic_id].append(prereq_rec)
            
            # Remediation recommendations for difficult topics
            if unit.get('difficulty', 1) >= 4:
                remediation_rec = AdaptiveRecommendation(
                    topic_id=topic_id,
                    recommendation_type='remediation',
                    confidence_score=0.8,
                    reasoning="High difficulty topic may require additional support",
                    suggested_resources=['practice_problems', 'video_tutorials', 'study_group'],
                    estimated_time=3.0,
                    difficulty_adjustment=-1.0
                )
                recommendations[topic_id].append(remediation_rec)
            
            # Acceleration recommendations for foundational topics
            if len(unit.get('prerequisites', [])) == 0 and unit.get('is_core', True):
                acceleration_rec = AdaptiveRecommendation(
                    topic_id=topic_id,
                    recommendation_type='acceleration',
                    confidence_score=0.7,
                    reasoning="Foundational topic suitable for acceleration",
                    suggested_resources=['advanced_problems', 'research_applications'],
                    estimated_time=1.5,
                    difficulty_adjustment=0.5
                )
                recommendations[topic_id].append(acceleration_rec)
            
            # Enrichment recommendations for electives
            if not unit.get('is_core', True):
                enrichment_rec = AdaptiveRecommendation(
                    topic_id=topic_id,
                    recommendation_type='enrichment',
                    confidence_score=0.6,
                    reasoning="Elective topic provides enrichment opportunities",
                    suggested_resources=['real_world_applications', 'current_research'],
                    estimated_time=2.5,
                    difficulty_adjustment=0.0
                )
                recommendations[topic_id].append(enrichment_rec)
        
        return dict(recommendations)

    def _create_assessment_strategies(self, curriculum_units: List[Dict]) -> Dict[str, Dict]:
        """Create assessment strategies for different topic types."""
        strategies = {}
        
        # Group topics by domain
        topics_by_domain = defaultdict(list)
        for unit in curriculum_units:
            domain = unit.get('domain', 'general')
            topics_by_domain[domain].append(unit)
        
        for domain, domain_topics in topics_by_domain.items():
            strategies[domain] = {
                'formative_assessments': self._design_formative_assessments(domain),
                'summative_assessments': self._design_summative_assessments(domain),
                'adaptive_testing': self._design_adaptive_testing(domain),
                'performance_analytics': self._design_performance_analytics(domain)
            }
        
        return strategies

    def _design_formative_assessments(self, domain: str) -> Dict:
        """Design formative assessment strategies for a domain."""
        domain_strategies = {
            'kinematics': ['motion_diagrams', 'graph_interpretation', 'calculation_checks'],
            'electricity': ['circuit_analysis', 'concept_questions', 'simulation_exercises'],
            'waves': ['wave_visualization', 'interference_patterns', 'frequency_analysis'],
            'default': ['concept_questions', 'problem_solving', 'peer_discussion']
        }
        
        strategies = domain_strategies.get(domain, domain_strategies['default'])
        
        return {
            'methods': strategies,
            'frequency': 'after_each_lesson',
            'feedback_timing': 'immediate',
            'adaptive_difficulty': True
        }

    def _design_summative_assessments(self, domain: str) -> Dict:
        """Design summative assessment strategies for a domain."""
        return {
            'methods': ['comprehensive_exam', 'project_based', 'laboratory_practical'],
            'frequency': 'end_of_unit',
            'weight_distribution': {'exam': 0.6, 'project': 0.3, 'lab': 0.1},
            'mastery_threshold': 0.8
        }

    def _design_adaptive_testing(self, domain: str) -> Dict:
        """Design adaptive testing strategies."""
        return {
            'algorithm': 'item_response_theory',
            'starting_difficulty': 'medium',
            'termination_criteria': 'standard_error_threshold',
            'item_selection': 'maximum_information',
            'ability_estimation': 'maximum_likelihood'
        }

    def _design_performance_analytics(self, domain: str) -> Dict:
        """Design performance analytics for tracking progress."""
        return {
            'metrics': ['completion_rate', 'accuracy_score', 'time_efficiency', 'concept_mastery'],
            'visualization': ['progress_charts', 'concept_maps', 'difficulty_heatmaps'],
            'reporting_frequency': 'weekly',
            'intervention_triggers': ['low_accuracy', 'slow_progress', 'concept_gaps']
        }

    def _define_personalization_rules(self) -> Dict[str, Any]:
        """Define rules for personalizing the learning experience."""
        return {
            'learning_style_adaptations': {
                'visual': {
                    'prefer_diagrams': True,
                    'video_content_weight': 0.4,
                    'text_content_weight': 0.2,
                    'interactive_weight': 0.4
                },
                'auditory': {
                    'prefer_lectures': True,
                    'audio_content_weight': 0.5,
                    'discussion_weight': 0.3,
                    'reading_weight': 0.2
                },
                'kinesthetic': {
                    'prefer_labs': True,
                    'hands_on_weight': 0.5,
                    'simulation_weight': 0.3,
                    'theory_weight': 0.2
                }
            },
            'difficulty_adaptations': {
                'struggling': {
                    'prerequisite_emphasis': 0.8,
                    'practice_problems_multiplier': 2.0,
                    'concept_review_frequency': 'high'
                },
                'advanced': {
                    'acceleration_factor': 1.5,
                    'enrichment_content': True,
                    'challenge_problems': True
                }
            },
            'pacing_rules': {
                'self_paced': {
                    'time_flexibility': True,
                    'checkpoint_enforcement': 'soft'
                },
                'instructor_paced': {
                    'time_flexibility': False,
                    'checkpoint_enforcement': 'hard'
                }
            }
        }

    def _identify_remediation_topics(self, topics: List[Dict]) -> List[str]:
        """Identify topics that might need remediation."""
        remediation = []
        for topic in topics:
            if (topic.get('difficulty', 1) >= 3 or 
                len(topic.get('prerequisites', [])) >= 2):
                remediation.append(topic['unit_id'])
        return remediation

    def _calculate_pathway_diversity(self, pathways: List[LearningPathway]) -> float:
        """Calculate diversity score across learning pathways."""
        if len(pathways) < 2:
            return 0.0
        
        # Compare pathway overlap
        total_comparisons = 0
        overlap_sum = 0
        
        for i in range(len(pathways)):
            for j in range(i + 1, len(pathways)):
                path1_topics = set(pathways[i].topic_sequence)
                path2_topics = set(pathways[j].topic_sequence)
                
                overlap = len(path1_topics.intersection(path2_topics))
                total_topics = len(path1_topics.union(path2_topics))
                
                if total_topics > 0:
                    overlap_ratio = overlap / total_topics
                    overlap_sum += overlap_ratio
                    total_comparisons += 1
        
        # Diversity is inverse of average overlap
        avg_overlap = overlap_sum / total_comparisons if total_comparisons > 0 else 1.0
        return 1.0 - avg_overlap

    def _calculate_adaptivity_quality_metrics(self, pathways: List[LearningPathway], 
                                            recommendations: Dict) -> Dict[str, float]:
        """Calculate quality metrics for the adaptive curriculum."""
        
        # Pathway diversity
        pathway_diversity = self._calculate_pathway_diversity(pathways)
        
        # Recommendation coverage
        total_topics = len(set().union(*(pathway.topic_sequence for pathway in pathways)))
        covered_topics = len(recommendations)
        recommendation_coverage = covered_topics / total_topics if total_topics > 0 else 0
        
        # Learning style coverage
        learning_styles_covered = len(set(pathway.learning_style for pathway in pathways))
        style_coverage = learning_styles_covered / len(self.learning_styles)
        
        # Difficulty progression coverage
        progressions_covered = len(set(pathway.difficulty_progression for pathway in pathways))
        progression_coverage = progressions_covered / len(self.difficulty_progressions)
        
        return {
            'pathway_diversity': pathway_diversity,
            'recommendation_coverage': recommendation_coverage,
            'learning_style_coverage': style_coverage,
            'difficulty_progression_coverage': progression_coverage,
            'overall_adaptivity_quality': (
                pathway_diversity * 0.3 +
                recommendation_coverage * 0.3 +
                style_coverage * 0.2 +
                progression_coverage * 0.2
            )
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create adaptive curriculum with multiple learning pathways")
    parser.add_argument("--discipline", required=True, help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--openai-api-key", help="OpenAI API key (not used in this step but accepted for compatibility)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        system = AdaptivitySystem()
        result = system.create_adaptive_curriculum(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nAdaptive Curriculum Summary for {args.discipline} ({args.language}):")
        print(f"Learning pathways created: {result['metrics']['total_pathways']}")
        print(f"Adaptive recommendations: {result['metrics']['total_recommendations']}")
        print(f"Pathway diversity score: {result['metrics']['pathway_diversity_score']:.3f}")
        print(f"Adaptivity coverage: {result['metrics']['adaptivity_coverage']:.3f}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        
        print(f"\nQuality Metrics:")
        for metric, value in result['quality_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\nLearning Pathways:")
        for pathway in result['learning_pathways']:
            print(f"  • {pathway['name']} ({pathway['learning_style']} learners)")
            print(f"    Duration: {pathway['estimated_duration_weeks']} weeks")
            print(f"    Topics: {len(pathway['topic_sequence'])}")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_adaptive_curriculum.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during adaptive curriculum creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()