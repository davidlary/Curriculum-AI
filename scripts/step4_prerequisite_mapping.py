#!/usr/bin/env python3
"""
Step 4: Prerequisite Mapping Module
Uses TOC ordering and physics domain knowledge to create prerequisite relationships.

The TOCs provide the pedagogical ordering that expert educators established.
This module preserves that ordering and creates dependency relationships
based on the natural progression within and across books.
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import time
import networkx as nx
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

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
CACHE_DIR = BASE_DIR / "Cache" / "Prerequisites"
OUTPUT_DIR = BASE_DIR / "Curriculum"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class PrerequisiteRelation:
    """Represents a prerequisite relationship between topics."""
    topic_id: str
    topic_title: str
    prerequisite_ids: List[str]
    prerequisite_titles: List[str]
    educational_level: str
    source_book: str
    toc_order: int
    depth_level: int
    domain: str
    confidence_score: float

@dataclass
class PrerequisiteGraph:
    """Complete prerequisite graph for the curriculum."""
    relations: List[PrerequisiteRelation]
    dependency_graph: Dict[str, List[str]]
    topological_order: List[str]
    circular_dependencies: List[Tuple[str, str]]
    level_progressions: Dict[str, List[str]]

class TOCBasedPrerequisiteMapper:
    """
    Creates prerequisite relationships using TOC ordering and physics knowledge.
    
    The TOCs represent expert pedagogical ordering. This class:
    1. Preserves the sequential ordering within each book
    2. Uses physics domain knowledge to connect across books
    3. Creates fine-grained prerequisite relationships
    4. Maintains educational level progressions
    """
    
    def __init__(self):
        self.physics_domains = self._init_physics_domains()
        self.fundamental_prerequisites = self._init_fundamental_prerequisites()
        self.cross_level_mapping = self._init_cross_level_mapping()
        logger.info("TOCBasedPrerequisiteMapper initialized")

    def _init_physics_domains(self) -> Dict[str, List[str]]:
        """Initialize physics domain keywords for classification."""
        return {
            'mathematics_fundamentals': ['units', 'measurement', 'vectors', 'scalars', 'mathematics', 'trigonometry', 'calculus'],
            'mechanics': ['motion', 'velocity', 'acceleration', 'force', 'newton', 'dynamics', 'kinematics', 'momentum', 'energy', 'work'],
            'waves_oscillations': ['waves', 'oscillation', 'vibration', 'frequency', 'amplitude', 'pendulum', 'harmonic'],
            'thermodynamics': ['heat', 'temperature', 'thermal', 'entropy', 'thermodynamics', 'gas', 'pressure'],
            'electricity': ['electric', 'charge', 'current', 'voltage', 'resistance', 'circuit', 'coulomb'],
            'magnetism': ['magnetic', 'magnet', 'field', 'flux', 'induction', 'faraday'],
            'optics': ['light', 'optics', 'lens', 'mirror', 'reflection', 'refraction', 'interference', 'diffraction'],
            'modern_physics': ['relativity', 'quantum', 'atomic', 'nuclear', 'photon', 'electron', 'particle'],
            'astronomy': ['star', 'planet', 'galaxy', 'solar', 'universe', 'cosmic', 'celestial', 'astronomy']
        }

    def _init_fundamental_prerequisites(self) -> Dict[str, List[str]]:
        """Define fundamental prerequisite relationships in physics."""
        return {
            'mechanics': ['mathematics_fundamentals'],
            'waves_oscillations': ['mechanics'],
            'thermodynamics': ['mechanics'],
            'electricity': ['mathematics_fundamentals'],
            'magnetism': ['electricity'],
            'optics': ['waves_oscillations'],
            'modern_physics': ['mechanics', 'electricity', 'magnetism', 'optics'],
            'astronomy': ['mechanics', 'optics', 'modern_physics']
        }

    def _init_cross_level_mapping(self) -> Dict[str, Dict[str, str]]:
        """Map how topics progress across educational levels."""
        return {
            'high_school': {
                'next_level': 'undergraduate',
                'depth_multiplier': 1.0,
                'complexity_cap': 3
            },
            'undergraduate': {
                'next_level': 'graduate',
                'depth_multiplier': 2.0,
                'complexity_cap': 5
            },
            'graduate': {
                'next_level': None,
                'depth_multiplier': 3.0,
                'complexity_cap': 7
            }
        }

    def create_prerequisite_mapping(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """
        Create comprehensive prerequisite mapping using TOC ordering and physics knowledge.
        
        This preserves the pedagogical ordering from TOCs while adding prerequisite relationships.
        """
        start_time = time.time()
        logger.info(f"Creating prerequisite mapping for {discipline} in {language}")
        
        # Load normalized topics from Step 3
        topics_file = OUTPUT_DIR / f"{discipline}_{language}_topics_normalized.json"
        if not topics_file.exists():
            raise FileNotFoundError(f"Normalized topics file not found: {topics_file}")
        
        with open(topics_file, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        
        # Load original TOC data for ordering information
        tocs_file = OUTPUT_DIR / f"{discipline}_{language}_tocs_extracted.json"
        with open(tocs_file, 'r', encoding='utf-8') as f:
            tocs_data = json.load(f)
        
        normalized_topics = topics_data['normalized_topics']
        logger.info(f"Processing {len(normalized_topics)} normalized topics")
        
        # Create detailed prerequisite relationships
        prerequisite_relations = self._create_toc_based_prerequisites(
            normalized_topics, tocs_data, discipline
        )
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(prerequisite_relations)
        
        # Create topological ordering (pedagogical sequence)
        topological_order, circular_deps = self._create_topological_order(dependency_graph)
        
        # Create level-based progressions
        level_progressions = self._create_level_progressions(prerequisite_relations)
        
        # Expand to fine-grained subtopics
        expanded_subtopics = self._expand_to_fine_grained_subtopics(
            prerequisite_relations, tocs_data
        )
        
        # Create final prerequisite graph
        prereq_graph = PrerequisiteGraph(
            relations=prerequisite_relations,
            dependency_graph=dependency_graph,
            topological_order=topological_order,
            circular_dependencies=circular_deps,
            level_progressions=level_progressions
        )
        
        processing_time = time.time() - start_time
        
        result = {
            'discipline': discipline,
            'language': language,
            'prerequisite_timestamp': datetime.now().isoformat(),
            'prerequisite_relations': [asdict(rel) for rel in prerequisite_relations],
            'dependency_graph': dependency_graph,
            'topological_order': topological_order,
            'circular_dependencies': circular_deps,
            'level_progressions': level_progressions,
            'expanded_subtopics': expanded_subtopics,
            'metrics': {
                'total_topics': len(prerequisite_relations),
                'total_prerequisites': sum(len(rel.prerequisite_ids) for rel in prerequisite_relations),
                'circular_dependencies': len(circular_deps),
                'processing_time': processing_time,
                'fine_grained_subtopics': len(expanded_subtopics)
            }
        }
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_prerequisites_mapped.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Prerequisite mapping completed: {len(prerequisite_relations)} relations, {len(expanded_subtopics)} fine-grained subtopics")
        return result

    def _create_toc_based_prerequisites(self, normalized_topics: List[Dict], tocs_data: Dict, discipline: str) -> List[PrerequisiteRelation]:
        """Create prerequisite relationships based on TOC ordering and physics knowledge."""
        relations = []
        
        # Create mapping from normalized topics back to original TOC entries
        topic_to_toc_mapping = self._map_topics_to_toc_entries(normalized_topics, tocs_data)
        
        for i, topic in enumerate(normalized_topics):
            topic_id = topic['id']
            topic_title = topic['canonical_name']
            educational_levels = topic['educational_levels']
            
            # Classify the topic domain
            domain = self._classify_physics_domain(topic_title)
            
            # Find prerequisites based on TOC ordering
            toc_prerequisites = self._find_toc_based_prerequisites(
                topic, topic_to_toc_mapping, normalized_topics[:i]
            )
            
            # Add domain-based prerequisites
            domain_prerequisites = self._find_domain_based_prerequisites(
                domain, normalized_topics[:i]
            )
            
            # Combine and deduplicate prerequisites
            all_prereqs = list(set(toc_prerequisites + domain_prerequisites))
            
            # Create relation for each educational level
            for level in educational_levels:
                relation = PrerequisiteRelation(
                    topic_id=topic_id,
                    topic_title=topic_title,
                    prerequisite_ids=all_prereqs,
                    prerequisite_titles=[self._get_topic_title(pid, normalized_topics) for pid in all_prereqs],
                    educational_level=level,
                    source_book=', '.join(topic.get('source_books', [])),
                    toc_order=i,
                    depth_level=len(topic.get('depth_progression', {})),
                    domain=domain,
                    confidence_score=self._calculate_confidence_score(topic, all_prereqs)
                )
                relations.append(relation)
        
        return relations

    def _map_topics_to_toc_entries(self, normalized_topics: List[Dict], tocs_data: Dict) -> Dict[str, List[Dict]]:
        """Map normalized topics back to their original TOC entries for ordering information."""
        topic_mapping = {}
        
        # Get all TOC entries with their order
        all_toc_entries = []
        for level, level_tocs in tocs_data.get('tocs_by_level', {}).items():
            for book_toc in level_tocs:
                for i, entry in enumerate(book_toc['toc_entries']):
                    entry['book_id'] = book_toc['book_id']
                    entry['book_title'] = book_toc['book_title']
                    entry['educational_level'] = level
                    entry['order_in_book'] = i
                    all_toc_entries.append(entry)
        
        # Match normalized topics to TOC entries
        for topic in normalized_topics:
            topic_id = topic['id']
            topic_mapping[topic_id] = []
            
            # Find matching TOC entries based on alternative names
            for alt_name in topic['alternative_names']:
                for toc_entry in all_toc_entries:
                    if self._title_similarity(alt_name, toc_entry['title']) > 0.8:
                        topic_mapping[topic_id].append(toc_entry)
        
        return topic_mapping

    def _find_toc_based_prerequisites(self, topic: Dict, topic_mapping: Dict, previous_topics: List[Dict]) -> List[str]:
        """Find prerequisites based on TOC ordering within books."""
        prerequisites = []
        topic_id = topic['id']
        
        if topic_id not in topic_mapping:
            return prerequisites
        
        # For each TOC entry this topic maps to
        for toc_entry in topic_mapping[topic_id]:
            book_id = toc_entry['book_id']
            order_in_book = toc_entry['order_in_book']
            
            # Find topics from the same book that come before this one
            for prev_topic in previous_topics:
                prev_id = prev_topic['id']
                if prev_id in topic_mapping:
                    for prev_toc_entry in topic_mapping[prev_id]:
                        if (prev_toc_entry['book_id'] == book_id and 
                            prev_toc_entry['order_in_book'] < order_in_book):
                            # Add as prerequisite if it's a related domain
                            if self._is_related_domain(topic, prev_topic):
                                prerequisites.append(prev_id)
        
        return list(set(prerequisites))

    def _find_domain_based_prerequisites(self, domain: str, previous_topics: List[Dict]) -> List[str]:
        """Find prerequisites based on physics domain knowledge."""
        prerequisites = []
        
        if domain in self.fundamental_prerequisites:
            required_domains = self.fundamental_prerequisites[domain]
            
            for prev_topic in previous_topics:
                prev_domain = self._classify_physics_domain(prev_topic['canonical_name'])
                if prev_domain in required_domains:
                    prerequisites.append(prev_topic['id'])
        
        return prerequisites

    def _classify_physics_domain(self, title: str) -> str:
        """Classify a topic into a physics domain."""
        title_lower = title.lower()
        
        domain_scores = {}
        for domain, keywords in self.physics_domains.items():
            score = sum(1 for keyword in keywords if keyword in title_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'general'

    def _expand_to_fine_grained_subtopics(self, prerequisite_relations: List[PrerequisiteRelation], 
                                        tocs_data: Dict) -> List[Dict[str, Any]]:
        """
        Expand normalized topics into ~1000 fine-grained subtopics using original TOC entries.
        
        This uses the original 1,161 TOC entries as the fine-grained subtopics,
        maintaining their pedagogical ordering and adding prerequisite relationships.
        """
        fine_grained_subtopics = []
        subtopic_id = 0
        
        # Process each TOC entry as a fine-grained subtopic
        for level, level_tocs in tocs_data.get('tocs_by_level', {}).items():
            for book_toc in level_tocs:
                book_title = book_toc['book_title']
                
                for i, entry in enumerate(book_toc['toc_entries']):
                    subtopic_id += 1
                    
                    # Find the normalized topic this entry belongs to
                    parent_topic = self._find_parent_normalized_topic(
                        entry['title'], prerequisite_relations
                    )
                    
                    # Create fine-grained subtopic
                    subtopic = {
                        'id': f"subtopic_{subtopic_id:04d}",
                        'title': entry['title'],
                        'hierarchy_level': entry.get('level', 1),
                        'educational_level': level,
                        'source_book': book_title,
                        'order_in_book': i,
                        'parent_topic_id': parent_topic['topic_id'] if parent_topic else None,
                        'domain': self._classify_physics_domain(entry['title']),
                        'prerequisites': self._determine_subtopic_prerequisites(
                            entry, i, level_tocs, subtopic_id
                        ),
                        'learning_objectives': self._generate_learning_objectives(
                            entry['title'], level
                        ),
                        'estimated_duration_hours': self._estimate_duration(
                            entry['title'], entry.get('level', 1)
                        )
                    }
                    
                    fine_grained_subtopics.append(subtopic)
        
        logger.info(f"Expanded to {len(fine_grained_subtopics)} fine-grained subtopics")
        return fine_grained_subtopics

    def _determine_subtopic_prerequisites(self, entry: Dict, order_in_book: int, 
                                        book_tocs: List[Dict], subtopic_id: int) -> List[str]:
        """Determine prerequisites for a fine-grained subtopic based on TOC order."""
        prerequisites = []
        
        # Sequential prerequisites within the same book
        if order_in_book > 0:
            # Previous topic in same book is typically a prerequisite
            prev_subtopic_id = f"subtopic_{subtopic_id-1:04d}"
            prerequisites.append(prev_subtopic_id)
        
        # Add foundational prerequisites based on domain
        domain = self._classify_physics_domain(entry['title'])
        if domain != 'mathematics_fundamentals' and order_in_book < 3:
            # Early topics need mathematical foundations
            prerequisites.append("subtopic_0001")  # Assume first topic is math foundations
        
        return prerequisites

    def _generate_learning_objectives(self, title: str, level: str) -> List[str]:
        """Generate learning objectives based on title and educational level."""
        objectives = []
        
        # Base objective
        if level == 'high_school':
            objectives.append(f"Understand the basic concepts of {title.lower()}")
            objectives.append(f"Apply fundamental principles of {title.lower()} to simple problems")
        elif level == 'undergraduate':
            objectives.append(f"Analyze complex scenarios involving {title.lower()}")
            objectives.append(f"Derive and apply mathematical relationships in {title.lower()}")
        else:  # graduate
            objectives.append(f"Critically evaluate advanced theories in {title.lower()}")
            objectives.append(f"Conduct research and analysis in {title.lower()}")
        
        return objectives

    def _estimate_duration(self, title: str, hierarchy_level: int) -> float:
        """Estimate study duration in hours based on topic complexity."""
        base_hours = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5}
        return base_hours.get(hierarchy_level, 1.0)

    # Additional helper methods...
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def _is_related_domain(self, topic1: Dict, topic2: Dict) -> bool:
        """Check if two topics are in related domains."""
        domain1 = self._classify_physics_domain(topic1['canonical_name'])
        domain2 = self._classify_physics_domain(topic2['canonical_name'])
        
        # Same domain or prerequisite domain
        if domain1 == domain2:
            return True
        if domain2 in self.fundamental_prerequisites.get(domain1, []):
            return True
        return False

    def _get_topic_title(self, topic_id: str, topics: List[Dict]) -> str:
        """Get topic title by ID."""
        for topic in topics:
            if topic['id'] == topic_id:
                return topic['canonical_name']
        return "Unknown Topic"

    def _calculate_confidence_score(self, topic: Dict, prerequisites: List[str]) -> float:
        """Calculate confidence score for prerequisite relationships."""
        # Higher confidence for topics with more educational levels
        level_bonus = len(topic.get('educational_levels', [])) * 0.1
        
        # Higher confidence for topics with clear source books
        source_bonus = 0.2 if topic.get('source_books') else 0.0
        
        # Moderate confidence for reasonable number of prerequisites
        prereq_score = min(len(prerequisites) * 0.1, 0.3)
        
        return min(0.5 + level_bonus + source_bonus + prereq_score, 1.0)

    def _find_parent_normalized_topic(self, toc_title: str, 
                                    prerequisite_relations: List[PrerequisiteRelation]) -> Optional[Dict]:
        """Find which normalized topic a TOC entry belongs to."""
        for relation in prerequisite_relations:
            if toc_title.lower() in relation.topic_title.lower() or \
               any(toc_title.lower() in alt.lower() for alt in relation.prerequisite_titles):
                return {'topic_id': relation.topic_id, 'topic_title': relation.topic_title}
        return None

    def _build_dependency_graph(self, relations: List[PrerequisiteRelation]) -> Dict[str, List[str]]:
        """Build dependency graph from prerequisite relations."""
        graph = defaultdict(list)
        for relation in relations:
            for prereq_id in relation.prerequisite_ids:
                graph[prereq_id].append(relation.topic_id)
        return dict(graph)

    def _create_topological_order(self, dependency_graph: Dict[str, List[str]]) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Create topological ordering of topics."""
        try:
            graph = nx.DiGraph(dependency_graph)
            topological_order = list(nx.topological_sort(graph))
            circular_deps = []
        except nx.NetworkXError:
            # Handle circular dependencies
            graph = nx.DiGraph(dependency_graph)
            try:
                cycles = list(nx.simple_cycles(graph))
                circular_deps = [(cycle[0], cycle[1]) for cycle in cycles if len(cycle) >= 2]
                
                # Remove one edge from each cycle to break it
                for cycle in cycles:
                    if len(cycle) >= 2:
                        graph.remove_edge(cycle[0], cycle[1])
                
                topological_order = list(nx.topological_sort(graph))
            except:
                # Fallback: use original order
                topological_order = list(dependency_graph.keys())
                circular_deps = []
        
        return topological_order, circular_deps

    def _create_level_progressions(self, relations: List[PrerequisiteRelation]) -> Dict[str, List[str]]:
        """Create educational level progressions."""
        progressions = defaultdict(list)
        
        # Group by educational level
        by_level = defaultdict(list)
        for relation in relations:
            by_level[relation.educational_level].append(relation)
        
        # Sort each level by TOC order
        for level, level_relations in by_level.items():
            sorted_relations = sorted(level_relations, key=lambda r: r.toc_order)
            progressions[level] = [r.topic_id for r in sorted_relations]
        
        return dict(progressions)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create prerequisite mapping from TOC ordering")
    parser.add_argument("--discipline", required=True, help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        mapper = TOCBasedPrerequisiteMapper()
        result = mapper.create_prerequisite_mapping(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nPrerequisite Mapping Summary for {args.discipline} ({args.language}):")
        print(f"Topics processed: {result['metrics']['total_topics']}")
        print(f"Total prerequisites: {result['metrics']['total_prerequisites']}")
        print(f"Fine-grained subtopics: {result['metrics']['fine_grained_subtopics']}")
        print(f"Circular dependencies: {result['metrics']['circular_dependencies']}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_prerequisites_mapped.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during prerequisite mapping: {e}")
        exit(1)


if __name__ == "__main__":
    main()