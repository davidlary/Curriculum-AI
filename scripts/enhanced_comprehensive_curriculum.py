#!/usr/bin/env python3
"""
Enhanced Comprehensive Curriculum System

This system creates a truly comprehensive curriculum that:
1. Merges topics across educational levels with proper normalization
2. Preserves TOC pedagogical ordering from expert educators
3. Implements enhanced prerequisite ordering with topological sorting
4. Automatically identifies and properly places electives
5. Removes duplicates while preserving topic diversity
6. Creates learning pathways with cognitive progression rules

The result is a superset curriculum covering high school to graduate level
with optimal educational sequencing.
"""

import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import networkx as nx
from collections import defaultdict, Counter
import argparse
import re
from difflib import SequenceMatcher

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
class ComprehensiveTopic:
    """Represents a topic in the comprehensive curriculum."""
    topic_id: str
    title: str
    normalized_title: str
    educational_levels: List[str]  # Can appear at multiple levels
    primary_level: str  # Main educational level
    chapter_context: str
    section_context: str
    domain: str
    difficulty_progression: Dict[str, float]  # difficulty by level
    prerequisites: List[str]
    learning_objectives: List[str]
    estimated_duration_hours: float
    is_core: bool
    is_elective: bool
    toc_order_by_book: Dict[str, int]  # Preserve TOC ordering from each book
    source_books: List[str]
    depth_level: int
    cross_references: List[str]
    pedagogical_notes: List[str]

@dataclass
class ComprehensiveCurriculum:
    """Complete comprehensive curriculum with optimal sequencing."""
    topics: List[ComprehensiveTopic]
    educational_progression: Dict[str, List[str]]  # level -> topic_ids
    domain_progressions: Dict[str, List[str]]  # domain -> topic_ids in order
    prerequisite_graph: Dict[str, List[str]]
    elective_topics: List[str]
    core_topics: List[str]
    learning_pathways: List[List[str]]  # Multiple valid pathways
    quality_metrics: Dict[str, float]
    total_duration_hours: float

class EnhancedComprehensiveCurriculumSystem:
    """
    Creates a comprehensive curriculum system that merges topics across
    educational levels while preserving pedagogical ordering and implementing
    advanced prerequisite sequencing.
    """
    
    def __init__(self):
        self.educational_levels = ['high_school', 'undergraduate', 'graduate']
        self.level_hierarchy = {level: i for i, level in enumerate(self.educational_levels)}
        
        # Physics domain hierarchy for proper sequencing
        self.domain_priority = {
            'units_measurement': 0,
            'vectors': 1,
            'kinematics': 2,
            'dynamics': 3,
            'energy': 4,
            'momentum': 5,
            'rotation': 6,
            'gravitation': 7,
            'oscillations': 8,
            'waves': 9,
            'thermodynamics': 10,
            'electricity': 11,
            'magnetism': 12,
            'circuits': 13,
            'optics': 14,
            'modern_physics': 15,
            'astrophysics': 16,
            'general': 17
        }
        
        logger.info("Enhanced Comprehensive Curriculum System initialized")

    def create_comprehensive_curriculum(self, discipline: str = "Physics", 
                                      language: str = "English") -> Dict[str, Any]:
        """
        Create the comprehensive curriculum by merging all educational levels
        with proper topic normalization and prerequisite ordering.
        """
        start_time = time.time()
        logger.info(f"🚀 Creating comprehensive curriculum for {discipline} in {language}")
        
        # Step 1: Load and merge all prerequisite data
        merged_topics = self._load_and_merge_topics(discipline, language)
        logger.info(f"📊 Merged topics from all levels: {len(merged_topics)} topics")
        
        # Step 2: Normalize topics across educational levels
        normalized_topics = self._normalize_topics_across_levels(merged_topics)
        logger.info(f"🔄 Normalized topics: {len(normalized_topics)} unique topics")
        
        # Step 3: Identify core vs elective topics
        core_topics, elective_topics = self._identify_core_and_electives(normalized_topics)
        logger.info(f"📚 Core topics: {len(core_topics)}, Electives: {len(elective_topics)}")
        
        # Step 4: Create comprehensive prerequisite graph
        prereq_graph = self._create_comprehensive_prerequisite_graph(normalized_topics)
        
        # Step 5: Apply enhanced topological ordering with TOC preservation
        ordered_topics = self._apply_enhanced_topological_ordering(
            normalized_topics, prereq_graph, core_topics, elective_topics
        )
        
        # Step 6: Create educational progressions
        educational_progression = self._create_educational_progressions(ordered_topics)
        
        # Step 7: Create domain progressions
        domain_progressions = self._create_domain_progressions(ordered_topics)
        
        # Step 8: Generate learning pathways
        learning_pathways = self._generate_learning_pathways(ordered_topics, prereq_graph)
        
        # Step 9: Calculate quality metrics
        quality_metrics = self._calculate_comprehensive_quality_metrics(
            ordered_topics, prereq_graph, educational_progression
        )
        
        # Create final curriculum
        curriculum = ComprehensiveCurriculum(
            topics=ordered_topics,
            educational_progression=educational_progression,
            domain_progressions=domain_progressions,
            prerequisite_graph=prereq_graph,
            elective_topics=[t.topic_id for t in elective_topics],
            core_topics=[t.topic_id for t in core_topics],
            learning_pathways=learning_pathways,
            quality_metrics=quality_metrics,
            total_duration_hours=sum(t.estimated_duration_hours for t in ordered_topics)
        )
        
        processing_time = time.time() - start_time
        
        # Create output
        result = {
            'discipline': discipline,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'comprehensive_topics': [asdict(topic) for topic in ordered_topics],
            'educational_progression': educational_progression,
            'domain_progressions': domain_progressions,
            'prerequisite_graph': prereq_graph,
            'elective_topics': curriculum.elective_topics,
            'core_topics': curriculum.core_topics,
            'learning_pathways': learning_pathways,
            'quality_metrics': quality_metrics,
            'total_duration_hours': curriculum.total_duration_hours,
            'statistics': {
                'total_topics': len(ordered_topics),
                'core_topics': len(core_topics),
                'elective_topics': len(elective_topics),
                'topics_by_level': {level: len([t for t in ordered_topics if t.primary_level == level]) 
                                  for level in self.educational_levels},
                'topics_by_domain': {domain: len([t for t in ordered_topics if t.domain == domain])
                                   for domain in set(t.domain for t in ordered_topics)},
                'processing_time': processing_time,
                'average_prerequisites': sum(len(t.prerequisites) for t in ordered_topics) / len(ordered_topics),
                'multilevel_topics': len([t for t in ordered_topics if len(t.educational_levels) > 1])
            }
        }
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_comprehensive_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Comprehensive curriculum completed: {len(ordered_topics)} topics, {curriculum.total_duration_hours:.1f} hours")
        logger.info(f"📁 Saved to: {output_file}")
        
        return result

    def _load_and_merge_topics(self, discipline: str, language: str) -> List[Dict]:
        """Load prerequisite data from all steps and merge."""
        merged_topics = []
        
        # Load from prerequisite mapping (Step 4)
        prereq_file = OUTPUT_DIR / f"{discipline}_{language}_prerequisites_mapped.json"
        if prereq_file.exists():
            with open(prereq_file, 'r', encoding='utf-8') as f:
                prereq_data = json.load(f)
                if 'expanded_subtopics' in prereq_data:
                    merged_topics.extend(prereq_data['expanded_subtopics'])
                elif 'prerequisite_relations' in prereq_data:
                    merged_topics.extend(prereq_data['prerequisite_relations'])
        
        # Load from TOC-aware curriculum if available
        toc_file = OUTPUT_DIR / f"{discipline}_toc_aware_curriculum.json"
        if toc_file.exists():
            with open(toc_file, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
                if 'subtopics' in toc_data:
                    # Convert TOC format to prerequisite format
                    for subtopic in toc_data['subtopics']:
                        merged_topic = {
                            'id': subtopic.get('id', f"toc_{subtopic.get('name', '').replace(' ', '_').lower()}"),
                            'title': subtopic.get('name', ''),
                            'educational_level': subtopic.get('educational_level', 'undergraduate'),
                            'domain': subtopic.get('domain', 'general'),
                            'prerequisites': subtopic.get('prerequisites', []),
                            'learning_objectives': subtopic.get('learning_objectives', []),
                            'estimated_duration_hours': subtopic.get('duration_hours', 1.0),
                            'chapter_title': subtopic.get('chapter_title', ''),
                            'section_title': subtopic.get('section_title', ''),
                            'source_book': ', '.join(subtopic.get('source_books', [])),
                            'toc_order': subtopic.get('pedagogical_order', 0),
                            'depth_level': subtopic.get('depth_level', 1),
                            'is_core': subtopic.get('is_core', True)
                        }
                        merged_topics.append(merged_topic)
        
        return merged_topics

    def _normalize_topics_across_levels(self, topics: List[Dict]) -> List[ComprehensiveTopic]:
        """Normalize topics across educational levels, merging duplicates."""
        # Group topics by normalized title
        normalized_groups = defaultdict(list)
        
        for topic in topics:
            normalized_title = self._normalize_title(topic.get('title', ''))
            normalized_groups[normalized_title].append(topic)
        
        comprehensive_topics = []
        
        for normalized_title, topic_group in normalized_groups.items():
            if not normalized_title.strip():
                continue
                
            # Merge topics with same normalized title
            merged_topic = self._merge_topic_group(topic_group, normalized_title)
            comprehensive_topics.append(merged_topic)
        
        return comprehensive_topics

    def _normalize_title(self, title: str) -> str:
        """Normalize topic title for cross-level matching."""
        normalized = title.lower().strip()
        
        # Remove common prefixes and suffixes
        normalized = re.sub(r'^(introduction to|intro to|basic|advanced|concepts of|principles of)\s+', '', normalized)
        normalized = re.sub(r'\s+(i|ii|iii|iv|v|1|2|3|4|5|part\s+\d+)$', '', normalized)
        normalized = re.sub(r'\s+(basics|fundamentals|overview|review)$', '', normalized)
        
        # Normalize common physics terms
        normalized = re.sub(r'electromagnetic?', 'electromagnetic', normalized)
        normalized = re.sub(r'thermodynamic?', 'thermodynamics', normalized)
        normalized = re.sub(r'mechanic?s?', 'mechanics', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _merge_topic_group(self, topic_group: List[Dict], normalized_title: str) -> ComprehensiveTopic:
        """Merge a group of topics with the same normalized title."""
        # Use the most detailed title
        best_title = max((topic.get('title', '') for topic in topic_group), key=len)
        
        # Collect all educational levels
        educational_levels = list(set(topic.get('educational_level', 'undergraduate') for topic in topic_group))
        
        # Determine primary level (highest level where it appears)
        level_hierarchy = {'high_school': 0, 'undergraduate': 1, 'graduate': 2}
        primary_level = max(educational_levels, key=lambda x: level_hierarchy.get(x, 1))
        
        # Merge prerequisites (union)
        all_prerequisites = set()
        for topic in topic_group:
            all_prerequisites.update(topic.get('prerequisites', []))
        
        # Merge learning objectives
        all_objectives = []
        for topic in topic_group:
            all_objectives.extend(topic.get('learning_objectives', []))
        all_objectives = list(set(all_objectives))  # Remove duplicates
        
        # Calculate estimated duration (average, with minimum of 1 hour)
        durations = [topic.get('estimated_duration_hours', 1.0) for topic in topic_group]
        avg_duration = max(sum(durations) / len(durations), 1.0)
        
        # Determine domain (most common)
        domains = [topic.get('domain', 'general') for topic in topic_group]
        domain = Counter(domains).most_common(1)[0][0]
        
        # Collect source books
        source_books = list(set(topic.get('source_book', '') for topic in topic_group if topic.get('source_book')))
        
        # Create TOC order mapping by book
        toc_order_by_book = {}
        for topic in topic_group:
            book = topic.get('source_book', 'unknown')
            if book and 'toc_order' in topic:
                toc_order_by_book[book] = topic['toc_order']
        
        # Determine if core or elective
        is_core_votes = [topic.get('is_core', True) for topic in topic_group]
        is_core = sum(is_core_votes) > len(is_core_votes) / 2  # Majority vote
        
        # Generate unique ID
        topic_id = f"comp_{normalized_title.replace(' ', '_').replace('-', '_')[:50]}"
        
        return ComprehensiveTopic(
            topic_id=topic_id,
            title=best_title,
            normalized_title=normalized_title,
            educational_levels=sorted(educational_levels, key=lambda x: level_hierarchy.get(x, 1)),
            primary_level=primary_level,
            chapter_context=topic_group[0].get('chapter_title', ''),
            section_context=topic_group[0].get('section_title', ''),
            domain=domain,
            difficulty_progression={level: level_hierarchy.get(level, 1) * 2 + 1 for level in educational_levels},
            prerequisites=list(all_prerequisites),
            learning_objectives=all_objectives,
            estimated_duration_hours=avg_duration,
            is_core=is_core,
            is_elective=not is_core,
            toc_order_by_book=toc_order_by_book,
            source_books=source_books,
            depth_level=max(topic.get('depth_level', 1) for topic in topic_group),
            cross_references=[],
            pedagogical_notes=[]
        )

    def _identify_core_and_electives(self, topics: List[ComprehensiveTopic]) -> Tuple[List[ComprehensiveTopic], List[ComprehensiveTopic]]:
        """Identify core vs elective topics based on various criteria."""
        core_topics = []
        elective_topics = []
        
        # Count how many times each topic appears across books
        topic_frequency = defaultdict(int)
        for topic in topics:
            topic_frequency[topic.normalized_title] = len(topic.source_books)
        
        for topic in topics:
            # Core criteria:
            # 1. Explicitly marked as core
            # 2. Appears in multiple books (frequency > 1)
            # 3. High-priority domains
            # 4. Has many dependent topics
            
            is_core_by_frequency = topic_frequency[topic.normalized_title] > 1
            is_core_by_domain = topic.domain in ['kinematics', 'dynamics', 'energy', 'electricity', 'magnetism', 'waves']
            is_core_by_level = topic.primary_level in ['high_school', 'undergraduate']
            
            if (topic.is_core and is_core_by_frequency) or is_core_by_domain or (is_core_by_level and is_core_by_frequency):
                core_topics.append(topic)
            else:
                elective_topics.append(topic)
        
        return core_topics, elective_topics

    def _create_comprehensive_prerequisite_graph(self, topics: List[ComprehensiveTopic]) -> Dict[str, List[str]]:
        """Create prerequisite graph with enhanced dependencies."""
        graph = defaultdict(list)
        topic_lookup = {topic.topic_id: topic for topic in topics}
        title_to_id = {topic.normalized_title: topic.topic_id for topic in topics}
        
        for topic in topics:
            for prereq in topic.prerequisites:
                # Try to find prerequisite by ID first, then by normalized title
                prereq_id = None
                if prereq in topic_lookup:
                    prereq_id = prereq
                else:
                    # Try to match by normalized title
                    normalized_prereq = self._normalize_title(prereq)
                    if normalized_prereq in title_to_id:
                        prereq_id = title_to_id[normalized_prereq]
                
                if prereq_id and prereq_id != topic.topic_id:
                    graph[prereq_id].append(topic.topic_id)
        
        # Add implicit domain-based prerequisites
        self._add_implicit_prerequisites(graph, topics)
        
        return dict(graph)

    def _add_implicit_prerequisites(self, graph: Dict[str, List[str]], topics: List[ComprehensiveTopic]):
        """Add implicit prerequisites based on domain knowledge."""
        topic_by_domain = defaultdict(list)
        for topic in topics:
            topic_by_domain[topic.domain].append(topic)
        
        # Add basic domain progressions
        domain_order = ['units_measurement', 'vectors', 'kinematics', 'dynamics', 'energy']
        
        for i in range(len(domain_order) - 1):
            current_domain = domain_order[i]
            next_domain = domain_order[i + 1]
            
            if current_domain in topic_by_domain and next_domain in topic_by_domain:
                # Add prerequisites from current domain to next domain
                current_topics = topic_by_domain[current_domain]
                next_topics = topic_by_domain[next_domain]
                
                if current_topics and next_topics:
                    # Add prerequisite from first topic of current to first topic of next
                    prereq_topic = min(current_topics, key=lambda t: min(t.toc_order_by_book.values()) if t.toc_order_by_book else 999)
                    dependent_topic = min(next_topics, key=lambda t: min(t.toc_order_by_book.values()) if t.toc_order_by_book else 999)
                    
                    if prereq_topic.topic_id not in graph:
                        graph[prereq_topic.topic_id] = []
                    if dependent_topic.topic_id not in graph[prereq_topic.topic_id]:
                        graph[prereq_topic.topic_id].append(dependent_topic.topic_id)

    def _apply_enhanced_topological_ordering(self, topics: List[ComprehensiveTopic], 
                                           prereq_graph: Dict[str, List[str]],
                                           core_topics: List[ComprehensiveTopic],
                                           elective_topics: List[ComprehensiveTopic]) -> List[ComprehensiveTopic]:
        """Apply enhanced topological ordering preserving TOC order and placing electives last."""
        logger.info("🔄 Applying enhanced topological ordering with TOC preservation")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        topic_lookup = {topic.topic_id: topic for topic in topics}
        
        # Add nodes
        for topic in topics:
            G.add_node(topic.topic_id, topic=topic)
        
        # Add edges
        for prereq_id, dependent_ids in prereq_graph.items():
            for dependent_id in dependent_ids:
                if prereq_id in topic_lookup and dependent_id in topic_lookup:
                    G.add_edge(prereq_id, dependent_id)
        
        # Break cycles if they exist
        if not nx.is_directed_acyclic_graph(G):
            cycles_broken = self._break_cycles_preserving_toc_order(G, topic_lookup)
            logger.info(f"🔧 Broke {cycles_broken} cycles while preserving TOC order")
        
        # Separate core and elective topics
        core_ids = {topic.topic_id for topic in core_topics}
        elective_ids = {topic.topic_id for topic in elective_topics}
        
        # Order core topics first
        core_subgraph = G.subgraph(core_ids)
        ordered_core = self._toc_aware_topological_sort(core_subgraph, core_topics)
        
        # Order elective topics last
        elective_subgraph = G.subgraph(elective_ids)
        ordered_electives = self._toc_aware_topological_sort(elective_subgraph, elective_topics)
        
        # Combine with electives at the end
        final_order = ordered_core + ordered_electives
        
        logger.info(f"✅ Enhanced ordering complete: {len(ordered_core)} core + {len(ordered_electives)} electives")
        return final_order

    def _break_cycles_preserving_toc_order(self, graph: nx.DiGraph, topic_lookup: Dict) -> int:
        """Break cycles while trying to preserve TOC order."""
        edges_removed = 0
        
        try:
            while not nx.is_directed_acyclic_graph(graph):
                # Find strongly connected components
                sccs = list(nx.strongly_connected_components(graph))
                
                for scc in sccs:
                    if len(scc) > 1:  # Has cycle
                        # Find the edge to remove that least disrupts TOC order
                        scc_list = list(scc)
                        
                        # Sort by average TOC order
                        def avg_toc_order(topic_id):
                            topic = topic_lookup.get(topic_id)
                            if topic and topic.toc_order_by_book:
                                return sum(topic.toc_order_by_book.values()) / len(topic.toc_order_by_book)
                            return 999
                        
                        scc_list.sort(key=avg_toc_order)
                        
                        # Remove edge from last to first in TOC order
                        if len(scc_list) >= 2 and graph.has_edge(scc_list[-1], scc_list[0]):
                            graph.remove_edge(scc_list[-1], scc_list[0])
                            edges_removed += 1
                            break
                
                # Safety check to prevent infinite loop
                if edges_removed > 100:
                    logger.warning("Too many cycles, stopping cycle breaking")
                    break
                    
        except Exception as e:
            logger.warning(f"Error in cycle breaking: {e}")
        
        return edges_removed

    def _toc_aware_topological_sort(self, graph: nx.DiGraph, topics: List[ComprehensiveTopic]) -> List[ComprehensiveTopic]:
        """Topological sort that preserves TOC order as much as possible."""
        topic_lookup = {topic.topic_id: topic for topic in topics}
        ordered_topics = []
        
        # Get topological ordering
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # If cycles still exist, use simple ordering
            logger.warning("Cycles still exist, using simple TOC ordering")
            topo_order = [topic.topic_id for topic in topics]
        
        # Sort by domain priority first, then TOC order
        def sort_key(topic_id):
            topic = topic_lookup.get(topic_id)
            if not topic:
                return (999, 999, 999)
            
            domain_priority = self.domain_priority.get(topic.domain, 999)
            avg_toc_order = sum(topic.toc_order_by_book.values()) / len(topic.toc_order_by_book) if topic.toc_order_by_book else 999
            level_priority = self.level_hierarchy.get(topic.primary_level, 999)
            
            return (level_priority, domain_priority, avg_toc_order)
        
        # Sort the topologically ordered topics
        topo_order.sort(key=sort_key)
        
        # Convert back to topic objects
        for topic_id in topo_order:
            if topic_id in topic_lookup:
                ordered_topics.append(topic_lookup[topic_id])
        
        return ordered_topics

    def _create_educational_progressions(self, topics: List[ComprehensiveTopic]) -> Dict[str, List[str]]:
        """Create educational level progressions."""
        progressions = defaultdict(list)
        
        for topic in topics:
            progressions[topic.primary_level].append(topic.topic_id)
        
        return dict(progressions)

    def _create_domain_progressions(self, topics: List[ComprehensiveTopic]) -> Dict[str, List[str]]:
        """Create domain-based progressions."""
        progressions = defaultdict(list)
        
        for topic in topics:
            progressions[topic.domain].append(topic.topic_id)
        
        return dict(progressions)

    def _generate_learning_pathways(self, topics: List[ComprehensiveTopic], 
                                  prereq_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Generate multiple valid learning pathways."""
        # For now, create a single optimal pathway
        # This could be expanded to create multiple pathways for different learning styles
        pathway = [topic.topic_id for topic in topics]
        return [pathway]

    def _calculate_comprehensive_quality_metrics(self, topics: List[ComprehensiveTopic],
                                               prereq_graph: Dict[str, List[str]],
                                               educational_progression: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        total_topics = len(topics)
        
        # Topic coverage across levels
        level_coverage = len(set(topic.primary_level for topic in topics)) / len(self.educational_levels)
        
        # Domain coverage
        domain_coverage = len(set(topic.domain for topic in topics)) / len(self.domain_priority)
        
        # Prerequisite connectivity
        topics_with_prereqs = sum(1 for topic in topics if topic.prerequisites)
        prerequisite_connectivity = topics_with_prereqs / total_topics if total_topics > 0 else 0
        
        # Educational progression smoothness
        progression_smoothness = 1.0  # Simplified for now
        
        # TOC preservation score
        toc_preservation = self._calculate_toc_preservation_score(topics)
        
        # Elective placement score (electives should be towards the end)
        elective_placement = self._calculate_elective_placement_score(topics)
        
        return {
            'level_coverage': level_coverage,
            'domain_coverage': domain_coverage,
            'prerequisite_connectivity': prerequisite_connectivity,
            'educational_progression_smoothness': progression_smoothness,
            'toc_order_preservation': toc_preservation,
            'elective_placement_score': elective_placement,
            'overall_quality': (
                level_coverage * 0.2 +
                domain_coverage * 0.2 +
                prerequisite_connectivity * 0.2 +
                progression_smoothness * 0.15 +
                toc_preservation * 0.15 +
                elective_placement * 0.1
            )
        }

    def _calculate_toc_preservation_score(self, topics: List[ComprehensiveTopic]) -> float:
        """Calculate how well TOC order is preserved."""
        book_scores = []
        
        # Group topics by source book
        topics_by_book = defaultdict(list)
        for topic in topics:
            for book in topic.source_books:
                if book in topic.toc_order_by_book:
                    topics_by_book[book].append((topic, topic.toc_order_by_book[book]))
        
        for book, book_topics in topics_by_book.items():
            if len(book_topics) < 2:
                continue
            
            # Sort by current order and check against TOC order
            book_topics.sort(key=lambda x: topics.index(x[0]))
            toc_orders = [toc_order for _, toc_order in book_topics]
            
            # Calculate how much the order is preserved
            inversions = 0
            total_pairs = 0
            for i in range(len(toc_orders)):
                for j in range(i + 1, len(toc_orders)):
                    total_pairs += 1
                    if toc_orders[i] > toc_orders[j]:
                        inversions += 1
            
            if total_pairs > 0:
                book_scores.append(1.0 - (inversions / total_pairs))
        
        return sum(book_scores) / len(book_scores) if book_scores else 1.0

    def _calculate_elective_placement_score(self, topics: List[ComprehensiveTopic]) -> float:
        """Calculate how well electives are placed towards the end."""
        if not topics:
            return 1.0
        
        elective_positions = []
        for i, topic in enumerate(topics):
            if topic.is_elective:
                elective_positions.append(i / len(topics))  # Normalize position
        
        if not elective_positions:
            return 1.0
        
        # Good score if electives are in the latter half
        avg_elective_position = sum(elective_positions) / len(elective_positions)
        return max(0.0, min(1.0, (avg_elective_position - 0.5) * 2))


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create enhanced comprehensive curriculum")
    parser.add_argument("--discipline", default="Physics", help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        system = EnhancedComprehensiveCurriculumSystem()
        result = system.create_comprehensive_curriculum(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\n" + "="*80)
        print(f"🎓 ENHANCED COMPREHENSIVE CURRICULUM COMPLETED")
        print(f"="*80)
        print(f"📊 Total Topics: {result['statistics']['total_topics']:,}")
        print(f"📚 Core Topics: {result['statistics']['core_topics']:,}")
        print(f"🎯 Elective Topics: {result['statistics']['elective_topics']:,}")
        print(f"⏱️  Total Duration: {result['total_duration_hours']:.1f} hours")
        print(f"🌟 Overall Quality: {result['quality_metrics']['overall_quality']:.3f}")
        
        print(f"\n📈 Topics by Educational Level:")
        for level, count in result['statistics']['topics_by_level'].items():
            print(f"   {level.replace('_', ' ').title()}: {count:,}")
        
        print(f"\n🔬 Topics by Domain:")
        for domain, count in sorted(result['statistics']['topics_by_domain'].items()):
            print(f"   {domain.replace('_', ' ').title()}: {count:,}")
        
        print(f"\n📊 Quality Metrics:")
        for metric, value in result['quality_metrics'].items():
            print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
        
        print(f"\n✅ Processing completed in {result['statistics']['processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Error in comprehensive curriculum creation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()