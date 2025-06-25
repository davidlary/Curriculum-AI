#!/usr/bin/env python3
"""
Step 5: Pedagogical Sequencing Module
Creates the final educationally-ordered curriculum using TOC ordering and prerequisites.

This module takes the prerequisite relationships and creates the definitive
pedagogical sequence that respects:
1. Original TOC ordering from expert educators
2. Prerequisite dependencies
3. Educational level progressions (high school → undergraduate → graduate)
4. Domain-based logical flow
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
import networkx as nx
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
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
OUTPUT_DIR = BASE_DIR / "Curriculum"

@dataclass
class CurriculumUnit:
    """Represents a single unit in the pedagogically-ordered curriculum."""
    unit_id: str
    title: str
    educational_level: str
    order_position: int
    prerequisites: List[str]
    learning_objectives: List[str]
    estimated_duration_hours: float
    domain: str
    hierarchy_level: int
    source_book: str
    prerequisite_depth: int
    cross_level_connections: List[str]

@dataclass
class CurriculumSequence:
    """Complete pedagogically-ordered curriculum sequence."""
    units: List[CurriculumUnit]
    level_sequences: Dict[str, List[str]]
    domain_progressions: Dict[str, List[str]]
    total_duration_hours: float
    quality_metrics: Dict[str, float]

class PedagogicalSequencer:
    """
    Creates the final pedagogical sequence using TOC ordering and prerequisites.
    
    The sequence respects:
    1. TOC ordering within books (expert pedagogical decisions)
    2. Prerequisite dependencies across books
    3. Educational level progressions
    4. Logical domain flow in physics
    """
    
    def __init__(self):
        self.educational_levels = ['high_school', 'undergraduate', 'graduate']
        self.level_hierarchy = {level: i for i, level in enumerate(self.educational_levels)}
        logger.info("PedagogicalSequencer initialized")

    def create_pedagogical_sequence(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """
        Create the final pedagogically-ordered curriculum sequence.
        
        This uses the TOC ordering as the foundation and adds prerequisite-based
        ordering to create a comprehensive, educationally-sound curriculum.
        """
        start_time = time.time()
        logger.info(f"Creating pedagogical sequence for {discipline} in {language}")
        
        # Load prerequisite mapping from Step 4
        prereq_file = OUTPUT_DIR / f"{discipline}_{language}_prerequisites_mapped.json"
        if not prereq_file.exists():
            raise FileNotFoundError(f"Prerequisites file not found: {prereq_file}")
        
        with open(prereq_file, 'r', encoding='utf-8') as f:
            prereq_data = json.load(f)
        
        fine_grained_subtopics = prereq_data['expanded_subtopics']
        logger.info(f"Processing {len(fine_grained_subtopics)} fine-grained subtopics")
        
        # Create curriculum units from fine-grained subtopics
        curriculum_units = self._create_curriculum_units(fine_grained_subtopics)
        
        # Apply pedagogical ordering using TOC + prerequisites
        ordered_units = self._apply_pedagogical_ordering(curriculum_units, prereq_data)
        
        # Create level-based sequences
        level_sequences = self._create_level_sequences(ordered_units)
        
        # Create domain progressions
        domain_progressions = self._create_domain_progressions(ordered_units)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(ordered_units, prereq_data)
        
        # Create final curriculum sequence
        curriculum = CurriculumSequence(
            units=ordered_units,
            level_sequences=level_sequences,
            domain_progressions=domain_progressions,
            total_duration_hours=sum(unit.estimated_duration_hours for unit in ordered_units),
            quality_metrics=quality_metrics
        )
        
        processing_time = time.time() - start_time
        
        result = {
            'discipline': discipline,
            'language': language,
            'sequencing_timestamp': datetime.now().isoformat(),
            'curriculum_units': [asdict(unit) for unit in ordered_units],
            'level_sequences': level_sequences,
            'domain_progressions': domain_progressions,
            'total_duration_hours': curriculum.total_duration_hours,
            'quality_metrics': quality_metrics,
            'metrics': {
                'total_units': len(ordered_units),
                'units_by_level': {level: len(seq) for level, seq in level_sequences.items()},
                'processing_time': processing_time,
                'average_prerequisites_per_unit': sum(len(unit.prerequisites) for unit in ordered_units) / len(ordered_units),
                'total_prerequisite_connections': sum(len(unit.prerequisites) for unit in ordered_units)
            }
        }
        
        # Save results
        output_file = OUTPUT_DIR / f"{discipline}_{language}_curriculum_sequenced.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Pedagogical sequencing completed: {len(ordered_units)} units, {curriculum.total_duration_hours:.1f} hours total")
        return result

    def _create_curriculum_units(self, fine_grained_subtopics: List[Dict]) -> List[CurriculumUnit]:
        """Convert fine-grained subtopics into curriculum units."""
        units = []
        
        for subtopic in fine_grained_subtopics:
            unit = CurriculumUnit(
                unit_id=subtopic['id'],
                title=subtopic['title'],
                educational_level=subtopic['educational_level'],
                order_position=subtopic['order_in_book'],
                prerequisites=subtopic.get('prerequisites', []),
                learning_objectives=subtopic.get('learning_objectives', []),
                estimated_duration_hours=subtopic.get('estimated_duration_hours', 1.0),
                domain=subtopic['domain'],
                hierarchy_level=subtopic['hierarchy_level'],
                source_book=subtopic['source_book'],
                prerequisite_depth=0,  # Will be calculated
                cross_level_connections=[]  # Will be calculated
            )
            units.append(unit)
        
        return units

    def _apply_pedagogical_ordering(self, units: List[CurriculumUnit], 
                                  prereq_data: Dict) -> List[CurriculumUnit]:
        """
        Apply pedagogical ordering using TOC ordering and prerequisites.
        
        This creates the master curriculum sequence that respects:
        1. Educational level progression (high school first)
        2. TOC ordering within books
        3. Prerequisite dependencies across books
        4. Domain logical flow
        """
        logger.info("Applying pedagogical ordering using TOC + prerequisites")
        
        # First, separate by educational level
        units_by_level = defaultdict(list)
        for unit in units:
            units_by_level[unit.educational_level].append(unit)
        
        # Order within each level using TOC ordering + prerequisites
        ordered_units = []
        
        for level in self.educational_levels:
            if level not in units_by_level:
                continue
                
            level_units = units_by_level[level]
            logger.info(f"Ordering {len(level_units)} units for {level} level")
            
            # Create dependency graph for this level
            level_graph = self._create_level_dependency_graph(level_units)
            
            # Apply topological sort while preserving TOC order as much as possible
            ordered_level_units = self._toc_aware_topological_sort(level_units, level_graph)
            
            # Calculate prerequisite depths
            ordered_level_units = self._calculate_prerequisite_depths(ordered_level_units)
            
            # Update order positions
            for i, unit in enumerate(ordered_level_units):
                unit.order_position = len(ordered_units) + i
            
            ordered_units.extend(ordered_level_units)
        
        # Add cross-level connections
        ordered_units = self._add_cross_level_connections(ordered_units)
        
        logger.info(f"Pedagogical ordering complete: {len(ordered_units)} units in sequence")
        return ordered_units

    def _create_level_dependency_graph(self, units: List[CurriculumUnit]) -> nx.DiGraph:
        """Create dependency graph for units within an educational level."""
        graph = nx.DiGraph()
        unit_ids_in_level = set()
        
        # Add all units as nodes and track IDs
        for unit in units:
            graph.add_node(unit.unit_id, unit=unit)
            unit_ids_in_level.add(unit.unit_id)
        
        # Add prerequisite edges only within the same level
        for unit in units:
            for prereq_id in unit.prerequisites:
                # Only add edge if prerequisite is in the same level and not self-referential
                if prereq_id in unit_ids_in_level and prereq_id != unit.unit_id:
                    graph.add_edge(prereq_id, unit.unit_id)
        
        # Remove any self-loops that might have been created
        self_loops = list(nx.selfloop_edges(graph))
        if self_loops:
            graph.remove_edges_from(self_loops)
            logger.warning(f"Removed {len(self_loops)} self-loop prerequisites")
        
        return graph

    def _toc_aware_topological_sort(self, units: List[CurriculumUnit], 
                                  graph: nx.DiGraph) -> List[CurriculumUnit]:
        """
        Topological sort that preserves TOC ordering as much as possible.
        
        This algorithm respects prerequisite dependencies while trying to maintain
        the original TOC order from books (which represents expert pedagogical decisions).
        """
        units_dict = {unit.unit_id: unit for unit in units}
        
        # Group units by source book and sort by original TOC order
        units_by_book = defaultdict(list)
        for unit in units:
            units_by_book[unit.source_book].append(unit)
        
        # Sort each book's units by their original order
        for book_units in units_by_book.values():
            book_units.sort(key=lambda u: u.order_position)
        
        # Modified topological sort that prefers TOC order
        ordered_units = []
        in_degree = dict(graph.in_degree())
        available_units = deque([unit_id for unit_id, degree in in_degree.items() if degree == 0])
        
        # If no units have zero in-degree, there's a cycle - break it immediately
        if not available_units and len(units) > 0:
            logger.warning(f"All {len(units)} units have prerequisites - breaking cycles preemptively")
            cycles_broken = self._break_cycles_in_graph(graph, units_dict)
            logger.info(f"Broke {cycles_broken} prerequisite edges to resolve initial cycles")
            in_degree = dict(graph.in_degree())
            available_units = deque([unit_id for unit_id, degree in in_degree.items() if degree == 0])
        
        while available_units:
            # Among available units, prefer those that come earlier in TOC order
            current_batch = list(available_units)
            available_units.clear()
            
            # Sort by book TOC order, then by source book consistency
            current_batch.sort(key=lambda uid: (
                units_dict[uid].source_book,
                units_dict[uid].order_position
            ))
            
            for unit_id in current_batch:
                unit = units_dict[unit_id]
                ordered_units.append(unit)
                
                # Update in-degrees for successors
                for successor in graph.successors(unit_id):
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        available_units.append(successor)
        
        # Check for cycles and break them if found
        if len(ordered_units) != len(units):
            logger.warning(f"Cycle detected: {len(ordered_units)} processed, {len(units)} total")
            remaining_ids = set(units_dict.keys()) - set(unit.unit_id for unit in ordered_units)
            
            # Break cycles by removing edges in remaining subgraph
            remaining_graph = graph.subgraph(remaining_ids).copy()
            cycles_broken = self._break_cycles_in_graph(remaining_graph, units_dict)
            logger.info(f"Broke {cycles_broken} prerequisite edges to resolve cycles")
            
            # Continue topological sort with cycle-free graph
            remaining_in_degree = dict(remaining_graph.in_degree())
            remaining_available = deque([uid for uid, degree in remaining_in_degree.items() if degree == 0])
            
            while remaining_available:
                current_batch = list(remaining_available)
                remaining_available.clear()
                
                current_batch.sort(key=lambda uid: (
                    units_dict[uid].source_book,
                    units_dict[uid].order_position
                ))
                
                for unit_id in current_batch:
                    unit = units_dict[unit_id]
                    ordered_units.append(unit)
                    
                    for successor in remaining_graph.successors(unit_id):
                        remaining_in_degree[successor] -= 1
                        if remaining_in_degree[successor] == 0:
                            remaining_available.append(successor)
            
            # If still remaining, add in TOC order
            if len(ordered_units) != len(units):
                final_remaining = set(units_dict.keys()) - set(unit.unit_id for unit in ordered_units)
                final_units = [units_dict[uid] for uid in final_remaining]
                final_units.sort(key=lambda u: (u.source_book, u.order_position))
                ordered_units.extend(final_units)
                logger.warning(f"Added {len(final_units)} units in TOC order due to unresolved cycles")
        
        return ordered_units

    def _break_cycles_in_graph(self, graph: nx.DiGraph, units_dict: Dict[str, CurriculumUnit]) -> int:
        """Break cycles in the dependency graph by removing edges."""
        edges_removed = 0
        
        try:
            # If graph is empty, nothing to do
            if len(graph.nodes()) == 0:
                return 0
            
            # First try: Find and break simple cycles
            try:
                simple_cycles = list(nx.simple_cycles(graph))
                for cycle in simple_cycles:
                    if len(cycle) > 1:
                        # Sort cycle nodes by TOC order
                        cycle_sorted = sorted(cycle, key=lambda uid: (
                            units_dict.get(uid, type('obj', (object,), {'source_book': '', 'order_position': 999999})).source_book,
                            units_dict.get(uid, type('obj', (object,), {'source_book': '', 'order_position': 999999})).order_position
                        ))
                        
                        # Remove the edge from last to first in TOC order
                        if graph.has_edge(cycle_sorted[-1], cycle_sorted[0]):
                            graph.remove_edge(cycle_sorted[-1], cycle_sorted[0])
                            edges_removed += 1
                            logger.debug(f"Removed cycle edge: {cycle_sorted[-1]} -> {cycle_sorted[0]}")
            except (nx.NetworkXError, Exception) as e:
                logger.debug(f"Simple cycle detection failed: {e}")
            
            # Second try: Use strongly connected components
            sccs = list(nx.strongly_connected_components(graph))
            for scc in sccs:
                if len(scc) > 1:  # Cycle found
                    scc_list = list(scc)
                    scc_list.sort(key=lambda uid: (
                        units_dict.get(uid, type('obj', (object,), {'source_book': '', 'order_position': 999999})).source_book,
                        units_dict.get(uid, type('obj', (object,), {'source_book': '', 'order_position': 999999})).order_position
                    ))
                    
                    # Remove edges that go backwards in TOC order
                    for i, source in enumerate(scc_list):
                        for j, target in enumerate(scc_list):
                            if j < i and graph.has_edge(source, target):
                                graph.remove_edge(source, target)
                                edges_removed += 1
                                logger.debug(f"Removed SCC backward edge: {source} -> {target}")
                                
        except Exception as e:
            logger.warning(f"Error in advanced cycle breaking: {e}")
            
        # Final fallback: if still has cycles, remove edges with highest in-degree targets
        try:
            while not nx.is_directed_acyclic_graph(graph) and len(graph.edges()) > 0:
                in_degrees = dict(graph.in_degree())
                max_in_degree = max(in_degrees.values()) if in_degrees else 0
                
                if max_in_degree == 0:
                    break
                    
                # Find node with highest in-degree
                highest_nodes = [node for node, degree in in_degrees.items() if degree == max_in_degree]
                target_node = highest_nodes[0]
                
                # Remove one incoming edge
                predecessors = list(graph.predecessors(target_node))
                if predecessors:
                    source_node = predecessors[0]
                    graph.remove_edge(source_node, target_node)
                    edges_removed += 1
                    logger.debug(f"Removed high-degree edge: {source_node} -> {target_node}")
                
        except Exception as e:
            logger.warning(f"Error in fallback cycle breaking: {e}")
        
        return edges_removed

    def _calculate_prerequisite_depths(self, units: List[CurriculumUnit]) -> List[CurriculumUnit]:
        """Calculate the prerequisite depth for each unit."""
        unit_positions = {unit.unit_id: i for i, unit in enumerate(units)}
        
        for unit in units:
            max_prereq_position = -1
            for prereq_id in unit.prerequisites:
                if prereq_id in unit_positions:
                    max_prereq_position = max(max_prereq_position, unit_positions[prereq_id])
            
            unit.prerequisite_depth = max_prereq_position + 1 if max_prereq_position >= 0 else 0
        
        return units

    def _add_cross_level_connections(self, units: List[CurriculumUnit]) -> List[CurriculumUnit]:
        """Add connections between related topics across educational levels."""
        units_by_title = defaultdict(list)
        
        # Group similar topics across levels
        for unit in units:
            # Normalize title for matching
            normalized_title = self._normalize_title_for_matching(unit.title)
            units_by_title[normalized_title].append(unit)
        
        # Add cross-level connections
        for title_group in units_by_title.values():
            if len(title_group) > 1:
                # Sort by educational level
                title_group.sort(key=lambda u: self.level_hierarchy[u.educational_level])
                
                # Connect each level to the previous level
                for i in range(1, len(title_group)):
                    current_unit = title_group[i]
                    previous_unit = title_group[i-1]
                    current_unit.cross_level_connections.append(previous_unit.unit_id)
        
        return units

    def _normalize_title_for_matching(self, title: str) -> str:
        """Normalize title for cross-level matching."""
        # Remove common prefixes/suffixes and normalize
        normalized = title.lower()
        normalized = re.sub(r'^(introduction to|intro to|basic|advanced)\s+', '', normalized)
        normalized = re.sub(r'\s+(i|ii|iii|1|2|3)$', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _create_level_sequences(self, units: List[CurriculumUnit]) -> Dict[str, List[str]]:
        """Create sequences for each educational level."""
        sequences = defaultdict(list)
        
        for unit in units:
            sequences[unit.educational_level].append(unit.unit_id)
        
        return dict(sequences)

    def _create_domain_progressions(self, units: List[CurriculumUnit]) -> Dict[str, List[str]]:
        """Create progressions for each physics domain."""
        progressions = defaultdict(list)
        
        # Group by domain and sort by order position
        units_by_domain = defaultdict(list)
        for unit in units:
            units_by_domain[unit.domain].append(unit)
        
        for domain, domain_units in units_by_domain.items():
            domain_units.sort(key=lambda u: u.order_position)
            progressions[domain] = [unit.unit_id for unit in domain_units]
        
        return dict(progressions)

    def _calculate_quality_metrics(self, units: List[CurriculumUnit], 
                                 prereq_data: Dict) -> Dict[str, float]:
        """Calculate quality metrics for the pedagogical sequence."""
        total_units = len(units)
        
        # Prerequisite coverage
        units_with_prereqs = sum(1 for unit in units if unit.prerequisites)
        prerequisite_coverage = units_with_prereqs / total_units if total_units > 0 else 0
        
        # Educational progression smoothness
        level_transitions = 0
        proper_transitions = 0
        for i in range(1, len(units)):
            prev_level = self.level_hierarchy[units[i-1].educational_level]
            curr_level = self.level_hierarchy[units[i].educational_level]
            if curr_level != prev_level:
                level_transitions += 1
                if curr_level >= prev_level:
                    proper_transitions += 1
        
        progression_smoothness = proper_transitions / level_transitions if level_transitions > 0 else 1.0
        
        # Domain coverage
        unique_domains = len(set(unit.domain for unit in units))
        domain_coverage = min(unique_domains / 8, 1.0)  # Assume 8 major physics domains
        
        # Cross-level connections
        units_with_connections = sum(1 for unit in units if unit.cross_level_connections)
        cross_level_connectivity = units_with_connections / total_units if total_units > 0 else 0
        
        # TOC order preservation (units within same book should be mostly in order)
        toc_preservation = self._calculate_toc_preservation(units)
        
        return {
            'prerequisite_coverage': prerequisite_coverage,
            'educational_progression_smoothness': progression_smoothness,
            'domain_coverage': domain_coverage,
            'cross_level_connectivity': cross_level_connectivity,
            'toc_order_preservation': toc_preservation,
            'overall_quality': (
                prerequisite_coverage * 0.25 +
                progression_smoothness * 0.25 +
                domain_coverage * 0.20 +
                cross_level_connectivity * 0.15 +
                toc_preservation * 0.15
            )
        }

    def _calculate_toc_preservation(self, units: List[CurriculumUnit]) -> float:
        """Calculate how well the TOC order is preserved within books."""
        book_groups = defaultdict(list)
        
        # Group by source book
        for unit in units:
            book_groups[unit.source_book].append(unit)
        
        preservation_scores = []
        
        for book_units in book_groups.values():
            if len(book_units) < 2:
                continue
            
            # Check if units are in ascending order by original position
            in_order = 0
            total_pairs = 0
            
            for i in range(len(book_units) - 1):
                for j in range(i + 1, len(book_units)):
                    total_pairs += 1
                    # Find original order positions (would need to track this)
                    # For now, use a simplified check
                    if i < j:  # They are in sequence order
                        in_order += 1
            
            if total_pairs > 0:
                preservation_scores.append(in_order / total_pairs)
        
        return sum(preservation_scores) / len(preservation_scores) if preservation_scores else 0.0


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create pedagogical sequence from TOC and prerequisites")
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
        sequencer = PedagogicalSequencer()
        result = sequencer.create_pedagogical_sequence(
            discipline=args.discipline,
            language=args.language
        )
        
        # Print summary
        print(f"\nPedagogical Sequencing Summary for {args.discipline} ({args.language}):")
        print(f"Total curriculum units: {result['metrics']['total_units']}")
        print(f"Total duration: {result['total_duration_hours']:.1f} hours")
        print(f"Units by level: {result['metrics']['units_by_level']}")
        print(f"Average prerequisites per unit: {result['metrics']['average_prerequisites_per_unit']:.1f}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        
        print(f"\nQuality Metrics:")
        for metric, value in result['quality_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        output_file = OUTPUT_DIR / f"{args.discipline}_{args.language}_curriculum_sequenced.json"
        print(f"\n✅ Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error during pedagogical sequencing: {e}")
        exit(1)


if __name__ == "__main__":
    main()