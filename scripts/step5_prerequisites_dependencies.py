#!/usr/bin/env python3
"""
Step 5: Prerequisites & Dependencies System

This module builds robust prerequisite relationships using both hierarchical position
inference and LLM-enhanced pedagogical dependency detection.

Features:
- Hierarchical position-based inference
- LLM-enhanced pedagogical dependency detection
- Physics domain knowledge integration
- Cycle detection and resolution
- Confidence scoring for prerequisites
- Graph structure construction
"""

import sys
import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import argparse
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.curriculum_utils import (
    CurriculumConfig, CurriculumLogger, LLMClient, FileManager, 
    DataValidator, load_config
)


@dataclass
class PrerequisiteRelationship:
    """Represents a prerequisite relationship between topics."""
    source: str  # The prerequisite topic
    target: str  # The topic that requires the prerequisite
    relationship_type: str  # "hierarchical", "domain", "llm", "explicit"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Human-readable explanation
    path: List[str]  # Full hierarchical path for context
    domain: str  # Physics domain
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relationship_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "path": self.path,
            "domain": self.domain
        }


class PrerequisitesDependenciesSystem:
    """Main system for building prerequisite relationships and dependency graphs."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.file_manager = FileManager(logger)
        
        # Graph structures
        self.prerequisite_graph = nx.DiGraph()
        self.relationships = []
        
        # Physics domain knowledge
        self.physics_prerequisites = self._define_physics_prerequisites()
        self.concept_difficulty_levels = self._define_difficulty_levels()
        
        # Caching for LLM responses
        self.llm_cache = {}
    
    def _define_physics_prerequisites(self) -> Dict[str, List[str]]:
        """Define known physics prerequisite relationships."""
        return {
            # Mathematical Prerequisites
            "algebra": [],
            "trigonometry": ["algebra"],
            "calculus": ["algebra", "trigonometry"],
            "differential_equations": ["calculus"],
            "linear_algebra": ["algebra"],
            "vector_analysis": ["calculus", "linear_algebra"],
            
            # Mechanics Prerequisites
            "kinematics": ["algebra", "trigonometry"],
            "dynamics": ["kinematics", "vector_analysis"],
            "newton_laws": ["kinematics"],
            "energy_work": ["newton_laws", "calculus"],
            "momentum": ["newton_laws"],
            "rotation": ["dynamics", "vector_analysis"],
            "oscillations": ["energy_work", "trigonometry"],
            "waves": ["oscillations", "trigonometry"],
            
            # Thermodynamics Prerequisites
            "temperature_heat": ["energy_work"],
            "kinetic_theory": ["statistical_mechanics", "mechanics"],
            "thermodynamic_laws": ["temperature_heat"],
            "heat_engines": ["thermodynamic_laws"],
            "entropy": ["thermodynamic_laws", "statistical_mechanics"],
            
            # Electricity and Magnetism Prerequisites
            "electrostatics": ["vector_analysis", "calculus"],
            "electric_fields": ["electrostatics"],
            "electric_potential": ["electric_fields", "energy_work"],
            "capacitance": ["electric_potential"],
            "current_resistance": ["electrostatics"],
            "circuits": ["current_resistance", "algebra"],
            "magnetic_fields": ["electrostatics", "vector_analysis"],
            "electromagnetic_induction": ["magnetic_fields", "calculus"],
            "ac_circuits": ["circuits", "oscillations"],
            "electromagnetic_waves": ["electromagnetic_induction", "waves"],
            
            # Modern Physics Prerequisites
            "special_relativity": ["mechanics", "electromagnetic_waves"],
            "quantum_mechanics": ["waves", "linear_algebra", "differential_equations"],
            "atomic_physics": ["quantum_mechanics", "electrostatics"],
            "nuclear_physics": ["atomic_physics"],
            "particle_physics": ["nuclear_physics", "special_relativity"],
            
            # Optics Prerequisites
            "geometric_optics": ["trigonometry"],
            "wave_optics": ["waves", "electromagnetic_waves"],
            "laser_physics": ["atomic_physics", "wave_optics"]
        }
    
    def _define_difficulty_levels(self) -> Dict[str, int]:
        """Define relative difficulty levels for physics concepts."""
        return {
            # Level 1: Basic concepts
            "units_measurement": 1,
            "scientific_notation": 1,
            "significant_figures": 1,
            "dimensional_analysis": 1,
            
            # Level 2: Introductory physics
            "kinematics": 2,
            "forces": 2,
            "energy": 2,
            "temperature_heat": 2,
            "waves_basic": 2,
            
            # Level 3: Intermediate physics
            "dynamics": 3,
            "rotation": 3,
            "oscillations": 3,
            "thermodynamics": 3,
            "electrostatics": 3,
            "circuits": 3,
            
            # Level 4: Advanced undergraduate
            "electromagnetic_fields": 4,
            "electromagnetic_waves": 4,
            "statistical_mechanics": 4,
            "quantum_mechanics_intro": 4,
            
            # Level 5: Graduate level
            "quantum_field_theory": 5,
            "general_relativity": 5,
            "many_body_physics": 5,
            "advanced_statistical_mechanics": 5
        }
    
    def extract_topics_from_hierarchy(self, hierarchy: Dict) -> Dict[str, Dict]:
        """Extract all topics with their full paths from the hierarchy."""
        topics = {}
        
        def traverse_hierarchy(obj, path=[], level=1):
            if level > 6:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = path + [key]
                    
                    # Store topic information
                    topic_id = " → ".join(current_path)
                    topics[topic_id] = {
                        "title": key,
                        "path": current_path.copy(),
                        "level": level,
                        "domain": current_path[0] if current_path else "unknown",
                        "full_path": " → ".join(current_path)
                    }
                    
                    # Recurse
                    traverse_hierarchy(value, current_path, level + 1)
            elif isinstance(obj, list):
                # Level 6 learning elements
                for i, element in enumerate(obj):
                    if isinstance(element, str):
                        element_path = path + [element]
                        topic_id = " → ".join(element_path)
                        topics[topic_id] = {
                            "title": element,
                            "path": element_path.copy(),
                            "level": level,
                            "domain": path[0] if path else "unknown",
                            "full_path": " → ".join(element_path)
                        }
        
        # Process core and electives
        for classification, content in hierarchy.items():
            if isinstance(content, dict):
                traverse_hierarchy(content, [classification])
        
        return topics
    
    def infer_hierarchical_prerequisites(self, topics: Dict[str, Dict]) -> List[PrerequisiteRelationship]:
        """Infer prerequisites based on hierarchical position."""
        self.logger.start_timer("hierarchical_prerequisites")
        
        relationships = []
        
        # Group topics by domain and path
        domain_topics = defaultdict(list)
        for topic_id, topic_info in topics.items():
            domain = topic_info.get("domain", "unknown")
            domain_topics[domain].append((topic_id, topic_info))
        
        for domain, domain_topic_list in domain_topics.items():
            # Sort by level and path length
            domain_topic_list.sort(key=lambda x: (x[1]["level"], len(x[1]["path"])))
            
            for i, (topic_id, topic_info) in enumerate(domain_topic_list):
                path = topic_info["path"]
                level = topic_info["level"]
                
                # Rule 1: Parent-child relationships
                if level > 1:
                    parent_path = path[:-1]
                    parent_id = " → ".join(parent_path)
                    
                    if parent_id in topics:
                        relationships.append(PrerequisiteRelationship(
                            source=parent_id,
                            target=topic_id,
                            relationship_type="hierarchical",
                            confidence=0.9,
                            reasoning=f"Hierarchical parent-child relationship in {domain}",
                            path=path,
                            domain=domain
                        ))
                
                # Rule 2: Sequential relationships within same parent
                for j in range(i):
                    other_topic_id, other_topic_info = domain_topic_list[j]
                    other_path = other_topic_info["path"]
                    other_level = other_topic_info["level"]
                    
                    # Same level, same parent, sequential
                    if (level == other_level and 
                        len(path) == len(other_path) and 
                        path[:-1] == other_path[:-1]):
                        
                        relationships.append(PrerequisiteRelationship(
                            source=other_topic_id,
                            target=topic_id,
                            relationship_type="hierarchical",
                            confidence=0.7,
                            reasoning=f"Sequential ordering within same category",
                            path=path,
                            domain=domain
                        ))
                        break  # Only immediate predecessor
        
        self.logger.info(f"Generated {len(relationships)} hierarchical prerequisites")
        self.logger.end_timer("hierarchical_prerequisites")
        return relationships
    
    def infer_domain_prerequisites(self, topics: Dict[str, Dict]) -> List[PrerequisiteRelationship]:
        """Infer prerequisites based on physics domain knowledge."""
        self.logger.start_timer("domain_prerequisites")
        
        relationships = []
        
        # Create topic lookup by normalized names
        normalized_topics = {}
        for topic_id, topic_info in topics.items():
            normalized_name = self._normalize_topic_name(topic_info["title"])
            if normalized_name not in normalized_topics:
                normalized_topics[normalized_name] = []
            normalized_topics[normalized_name].append((topic_id, topic_info))
        
        # Apply domain knowledge
        for concept, prereq_concepts in self.physics_prerequisites.items():
            if concept in normalized_topics:
                target_topics = normalized_topics[concept]
                
                for prereq_concept in prereq_concepts:
                    if prereq_concept in normalized_topics:
                        source_topics = normalized_topics[prereq_concept]
                        
                        # Create relationships between matching topics
                        for source_id, source_info in source_topics:
                            for target_id, target_info in target_topics:
                                # Ensure prerequisite comes before target in difficulty
                                source_difficulty = self._get_topic_difficulty(source_info)
                                target_difficulty = self._get_topic_difficulty(target_info)
                                
                                if source_difficulty <= target_difficulty:
                                    relationships.append(PrerequisiteRelationship(
                                        source=source_id,
                                        target=target_id,
                                        relationship_type="domain",
                                        confidence=0.8,
                                        reasoning=f"Physics domain knowledge: {prereq_concept} → {concept}",
                                        path=target_info["path"],
                                        domain=target_info["domain"]
                                    ))
        
        self.logger.info(f"Generated {len(relationships)} domain-based prerequisites")
        self.logger.end_timer("domain_prerequisites")
        return relationships
    
    def _normalize_topic_name(self, title: str) -> str:
        """Normalize topic name for matching against domain knowledge."""
        normalized = title.lower().replace("'", "").replace("-", "_")
        
        # Physics concept mappings
        mappings = {
            "motion": "kinematics",
            "force": "dynamics",
            "newton": "newton_laws",
            "energy": "energy_work",
            "heat": "temperature_heat",
            "electric": "electrostatics",
            "magnetic": "magnetic_fields",
            "wave": "waves",
            "quantum": "quantum_mechanics",
            "relativity": "special_relativity",
            "optics": "geometric_optics"
        }
        
        for key, value in mappings.items():
            if key in normalized:
                return value
        
        return normalized.replace(" ", "_")
    
    def _get_topic_difficulty(self, topic_info: Dict) -> int:
        """Estimate topic difficulty based on various factors."""
        # Base difficulty on hierarchy level
        level = topic_info.get("level", 1)
        base_difficulty = min(level, 5)
        
        # Adjust based on domain knowledge
        normalized_name = self._normalize_topic_name(topic_info["title"])
        if normalized_name in self.concept_difficulty_levels:
            return self.concept_difficulty_levels[normalized_name]
        
        # Adjust based on domain
        domain = topic_info.get("domain", "")
        domain_adjustments = {
            "mathematical_prerequisites": -1,
            "units_and_measurements": -1,
            "problem_solving_strategies": -1,
            "modern_physics": +1,
            "quantum": +2,
            "relativity": +2
        }
        
        for domain_key, adjustment in domain_adjustments.items():
            if domain_key.lower() in domain.lower():
                base_difficulty += adjustment
                break
        
        return max(1, min(5, base_difficulty))
    
    def llm_enhanced_prerequisites(self, topics: Dict[str, Dict], 
                                 sample_size: int = 20) -> List[PrerequisiteRelationship]:
        """Use LLM to identify additional prerequisite relationships."""
        if not self.llm_client.is_available():
            self.logger.warning("LLM not available for prerequisite enhancement")
            return []
        
        self.logger.start_timer("llm_prerequisites")
        
        relationships = []
        
        # Sample topics for LLM analysis (to avoid excessive API calls)
        topic_list = list(topics.items())
        if len(topic_list) > sample_size:
            # Sample strategically: different domains and levels
            sampled_topics = self._strategic_sample(topic_list, sample_size)
        else:
            sampled_topics = topic_list
        
        # Group topics by domain for context
        domain_groups = defaultdict(list)
        for topic_id, topic_info in sampled_topics:
            domain = topic_info.get("domain", "unknown")
            domain_groups[domain].append((topic_id, topic_info))
        
        for domain, domain_topics in domain_groups.items():
            if len(domain_topics) < 2:
                continue
            
            # Create topic context for LLM
            topic_context = []
            for topic_id, topic_info in domain_topics[:10]:  # Limit to 10 per domain
                topic_context.append({
                    "id": topic_id,
                    "title": topic_info["title"],
                    "path": " → ".join(topic_info["path"]),
                    "level": topic_info["level"]
                })
            
            # LLM prompt for prerequisite analysis
            llm_relationships = self._analyze_prerequisites_with_llm(domain, topic_context)
            relationships.extend(llm_relationships)
        
        self.logger.info(f"Generated {len(relationships)} LLM-enhanced prerequisites")
        self.logger.end_timer("llm_prerequisites")
        return relationships
    
    def _strategic_sample(self, topic_list: List[Tuple], sample_size: int) -> List[Tuple]:
        """Strategically sample topics to maximize LLM analysis value."""
        # Group by domain and level
        domain_level_groups = defaultdict(lambda: defaultdict(list))
        
        for topic_id, topic_info in topic_list:
            domain = topic_info.get("domain", "unknown")
            level = topic_info.get("level", 1)
            domain_level_groups[domain][level].append((topic_id, topic_info))
        
        sampled = []
        per_domain_quota = max(1, sample_size // len(domain_level_groups))
        
        for domain, level_groups in domain_level_groups.items():
            domain_sample = []
            per_level_quota = max(1, per_domain_quota // len(level_groups))
            
            for level, topics in level_groups.items():
                # Sample evenly from each level
                level_sample = topics[:per_level_quota]
                domain_sample.extend(level_sample)
            
            sampled.extend(domain_sample[:per_domain_quota])
        
        return sampled[:sample_size]
    
    def _analyze_prerequisites_with_llm(self, domain: str, 
                                      topic_context: List[Dict]) -> List[PrerequisiteRelationship]:
        """Use LLM to analyze prerequisites for a domain."""
        if len(topic_context) < 2:
            return []
        
        # Create cache key
        cache_key = f"prereq_{domain}_{hash(str(sorted([t['id'] for t in topic_context])))}"
        
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
        
        prompt = f"""
        You are a physics education expert. Analyze the following physics topics from the domain "{domain}" and identify prerequisite relationships.

        Topics to analyze:
        {json.dumps(topic_context, indent=2)}

        Instructions:
        1. Identify which topics should be learned BEFORE others
        2. Consider pedagogical flow and conceptual dependencies
        3. Only suggest prerequisites that are ESSENTIAL for understanding
        4. Assign confidence scores (0.0 to 1.0) based on how certain you are
        5. Provide brief reasoning for each relationship

        Respond with JSON format:
        {{
            "prerequisites": [
                {{
                    "source": "prerequisite_topic_id",
                    "target": "dependent_topic_id", 
                    "confidence": 0.8,
                    "reasoning": "Brief explanation why source is needed before target"
                }}
            ]
        }}

        Example:
        {{
            "prerequisites": [
                {{
                    "source": "core → Mechanics → Kinematics → Position",
                    "target": "core → Mechanics → Dynamics → Newton's Laws",
                    "confidence": 0.9,
                    "reasoning": "Understanding position and motion concepts is essential before learning forces"
                }}
            ]
        }}
        """
        
        response = self.llm_client.generate_completion(
            prompt=prompt,
            cache_key=cache_key,
            temperature=0.1
        )
        
        relationships = []
        
        if response:
            try:
                result = json.loads(response)
                prerequisites = result.get("prerequisites", [])
                
                for prereq in prerequisites:
                    source_id = prereq.get("source", "")
                    target_id = prereq.get("target", "")
                    confidence = float(prereq.get("confidence", 0.5))
                    reasoning = prereq.get("reasoning", "LLM-identified relationship")
                    
                    # Validate that topics exist in our context
                    source_exists = any(t["id"] == source_id for t in topic_context)
                    target_exists = any(t["id"] == target_id for t in topic_context)
                    
                    if source_exists and target_exists and source_id != target_id:
                        # Find topic info
                        target_info = next(t for t in topic_context if t["id"] == target_id)
                        
                        relationships.append(PrerequisiteRelationship(
                            source=source_id,
                            target=target_id,
                            relationship_type="llm",
                            confidence=confidence,
                            reasoning=reasoning,
                            path=target_info["path"].split(" → "),
                            domain=domain
                        ))
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"Failed to parse LLM prerequisite response: {e}")
        
        # Cache the result
        self.llm_cache[cache_key] = relationships
        return relationships
    
    def build_prerequisite_graph(self, relationships: List[PrerequisiteRelationship]) -> nx.DiGraph:
        """Build a directed graph of prerequisite relationships."""
        self.logger.start_timer("build_graph")
        
        graph = nx.DiGraph()
        
        # Add nodes and edges
        for rel in relationships:
            # Add nodes with attributes
            graph.add_node(rel.source, title=rel.source.split(" → ")[-1])
            graph.add_node(rel.target, title=rel.target.split(" → ")[-1])
            
            # Add edge with relationship data
            graph.add_edge(
                rel.source, 
                rel.target,
                weight=rel.confidence,
                type=rel.relationship_type,
                reasoning=rel.reasoning,
                domain=rel.domain
            )
        
        self.logger.info(f"Built prerequisite graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        self.logger.end_timer("build_graph")
        return graph
    
    def detect_and_resolve_cycles(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[Tuple]]:
        """Detect and resolve cycles in the prerequisite graph using efficient approach."""
        self.logger.start_timer("cycle_resolution")
        
        removed_edges = []
        max_iterations = 1000  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Use faster cycle detection for large graphs
                if graph.number_of_edges() > 10000:
                    # Use greedy approach: remove lowest confidence edges until DAG
                    if not nx.is_directed_acyclic_graph(graph):
                        # Find all edges sorted by confidence (ascending)
                        edges_by_confidence = sorted(
                            [(u, v, data.get('weight', 0.5)) for u, v, data in graph.edges(data=True)],
                            key=lambda x: x[2]
                        )
                        
                        # Remove lowest confidence edges until DAG
                        for u, v, weight in edges_by_confidence:
                            if graph.has_edge(u, v):
                                edge_data = graph[u][v].copy()
                                graph.remove_edge(u, v)
                                removed_edges.append((u, v, edge_data))
                                
                                # Check if DAG now
                                if nx.is_directed_acyclic_graph(graph):
                                    break
                        break
                else:
                    # Use precise cycle detection for smaller graphs
                    try:
                        cycle = next(nx.simple_cycles(graph))
                        if len(cycle) < 2:
                            break
                        
                        # Find edges in cycle with their weights
                        cycle_edges = []
                        for i in range(len(cycle)):
                            u = cycle[i]
                            v = cycle[(i + 1) % len(cycle)]
                            if graph.has_edge(u, v):
                                weight = graph[u][v].get('weight', 0.5)
                                cycle_edges.append((u, v, weight))
                        
                        if cycle_edges:
                            # Remove edge with minimum weight (lowest confidence)
                            min_edge = min(cycle_edges, key=lambda x: x[2])
                            u, v, weight = min_edge
                            
                            edge_data = graph[u][v].copy()
                            graph.remove_edge(u, v)
                            removed_edges.append((u, v, edge_data))
                            
                            self.logger.debug(f"Removed cycle edge: {u} → {v} (confidence: {weight:.2f})")
                    
                    except StopIteration:
                        # No more cycles
                        break
                
            except Exception as e:
                self.logger.warning(f"Error in cycle detection: {e}")
                break
            
            iteration += 1
        
        if iteration >= max_iterations:
            self.logger.warning(f"Cycle resolution stopped after {max_iterations} iterations")
        
        self.logger.info(f"Removed {len(removed_edges)} edges to resolve cycles")
        self.logger.end_timer("cycle_resolution")
        return graph, removed_edges
    
    def analyze_graph_properties(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze properties of the prerequisite graph."""
        analysis = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_dag": nx.is_directed_acyclic_graph(graph),
            "weakly_connected_components": nx.number_weakly_connected_components(graph),
            "strongly_connected_components": nx.number_strongly_connected_components(graph)
        }
        
        if analysis["is_dag"]:
            try:
                analysis["topological_levels"] = len(list(nx.topological_generations(graph)))
            except:
                analysis["topological_levels"] = 0
        
        # Confidence distribution
        weights = [data.get('weight', 0.5) for _, _, data in graph.edges(data=True)]
        if weights:
            analysis["avg_confidence"] = sum(weights) / len(weights)
            analysis["min_confidence"] = min(weights)
            analysis["max_confidence"] = max(weights)
        
        # Relationship type distribution
        types = [data.get('type', 'unknown') for _, _, data in graph.edges(data=True)]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        analysis["relationship_types"] = type_counts
        
        return analysis
    
    def process_hierarchy(self, hierarchy_data: Dict) -> Dict[str, Any]:
        """Main processing function for building prerequisites from hierarchy."""
        self.logger.start_timer("full_prerequisites_processing")
        
        hierarchy = hierarchy_data.get("hierarchy", {})
        
        # Extract topics
        self.logger.info("Extracting topics from hierarchy...")
        topics = self.extract_topics_from_hierarchy(hierarchy)
        self.logger.info(f"Extracted {len(topics)} topics")
        
        # Generate prerequisite relationships
        all_relationships = []
        
        # 1. Hierarchical prerequisites
        hierarchical_rels = self.infer_hierarchical_prerequisites(topics)
        all_relationships.extend(hierarchical_rels)
        
        # 2. Domain-based prerequisites
        domain_rels = self.infer_domain_prerequisites(topics)
        all_relationships.extend(domain_rels)
        
        # 3. LLM-enhanced prerequisites (sample size based on config)
        llm_sample_size = min(50, len(topics) // 5)  # Sample up to 50 or 20% of topics
        if llm_sample_size > 5:
            llm_rels = self.llm_enhanced_prerequisites(topics, llm_sample_size)
            all_relationships.extend(llm_rels)
        
        self.logger.info(f"Generated {len(all_relationships)} total prerequisite relationships")
        
        # Build graph
        graph = self.build_prerequisite_graph(all_relationships)
        
        # Detect and resolve cycles
        if self.config.cycle_detection_enabled:
            graph, removed_edges = self.detect_and_resolve_cycles(graph)
        else:
            removed_edges = []
        
        # Analyze graph properties
        graph_analysis = self.analyze_graph_properties(graph)
        
        # Store results
        self.prerequisite_graph = graph
        self.relationships = all_relationships
        
        self.logger.end_timer("full_prerequisites_processing")
        
        return {
            "topics": topics,
            "relationships": [rel.to_dict() for rel in all_relationships],
            "graph_analysis": graph_analysis,
            "removed_cycles": len(removed_edges),
            "total_relationships": len(all_relationships)
        }
    
    def create_output(self, processing_results: Dict) -> Dict[str, Any]:
        """Create structured output with all prerequisite information."""
        return {
            "metadata": {
                "total_topics": len(processing_results["topics"]),
                "total_relationships": processing_results["total_relationships"],
                "removed_cycles": processing_results["removed_cycles"],
                "graph_properties": processing_results["graph_analysis"],
                "confidence_threshold": self.config.confidence_threshold,
                "cycle_detection_enabled": self.config.cycle_detection_enabled,
                "timestamp": self.logger.logger.handlers[0].formatter.formatTime(
                    self.logger.logger.makeRecord("", 0, "", 0, "", (), None)
                )
            },
            "topics": processing_results["topics"],
            "prerequisites": processing_results["relationships"],
            "graph_data": {
                "nodes": [
                    {
                        "id": node,
                        "title": data.get("title", node.split(" → ")[-1]),
                        "domain": processing_results["topics"].get(node, {}).get("domain", "unknown")
                    }
                    for node, data in self.prerequisite_graph.nodes(data=True)
                ],
                "edges": [
                    {
                        "source": source,
                        "target": target,
                        "weight": data.get("weight", 0.5),
                        "type": data.get("type", "unknown"),
                        "reasoning": data.get("reasoning", ""),
                        "domain": data.get("domain", "unknown")
                    }
                    for source, target, data in self.prerequisite_graph.edges(data=True)
                ]
            }
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prerequisites & Dependencies System")
    parser.add_argument("--input", "-i", default="six_level_hierarchy.json",
                       help="Input six-level hierarchy file")
    parser.add_argument("--output", "-o", default="curriculum_with_prerequisites.json",
                       help="Output curriculum with prerequisites file")
    parser.add_argument("--config", "-c", default="config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--confidence-threshold", "-t", type=float,
                       help="Minimum confidence threshold for prerequisites")
    parser.add_argument("--no-cycles", action="store_true",
                       help="Disable cycle detection and resolution")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM-enhanced prerequisite detection")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    if args.confidence_threshold:
        config.confidence_threshold = args.confidence_threshold
    if args.no_cycles:
        config.cycle_detection_enabled = False
    if args.no_llm:
        config.openai_api_key = ""
        import os
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    logger = CurriculumLogger("step5_prerequisites", "DEBUG" if args.verbose else "INFO")
    file_manager = FileManager(logger)
    
    logger.info("Starting Prerequisites & Dependencies System")
    logger.info(f"Confidence threshold: {config.confidence_threshold}")
    logger.info(f"Cycle detection: {'enabled' if config.cycle_detection_enabled else 'disabled'}")
    logger.info(f"LLM enhancement: {'enabled' if config.openai_api_key else 'disabled'}")
    
    # Load input data
    logger.start_timer("data_loading")
    hierarchy_data = file_manager.load_json(args.input)
    if not hierarchy_data:
        logger.error(f"Failed to load input file: {args.input}")
        return 1
    
    logger.end_timer("data_loading")
    
    # Process prerequisites
    processor = PrerequisitesDependenciesSystem(config, logger)
    
    try:
        processing_results = processor.process_hierarchy(hierarchy_data)
        output_data = processor.create_output(processing_results)
        
        # Save results
        logger.start_timer("output_saving")
        if file_manager.save_json(output_data, args.output):
            logger.info(f"Prerequisites and dependencies saved to: {args.output}")
        else:
            logger.error("Failed to save output")
            return 1
        logger.end_timer("output_saving")
        
        # Performance summary
        logger.log_performance_summary()
        
        # Summary statistics
        metadata = output_data["metadata"]
        logger.info("Prerequisites Summary:")
        logger.info(f"  Total topics: {metadata['total_topics']}")
        logger.info(f"  Total relationships: {metadata['total_relationships']}")
        logger.info(f"  Graph density: {metadata['graph_properties']['density']:.3f}")
        logger.info(f"  Is DAG: {metadata['graph_properties']['is_dag']}")
        logger.info(f"  Cycles removed: {metadata['removed_cycles']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prerequisites processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())