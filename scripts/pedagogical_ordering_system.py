#!/usr/bin/env python3
"""
Comprehensive Pedagogical Ordering System

This module implements the phases outlined for proper curriculum sequencing:
1. TOC Extraction - Capture expert-ordered outlines
2. Topic Normalization - Unify cross-book content  
3. Prerequisite Mapping - Discover knowledge dependencies
4. Sequencing - Build learning pathway
5. Visualization - Enable expert review
6. Adaptivity - Improve learner outcomes
"""

import logging
import re
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, deque
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class TOCEntry:
    """Represents a table of contents entry with pedagogical metadata."""
    title: str
    level: int  # 1=chapter, 2=section, 3=subsection
    sequence_order: int  # Position in original TOC
    book_source: str
    page_number: Optional[int] = None
    section_number: Optional[str] = None
    parent_id: Optional[str] = None
    entry_id: str = ""
    children: List['TOCEntry'] = field(default_factory=list)
    normalized_concept: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    pedagogical_weight: float = 1.0
    
@dataclass 
class PedagogicalSequence:
    """Represents an ordered learning sequence with justification."""
    sequence_id: str
    concept_name: str
    order_position: int
    prerequisite_concepts: List[str]
    justification: str
    confidence_score: float
    source_books: List[str]
    educational_level: str
    
class OpenStaxTOCExtractor:
    """Extracts TOC from OpenStax collection XML files."""
    
    def __init__(self):
        self.namespace = {
            'col': 'http://cnx.rice.edu/collxml',
            'md': 'http://cnx.rice.edu/mdml'
        }
    
    def extract_from_collection_xml(self, xml_path: Path) -> List[TOCEntry]:
        """Extract TOC entries from OpenStax collection XML."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            entries = []
            sequence_order = 0
            
            # Find all subcollections (chapters) and modules (sections)
            content = root.find('.//col:content', self.namespace)
            if content is not None:
                entries.extend(self._parse_content_recursive(content, level=1, sequence_order=0, book_source=str(xml_path)))
            
            logger.info(f"Extracted {len(entries)} TOC entries from {xml_path}")
            return entries
            
        except Exception as e:
            logger.error(f"Failed to parse {xml_path}: {e}")
            return []
    
    def _parse_content_recursive(self, content_element, level: int, sequence_order: int, book_source: str) -> List[TOCEntry]:
        """Recursively parse content elements."""
        entries = []
        current_order = sequence_order
        
        for child in content_element:
            if child.tag.endswith('subcollection'):
                # This is a chapter/major section
                title_elem = child.find('.//md:title', self.namespace)
                title = title_elem.text if title_elem is not None else f"Chapter {current_order}"
                
                entry = TOCEntry(
                    title=title,
                    level=level,
                    sequence_order=current_order,
                    book_source=book_source,
                    entry_id=f"{book_source}_{level}_{current_order}"
                )
                
                # Parse subcontent
                subcontent = child.find('.//col:content', self.namespace)
                if subcontent is not None:
                    sub_entries = self._parse_content_recursive(subcontent, level + 1, 0, book_source)
                    entry.children.extend(sub_entries)
                
                entries.append(entry)
                current_order += 1
                
            elif child.tag.endswith('module'):
                # This is a section/module
                doc_id = child.get('document', f'module_{current_order}')
                entry = TOCEntry(
                    title=f"Module {doc_id}",
                    level=level,
                    sequence_order=current_order,
                    book_source=book_source,
                    entry_id=f"{book_source}_{level}_{current_order}_{doc_id}"
                )
                entries.append(entry)
                current_order += 1
        
        return entries

class ConceptNormalizer:
    """Normalizes physics concepts across different textbooks."""
    
    def __init__(self):
        # Physics concept mapping for normalization
        self.concept_mappings = {
            # Mechanics concepts
            'kinematics': ['motion', 'velocity', 'acceleration', 'displacement'],
            'dynamics': ['forces', 'newton\'s laws', 'friction', 'tension'],
            'energy_work': ['work', 'energy', 'power', 'conservation of energy'],
            'momentum': ['momentum', 'impulse', 'conservation of momentum', 'collisions'],
            'rotational_motion': ['rotation', 'torque', 'angular momentum', 'moment of inertia'],
            'oscillations': ['simple harmonic motion', 'waves', 'pendulum', 'springs'],
            'gravitation': ['gravity', 'gravitational force', 'orbits', 'kepler'],
            
            # Thermodynamics concepts  
            'temperature_heat': ['temperature', 'heat', 'thermal equilibrium'],
            'thermodynamic_laws': ['first law', 'second law', 'entropy', 'enthalpy'],
            'kinetic_theory': ['kinetic theory', 'ideal gas', 'molecular motion'],
            'heat_transfer': ['conduction', 'convection', 'radiation'],
            
            # Electromagnetism concepts
            'electrostatics': ['electric charge', 'electric field', 'coulomb\'s law', 'gauss\'s law'],
            'electric_potential': ['electric potential', 'voltage', 'capacitance'],
            'electric_current': ['current', 'resistance', 'ohm\'s law', 'circuits'],
            'magnetism': ['magnetic field', 'magnetic force', 'ampere\'s law'],
            'electromagnetic_induction': ['faraday\'s law', 'lenz\'s law', 'inductance'],
            'electromagnetic_waves': ['maxwell\'s equations', 'electromagnetic radiation', 'light'],
            
            # Waves and Optics
            'wave_properties': ['wavelength', 'frequency', 'amplitude', 'wave speed'],
            'wave_behavior': ['interference', 'diffraction', 'reflection', 'refraction'],
            'sound': ['sound waves', 'acoustics', 'doppler effect'],
            'geometric_optics': ['mirrors', 'lenses', 'ray optics', 'image formation'],
            'wave_optics': ['interference', 'diffraction', 'polarization'],
            
            # Modern Physics
            'special_relativity': ['time dilation', 'length contraction', 'lorentz transformation'],
            'quantum_mechanics': ['photons', 'wave-particle duality', 'uncertainty principle'],
            'atomic_physics': ['atomic structure', 'electron configurations', 'spectroscopy'],
            'nuclear_physics': ['radioactivity', 'nuclear reactions', 'fission', 'fusion']
        }
        
        # Standard physics curriculum sequence (based on expert textbook analysis)
        self.standard_sequence = [
            'units_measurement',           # 1
            'kinematics',                 # 2  
            'dynamics',                   # 3
            'energy_work',                # 4
            'momentum',                   # 5
            'rotational_motion',          # 6
            'oscillations',              # 7
            'gravitation',               # 8
            'temperature_heat',          # 9
            'thermodynamic_laws',        # 10
            'kinetic_theory',            # 11
            'heat_transfer',             # 12
            'electrostatics',            # 13
            'electric_potential',        # 14
            'electric_current',          # 15
            'magnetism',                 # 16
            'electromagnetic_induction', # 17
            'electromagnetic_waves',     # 18
            'wave_properties',           # 19
            'wave_behavior',             # 20
            'sound',                     # 21
            'geometric_optics',          # 22
            'wave_optics',               # 23
            'special_relativity',        # 24
            'quantum_mechanics',         # 25
            'atomic_physics',            # 26
            'nuclear_physics'            # 27
        ]
    
    def normalize_concept(self, title: str) -> Optional[str]:
        """Map a TOC title to a normalized physics concept."""
        title_lower = title.lower()
        
        for concept, keywords in self.concept_mappings.items():
            if any(keyword in title_lower for keyword in keywords):
                return concept
        
        return None
    
    def get_concept_order(self, concept: str) -> int:
        """Get the standard pedagogical order for a concept."""
        try:
            return self.standard_sequence.index(concept) + 1
        except ValueError:
            return 999  # Unknown concepts go to end

class PrerequisiteMapper:
    """Maps prerequisite relationships between physics concepts."""
    
    def __init__(self):
        # Expert-defined prerequisite relationships in physics
        self.prerequisite_graph = {
            'kinematics': [],
            'dynamics': ['kinematics'],
            'energy_work': ['dynamics'],
            'momentum': ['dynamics'],
            'rotational_motion': ['dynamics', 'energy_work'],
            'oscillations': ['energy_work', 'dynamics'],
            'gravitation': ['dynamics', 'energy_work'],
            'temperature_heat': ['energy_work'],
            'thermodynamic_laws': ['temperature_heat'],
            'kinetic_theory': ['temperature_heat'],
            'heat_transfer': ['temperature_heat'],
            'electrostatics': [],
            'electric_potential': ['electrostatics'],
            'electric_current': ['electric_potential'],
            'magnetism': ['electric_current'],
            'electromagnetic_induction': ['magnetism', 'electric_current'],
            'electromagnetic_waves': ['electromagnetic_induction'],
            'wave_properties': [],
            'wave_behavior': ['wave_properties'],
            'sound': ['wave_properties', 'wave_behavior'],
            'geometric_optics': ['wave_properties'],
            'wave_optics': ['wave_properties', 'wave_behavior'],
            'special_relativity': ['dynamics', 'energy_work', 'electromagnetic_waves'],
            'quantum_mechanics': ['wave_behavior', 'electromagnetic_waves'],
            'atomic_physics': ['quantum_mechanics', 'electrostatics'],
            'nuclear_physics': ['atomic_physics', 'energy_work']
        }
    
    def get_prerequisites(self, concept: str) -> List[str]:
        """Get the prerequisite concepts for a given concept."""
        return self.prerequisite_graph.get(concept, [])
    
    def validate_sequence(self, concepts: List[str]) -> Tuple[bool, List[str]]:
        """Validate if a sequence respects prerequisite relationships."""
        seen_concepts = set()
        violations = []
        
        for concept in concepts:
            prerequisites = self.get_prerequisites(concept)
            
            for prereq in prerequisites:
                if prereq not in seen_concepts:
                    violations.append(f"{concept} requires {prereq} but {prereq} not seen yet")
            
            seen_concepts.add(concept)
        
        return len(violations) == 0, violations

class PedagogicalSequencer:
    """Creates pedagogically sound learning sequences using topological sorting."""
    
    def __init__(self, normalizer: ConceptNormalizer, prerequisite_mapper: PrerequisiteMapper):
        self.normalizer = normalizer
        self.prerequisite_mapper = prerequisite_mapper
    
    def create_learning_sequence(self, toc_entries: List[TOCEntry]) -> List[PedagogicalSequence]:
        """Create a pedagogically ordered learning sequence from TOC entries."""
        # Phase 1: Normalize concepts
        normalized_entries = []
        for entry in toc_entries:
            normalized_concept = self.normalizer.normalize_concept(entry.title)
            if normalized_concept:
                entry.normalized_concept = normalized_concept
                normalized_entries.append(entry)
        
        # Phase 2: Group by concept and find consensus ordering
        concept_groups = defaultdict(list)
        for entry in normalized_entries:
            concept_groups[entry.normalized_concept].append(entry)
        
        # Phase 3: Create dependency graph
        graph = nx.DiGraph()
        
        # Add nodes
        for concept in concept_groups.keys():
            graph.add_node(concept)
        
        # Add prerequisite edges
        for concept in concept_groups.keys():
            prerequisites = self.prerequisite_mapper.get_prerequisites(concept)
            for prereq in prerequisites:
                if prereq in concept_groups:
                    graph.add_edge(prereq, concept)
        
        # Phase 4: Topological sort for learning sequence
        try:
            ordered_concepts = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            logger.warning("Circular dependencies detected, using heuristic ordering")
            ordered_concepts = self._heuristic_ordering(concept_groups.keys())
        
        # Phase 5: Create pedagogical sequences
        sequences = []
        for i, concept in enumerate(ordered_concepts):
            entries = concept_groups[concept]
            source_books = list(set(entry.book_source for entry in entries))
            
            sequence = PedagogicalSequence(
                sequence_id=f"seq_{i:03d}",
                concept_name=concept,
                order_position=i + 1,
                prerequisite_concepts=self.prerequisite_mapper.get_prerequisites(concept),
                justification=f"Standard physics curriculum order based on {len(source_books)} textbook(s)",
                confidence_score=min(1.0, len(entries) / 3.0),  # Higher confidence with more sources
                source_books=source_books,
                educational_level=self._determine_level(entries)
            )
            sequences.append(sequence)
        
        return sequences
    
    def _heuristic_ordering(self, concepts: Set[str]) -> List[str]:
        """Fallback heuristic ordering when topological sort fails."""
        # Use the standard sequence as fallback
        ordered = []
        remaining = set(concepts)
        
        for standard_concept in self.normalizer.standard_sequence:
            if standard_concept in remaining:
                ordered.append(standard_concept)
                remaining.remove(standard_concept)
        
        # Add any remaining concepts at the end
        ordered.extend(sorted(remaining))
        return ordered
    
    def _determine_level(self, entries: List[TOCEntry]) -> str:
        """Determine educational level from TOC entries."""
        # Simple heuristic based on complexity indicators
        for entry in entries:
            title_lower = entry.title.lower()
            if any(advanced in title_lower for advanced in ['quantum', 'relativity', 'field theory', 'graduate']):
                return 'graduate'
            elif any(intermediate in title_lower for intermediate in ['calculus', 'vector', 'differential']):
                return 'undergraduate'
        
        return 'high_school'

class CurriculumVisualizer:
    """Creates visualizations and summaries of the pedagogical sequence."""
    
    @staticmethod
    def create_sequence_table(sequences: List[PedagogicalSequence]) -> str:
        """Create a tabular summary of the pedagogical sequence."""
        headers = ["Order", "Concept", "Prerequisites", "Level", "Confidence", "Sources"]
        
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|"
        ]
        
        for seq in sequences:
            prereq_str = ", ".join(seq.prerequisite_concepts) if seq.prerequisite_concepts else "None"
            sources_str = f"{len(seq.source_books)} book(s)"
            confidence_str = f"{seq.confidence_score:.2f}"
            
            row = [
                str(seq.order_position),
                seq.concept_name,
                prereq_str,
                seq.educational_level,
                confidence_str,
                sources_str
            ]
            
            table_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(table_lines)
    
    @staticmethod
    def create_prerequisite_graph_data(sequences: List[PedagogicalSequence]) -> Dict[str, Any]:
        """Create data structure for prerequisite graph visualization."""
        nodes = []
        edges = []
        
        for seq in sequences:
            nodes.append({
                'id': seq.concept_name,
                'label': seq.concept_name.replace('_', ' ').title(),
                'order': seq.order_position,
                'level': seq.educational_level,
                'confidence': seq.confidence_score
            })
            
            for prereq in seq.prerequisite_concepts:
                edges.append({
                    'source': prereq,
                    'target': seq.concept_name
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_concepts': len(nodes),
                'total_prerequisites': len(edges),
                'levels': list(set(seq.educational_level for seq in sequences))
            }
        }

class PedagogicalOrderingSystem:
    """Main system that coordinates all phases of pedagogical ordering."""
    
    def __init__(self):
        self.toc_extractor = OpenStaxTOCExtractor()
        self.normalizer = ConceptNormalizer()
        self.prerequisite_mapper = PrerequisiteMapper()
        self.sequencer = PedagogicalSequencer(self.normalizer, self.prerequisite_mapper)
        self.visualizer = CurriculumVisualizer()
    
    def generate_pedagogical_curriculum(self, physics_books_path: Path) -> Dict[str, Any]:
        """
        Complete pedagogical ordering pipeline.
        
        Returns comprehensive curriculum data with proper pedagogical sequencing.
        """
        logger.info("Starting comprehensive pedagogical ordering system")
        
        # Phase 1: TOC Extraction
        logger.info("Phase 1: Extracting TOC data from textbooks")
        all_toc_entries = []
        
        # Find all collection XML files
        xml_files = list(physics_books_path.glob("**/collections/*.collection.xml"))
        logger.info(f"Found {len(xml_files)} collection XML files")
        
        for xml_file in xml_files:
            entries = self.toc_extractor.extract_from_collection_xml(xml_file)
            all_toc_entries.extend(entries)
        
        logger.info(f"Extracted {len(all_toc_entries)} total TOC entries")
        
        # Phase 2: Topic Normalization  
        logger.info("Phase 2: Normalizing concepts across textbooks")
        for entry in all_toc_entries:
            entry.normalized_concept = self.normalizer.normalize_concept(entry.title)
        
        # Phase 3: Prerequisite Mapping & Phase 4: Sequencing
        logger.info("Phase 3-4: Building prerequisite relationships and creating learning sequence")
        pedagogical_sequences = self.sequencer.create_learning_sequence(all_toc_entries)
        
        # Phase 5: Visualization
        logger.info("Phase 5: Creating visualizations and summaries")
        sequence_table = self.visualizer.create_sequence_table(pedagogical_sequences)
        graph_data = self.visualizer.create_prerequisite_graph_data(pedagogical_sequences)
        
        # Validation
        concept_names = [seq.concept_name for seq in pedagogical_sequences]
        is_valid, violations = self.prerequisite_mapper.validate_sequence(concept_names)
        
        result = {
            'pedagogical_sequences': [
                {
                    'sequence_id': seq.sequence_id,
                    'concept_name': seq.concept_name,
                    'order_position': seq.order_position,
                    'prerequisite_concepts': seq.prerequisite_concepts,
                    'justification': seq.justification,
                    'confidence_score': seq.confidence_score,
                    'source_books': seq.source_books,
                    'educational_level': seq.educational_level
                }
                for seq in pedagogical_sequences
            ],
            'sequence_table': sequence_table,
            'prerequisite_graph': graph_data,
            'validation': {
                'is_valid': is_valid,
                'violations': violations
            },
            'statistics': {
                'total_concepts': len(pedagogical_sequences),
                'total_toc_entries': len(all_toc_entries),
                'source_books': len(xml_files),
                'average_confidence': sum(seq.confidence_score for seq in pedagogical_sequences) / len(pedagogical_sequences) if pedagogical_sequences else 0
            }
        }
        
        logger.info("✅ Pedagogical ordering system completed successfully")
        return result

def main():
    """Test the pedagogical ordering system."""
    logging.basicConfig(level=logging.INFO)
    
    system = PedagogicalOrderingSystem()
    physics_path = Path("/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics")
    
    result = system.generate_pedagogical_curriculum(physics_path)
    
    print("=== PEDAGOGICAL SEQUENCE SUMMARY ===")
    print(result['sequence_table'])
    print("\n=== VALIDATION RESULTS ===")
    print(f"Valid sequence: {result['validation']['is_valid']}")
    if result['validation']['violations']:
        print("Violations:")
        for violation in result['validation']['violations']:
            print(f"  - {violation}")
    
    print("\n=== STATISTICS ===")
    for key, value in result['statistics'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()