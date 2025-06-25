#!/usr/bin/env python3
"""
Step 4: Six-Level Hierarchy Builder

This module enforces a six-level taxonomy hierarchy and builds a normalized curriculum
structure from classified TOC data using LLM-assisted synonym normalization.

Six-Level Hierarchy:
1. Domain (e.g., Mechanics)
2. Category (e.g., Dynamics) 
3. Concept (e.g., Newton's Laws)
4. Topic (e.g., Newton's Second Law)
5. Subtopic (e.g., F = ma)
6. Learning Elements (e.g., [Sample Problems, Labs, Applications])

Features:
- Enforced six-level taxonomy validation
- ChatGPT-4 synonym normalization
- Progressive academic level merging (HS → UG → Grad)
- Foundational content injection
- Automatic hierarchy restructuring
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.curriculum_utils import (
    CurriculumConfig, CurriculumLogger, LLMClient, FileManager, 
    DataValidator, load_config
)


@dataclass
class HierarchyNode:
    """Represents a node in the six-level hierarchy."""
    title: str
    level: int
    children: Dict[str, 'HierarchyNode']
    original_entries: List[Dict]
    canonical_name: str = ""
    alternative_names: List[str] = None
    academic_levels: Set[str] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.alternative_names is None:
            self.alternative_names = []
        if self.academic_levels is None:
            self.academic_levels = set()
        if not self.canonical_name:
            self.canonical_name = self.title


class SixLevelHierarchyBuilder:
    """Main builder for creating six-level hierarchical curriculum structure."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.file_manager = FileManager(logger)
        
        # Hierarchy tracking
        self.hierarchy = {"core": {}, "electives": {}}
        self.synonym_mappings = {}
        self.level_names = [
            "Domain", "Category", "Concept", "Topic", "Subtopic", "Learning Element"
        ]
        
        # Foundational content definitions
        self.foundational_content = self._define_foundational_content()
        
        # Physics domain knowledge
        self.physics_domains = self._define_physics_domains()
    
    def _define_foundational_content(self) -> Dict[str, Dict]:
        """Define foundational content that should always be included."""
        return {
            "Mathematical Prerequisites": {
                "Algebra and Trigonometry": {
                    "Basic Algebraic Operations": {
                        "Linear Equations": {
                            "Solving Linear Equations": [
                                "Single-variable equations",
                                "Multi-variable equations", 
                                "Systems of equations",
                                "Practice problems"
                            ]
                        },
                        "Quadratic Equations": {
                            "Solving Quadratic Equations": [
                                "Factoring method",
                                "Quadratic formula",
                                "Completing the square",
                                "Applications"
                            ]
                        }
                    },
                    "Trigonometric Functions": {
                        "Basic Trigonometry": {
                            "Trigonometric Ratios": [
                                "Sine, cosine, tangent",
                                "Unit circle",
                                "Trigonometric identities",
                                "Problem solving"
                            ]
                        }
                    }
                }
            },
            "Units and Measurements": {
                "Measurement Systems": {
                    "SI Units": {
                        "Base Units": {
                            "Fundamental Units": [
                                "Length (meter)",
                                "Mass (kilogram)",
                                "Time (second)",
                                "Electric current (ampere)",
                                "Temperature (kelvin)",
                                "Amount of substance (mole)",
                                "Luminous intensity (candela)"
                            ]
                        },
                        "Derived Units": {
                            "Common Derived Units": [
                                "Force (newton)",
                                "Energy (joule)",
                                "Power (watt)",
                                "Pressure (pascal)"
                            ]
                        }
                    }
                },
                "Measurement Techniques": {
                    "Precision and Accuracy": {
                        "Significant Figures": {
                            "Rules for Significant Figures": [
                                "Counting significant figures",
                                "Operations with significant figures",
                                "Rounding rules",
                                "Scientific notation"
                            ]
                        }
                    },
                    "Dimensional Analysis": {
                        "Unit Conversion": {
                            "Conversion Factors": [
                                "Unit conversion methods",
                                "Dimensional analysis technique",
                                "Complex unit conversions",
                                "Problem-solving applications"
                            ]
                        }
                    }
                }
            },
            "Problem-Solving Strategies": {
                "General Problem-Solving": {
                    "Systematic Approach": {
                        "Problem-Solving Steps": {
                            "Standard Method": [
                                "Identify given information",
                                "Determine what to find",
                                "Choose appropriate equations",
                                "Solve and check answer"
                            ]
                        }
                    },
                    "Estimation Techniques": {
                        "Order of Magnitude": {
                            "Approximation Methods": [
                                "Fermi problems",
                                "Back-of-envelope calculations",
                                "Reasonableness checks",
                                "Real-world applications"
                            ]
                        }
                    }
                }
            }
        }
    
    def _define_physics_domains(self) -> Dict[str, List[str]]:
        """Define physics domain categories for intelligent grouping."""
        return {
            "Mechanics": [
                "motion", "force", "energy", "momentum", "rotation", "oscillation",
                "dynamics", "kinematics", "statics", "work", "power", "conservation"
            ],
            "Thermodynamics": [
                "heat", "temperature", "entropy", "thermal", "gas", "kinetic theory",
                "thermodynamic", "calorimetry", "phase", "engine"
            ],
            "Electricity and Magnetism": [
                "electric", "magnetic", "electromagnetic", "circuit", "current", "voltage",
                "field", "charge", "capacitor", "inductor", "resistance", "ohm"
            ],
            "Waves and Optics": [
                "wave", "sound", "light", "optics", "interference", "diffraction",
                "reflection", "refraction", "lens", "mirror", "electromagnetic radiation"
            ],
            "Modern Physics": [
                "quantum", "relativity", "atomic", "nuclear", "particle", "photon",
                "electron", "radiation", "radioactive", "fusion", "fission"
            ],
            "Fluid Mechanics": [
                "fluid", "pressure", "buoyancy", "flow", "viscosity", "bernoulli",
                "hydrostatic", "hydrodynamic", "turbulence"
            ]
        }
    
    def normalize_title(self, title: str) -> str:
        """Normalize a title by removing common patterns and formatting."""
        if not title:
            return ""
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^(Chapter \d+:?\s*)', '', title.strip())
        title = re.sub(r'^(\d+\.?\s*)', '', title)
        title = re.sub(r'^(Section \d+\.?\d*:?\s*)', '', title)
        title = re.sub(r'^(Part [IVX]+:?\s*)', '', title)
        
        # Clean up formatting
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces to single
        title = title.strip()
        
        # Title case
        title = title.title()
        
        return title
    
    def classify_domain(self, title: str, toc_entries: List[Dict]) -> str:
        """Classify content into physics domains using keyword matching."""
        title_lower = title.lower()
        
        # Collect all text for analysis
        all_text = title_lower + " "
        for entry in toc_entries[:10]:  # Sample first 10 entries
            all_text += entry.get("title", "").lower() + " "
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in self.physics_domains.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                domain_scores[domain] = score
        
        # Return highest scoring domain or default
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return "General Physics"
    
    def llm_normalize_synonyms(self, titles: List[str], context: str = "") -> Dict[str, str]:
        """Use LLM to identify synonyms and create canonical mappings."""
        if not self.llm_client.is_available() or len(titles) < 2:
            return {title: title for title in titles}
        
        # Group similar titles for analysis
        title_groups = self._group_similar_titles(titles)
        
        mappings = {}
        
        for group in title_groups:
            if len(group) == 1:
                mappings[group[0]] = group[0]
                continue
            
            prompt = f"""
            You are a physics curriculum expert. Analyze these physics topic titles and identify which ones are synonyms or refer to the same concept. Create a canonical (preferred) title for each group of synonyms.

            Context: {context}
            
            Titles to analyze:
            {chr(10).join([f"- {title}" for title in group])}
            
            Rules:
            1. Group titles that refer to the same physics concept
            2. Choose the most clear, standard, and pedagogically appropriate title as canonical
            3. Maintain scientific accuracy and standard physics terminology
            4. Keep titles concise but descriptive
            
            Respond with JSON format:
            {{
                "groups": [
                    {{
                        "canonical": "Canonical Title",
                        "synonyms": ["synonym1", "synonym2", "..."]
                    }}
                ]
            }}
            
            Example:
            {{
                "groups": [
                    {{
                        "canonical": "Newton's Laws of Motion",
                        "synonyms": ["Newton's Laws", "Laws of Motion", "Newton's Three Laws"]
                    }}
                ]
            }}
            """
            
            cache_key = f"synonyms_{hash(str(sorted(group)))}_{hash(context)}"
            
            response = self.llm_client.generate_completion(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.1
            )
            
            if response:
                try:
                    result = json.loads(response)
                    
                    for group_data in result.get("groups", []):
                        canonical = group_data.get("canonical", "")
                        synonyms = group_data.get("synonyms", [])
                        
                        if canonical and synonyms:
                            for synonym in synonyms:
                                if synonym in group:  # Verify synonym is in original group
                                    mappings[synonym] = canonical
                        
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse LLM synonym response: {e}")
                    # Fallback to identity mapping
                    for title in group:
                        mappings[title] = title
            else:
                # Fallback to identity mapping
                for title in group:
                    mappings[title] = title
        
        return mappings
    
    def _group_similar_titles(self, titles: List[str]) -> List[List[str]]:
        """Group titles that might be synonyms based on similarity."""
        import difflib
        
        groups = []
        used_titles = set()
        
        for title in titles:
            if title in used_titles:
                continue
                
            # Find similar titles
            similar_titles = [title]
            used_titles.add(title)
            
            for other_title in titles:
                if other_title != title and other_title not in used_titles:
                    # Check similarity
                    similarity = difflib.SequenceMatcher(None, title.lower(), other_title.lower()).ratio()
                    
                    # Also check for common physics synonyms
                    if (similarity > 0.6 or 
                        self._are_physics_synonyms(title, other_title)):
                        similar_titles.append(other_title)
                        used_titles.add(other_title)
            
            groups.append(similar_titles)
        
        return groups
    
    def _are_physics_synonyms(self, title1: str, title2: str) -> bool:
        """Check if two titles are likely physics synonyms."""
        # Common physics synonym patterns
        synonym_patterns = [
            (r"newton'?s laws?", r"laws? of motion"),
            (r"conservation of energy", r"energy conservation"),
            (r"electric field", r"electrical field"),
            (r"magnetic field", r"magnetism"),
            (r"simple harmonic motion", r"shm"),
            (r"electromagnetic", r"em"),
        ]
        
        t1_lower = title1.lower()
        t2_lower = title2.lower()
        
        for pattern1, pattern2 in synonym_patterns:
            if ((re.search(pattern1, t1_lower) and re.search(pattern2, t2_lower)) or
                (re.search(pattern2, t1_lower) and re.search(pattern1, t2_lower))):
                return True
        
        return False
    
    def extract_hierarchy_from_toc(self, book_data: Dict, classification: str) -> Dict:
        """Extract hierarchical structure from a book's TOC."""
        book_title = book_data.get("book_title", "")
        toc_entries = book_data.get("toc_entries", [])
        
        if not toc_entries:
            return {}
        
        # Classify domain
        domain = self.classify_domain(book_title, toc_entries)
        
        # Build hierarchy from TOC entries
        hierarchy = {}
        current_path = []
        
        for entry in toc_entries:
            title = self.normalize_title(entry.get("title", ""))
            level = entry.get("level", 1)
            
            if not title:
                continue
            
            # Adjust path based on level
            if level <= len(current_path):
                current_path = current_path[:level-1]
            
            current_path.append(title)
            
            # Place in hierarchy (up to 6 levels)
            self._place_in_hierarchy(hierarchy, current_path[:6], entry, domain)
        
        return {domain: hierarchy} if hierarchy else {}
    
    def _place_in_hierarchy(self, hierarchy: Dict, path: List[str], 
                           original_entry: Dict, domain: str) -> None:
        """Place an entry in the hierarchy at the specified path."""
        current = hierarchy
        
        for i, part in enumerate(path):
            if i == 5:  # Level 6 - should be a list
                if part not in current:
                    current[part] = []
                if isinstance(current[part], list):
                    current[part].append({
                        "title": part,
                        "original_entry": original_entry,
                        "domain": domain
                    })
                else:
                    # Convert to list if it's not already
                    current[part] = [current[part], {
                        "title": part,
                        "original_entry": original_entry,
                        "domain": domain
                    }]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
    
    def merge_hierarchies(self, hierarchies: List[Dict], classification: str) -> Dict:
        """Merge multiple hierarchies with synonym normalization."""
        self.logger.start_timer(f"merge_hierarchies_{classification}")
        
        if not hierarchies:
            return {}
        
        # Collect all titles at each level for synonym analysis
        level_titles = [set() for _ in range(6)]
        
        def collect_titles(obj, level=0):
            if level >= 6:
                return
            if isinstance(obj, dict):
                for key, value in obj.items():
                    level_titles[level].add(key)
                    collect_titles(value, level + 1)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "title" in item:
                        level_titles[level].add(item["title"])
        
        for hierarchy in hierarchies:
            collect_titles(hierarchy)
        
        # Get synonym mappings for each level
        level_mappings = []
        for level, titles in enumerate(level_titles):
            if titles:
                context = f"Physics curriculum level {level + 1} ({self.level_names[level]})"
                mappings = self.llm_normalize_synonyms(list(titles), context)
                level_mappings.append(mappings)
            else:
                level_mappings.append({})
        
        # Merge hierarchies using synonym mappings
        merged = {}
        
        for hierarchy in hierarchies:
            self._merge_single_hierarchy(merged, hierarchy, level_mappings)
        
        self.logger.end_timer(f"merge_hierarchies_{classification}")
        return merged
    
    def _merge_single_hierarchy(self, target: Dict, source: Dict, 
                               mappings: List[Dict], level: int = 0) -> None:
        """Merge a single hierarchy into the target using synonym mappings."""
        if level >= 6:
            return
            
        if isinstance(source, dict):
            for key, value in source.items():
                # Get canonical name
                canonical_key = mappings[level].get(key, key) if level < len(mappings) else key
                
                if level == 5:  # Level 6 - lists
                    if canonical_key not in target:
                        target[canonical_key] = []
                    
                    if isinstance(value, list):
                        target[canonical_key].extend(value)
                    else:
                        target[canonical_key].append(value)
                else:
                    if canonical_key not in target:
                        target[canonical_key] = {}
                    
                    self._merge_single_hierarchy(
                        target[canonical_key], value, mappings, level + 1
                    )
    
    def inject_foundational_content(self, hierarchy: Dict) -> Dict:
        """Inject foundational content into the hierarchy."""
        if not self.config.include_foundational_content:
            return hierarchy
        
        self.logger.start_timer("inject_foundational")
        
        # Add foundational content to core curriculum
        if "core" not in hierarchy:
            hierarchy["core"] = {}
        
        # Merge foundational content
        for domain, content in self.foundational_content.items():
            if domain not in hierarchy["core"]:
                hierarchy["core"][domain] = content
            else:
                # Merge with existing content
                self._deep_merge_dict(hierarchy["core"][domain], content)
        
        self.logger.end_timer("inject_foundational")
        return hierarchy
    
    def _deep_merge_dict(self, target: Dict, source: Dict) -> None:
        """Deep merge dictionaries, handling list values at level 6."""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge_dict(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    # Merge lists, avoiding duplicates
                    for item in value:
                        if item not in target[key]:
                            target[key].append(item)
            else:
                target[key] = value
    
    def validate_hierarchy(self, hierarchy: Dict) -> Tuple[bool, List[str]]:
        """Validate that the hierarchy follows the six-level structure."""
        return DataValidator.validate_six_level_hierarchy(hierarchy)
    
    def enforce_six_levels(self, hierarchy: Dict) -> Dict:
        """Ensure all paths in the hierarchy have exactly 6 levels."""
        self.logger.start_timer("enforce_six_levels")
        
        def enforce_levels(obj, current_level=1, path=""):
            if current_level > 6:
                return obj
            
            if current_level == 6:
                # Level 6 must be a list
                if not isinstance(obj, list):
                    if isinstance(obj, dict):
                        # Convert dict to list of learning elements
                        elements = []
                        for key, value in obj.items():
                            if isinstance(value, list):
                                elements.extend(value)
                            else:
                                elements.append(str(value))
                        return elements if elements else [f"Learning activities for {path}"]
                    else:
                        return [str(obj)] if obj else [f"Learning activities for {path}"]
                return obj
            
            if not isinstance(obj, dict):
                # Need to expand to reach 6 levels
                remaining_levels = 6 - current_level
                result = obj
                
                for i in range(remaining_levels - 1):
                    result = {"General Topic": result}
                
                # Final level should be a list
                if remaining_levels > 0:
                    result = {"General Topic": [str(obj)] if obj else ["Learning activities"]}
                
                return result
            
            # Process each key-value pair
            result = {}
            for key, value in obj.items():
                new_path = f"{path}/{key}" if path else key
                result[key] = enforce_levels(value, current_level + 1, new_path)
            
            return result
        
        enforced_hierarchy = {}
        for classification, content in hierarchy.items():
            enforced_hierarchy[classification] = enforce_levels(content, 1, classification)
        
        self.logger.end_timer("enforce_six_levels")
        return enforced_hierarchy
    
    def process_academic_levels(self, classified_data: Dict) -> Dict:
        """Process content across academic levels (HS → UG → Grad)."""
        self.logger.start_timer("process_academic_levels")
        
        result = {"core": {}, "electives": {}}
        
        # Process core and elective classifications separately
        for classification in ["core", "electives"]:
            if classification not in classified_data:
                continue
            
            # Collect hierarchies from all academic levels
            all_hierarchies = []
            
            classification_data = classified_data[classification]
            
            if isinstance(classification_data, dict):
                # Process by academic level
                for level in self.config.academic_level_priority:
                    if level in classification_data:
                        level_books = classification_data[level]
                        
                        if isinstance(level_books, list):
                            for book in level_books:
                                hierarchy = self.extract_hierarchy_from_toc(book, classification)
                                if hierarchy:
                                    all_hierarchies.append(hierarchy)
                        elif isinstance(level_books, dict):
                            # Direct domain organization
                            all_hierarchies.append(level_books)
                
                # Also handle direct domain organization
                for key, value in classification_data.items():
                    if key not in self.config.academic_level_priority:
                        if isinstance(value, list):
                            for book in value:
                                hierarchy = self.extract_hierarchy_from_toc(book, classification)
                                if hierarchy:
                                    all_hierarchies.append(hierarchy)
            
            # Merge all hierarchies for this classification
            if all_hierarchies:
                result[classification] = self.merge_hierarchies(all_hierarchies, classification)
        
        self.logger.end_timer("process_academic_levels")
        return result
    
    def create_output(self, hierarchy: Dict) -> Dict:
        """Create structured output with metadata."""
        # Validate hierarchy
        is_valid, errors = self.validate_hierarchy(hierarchy)
        
        if not is_valid:
            self.logger.warning(f"Hierarchy validation issues: {errors}")
            # Try to fix by enforcing six levels
            hierarchy = self.enforce_six_levels(hierarchy)
            is_valid, errors = self.validate_hierarchy(hierarchy)
        
        # Inject foundational content
        hierarchy = self.inject_foundational_content(hierarchy)
        
        # Final enforcement of six levels
        hierarchy = self.enforce_six_levels(hierarchy)
        
        # Calculate statistics
        stats = self._calculate_hierarchy_stats(hierarchy)
        
        output = {
            "metadata": {
                "hierarchy_valid": is_valid,
                "validation_errors": errors,
                "statistics": stats,
                "level_names": self.level_names,
                "foundational_content_included": self.config.include_foundational_content,
                "timestamp": self.logger.logger.handlers[0].formatter.formatTime(
                    self.logger.logger.makeRecord("", 0, "", 0, "", (), None)
                )
            },
            "hierarchy": hierarchy
        }
        
        return output
    
    def _calculate_hierarchy_stats(self, hierarchy: Dict) -> Dict:
        """Calculate statistics about the hierarchy."""
        stats = {
            "total_domains": 0,
            "total_categories": 0,
            "total_concepts": 0,
            "total_topics": 0,
            "total_subtopics": 0,
            "total_learning_elements": 0,
            "core_domains": 0,
            "elective_domains": 0
        }
        
        def count_levels(obj, level=1):
            if level > 6:
                return
            
            if level == 1:  # Domains
                if isinstance(obj, dict):
                    stats["total_domains"] += len(obj)
                    if "core" in hierarchy and obj is hierarchy["core"]:
                        stats["core_domains"] = len(obj)
                    elif "electives" in hierarchy and obj is hierarchy["electives"]:
                        stats["elective_domains"] = len(obj)
            elif level == 2:  # Categories
                if isinstance(obj, dict):
                    stats["total_categories"] += len(obj)
            elif level == 3:  # Concepts
                if isinstance(obj, dict):
                    stats["total_concepts"] += len(obj)
            elif level == 4:  # Topics
                if isinstance(obj, dict):
                    stats["total_topics"] += len(obj)
            elif level == 5:  # Subtopics
                if isinstance(obj, dict):
                    stats["total_subtopics"] += len(obj)
            elif level == 6:  # Learning Elements
                if isinstance(obj, list):
                    stats["total_learning_elements"] += len(obj)
                return  # Stop recursion at level 6
            
            if isinstance(obj, dict):
                for value in obj.values():
                    count_levels(value, level + 1)
        
        if isinstance(hierarchy, dict):
            for section_name, section_content in hierarchy.items():
                count_levels(section_content, 1)
        
        return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Six-Level Hierarchy Builder")
    parser.add_argument("--input", "-i", default="classified_curriculum.json",
                       help="Input classified curriculum file")
    parser.add_argument("--output", "-o", default="six_level_hierarchy.json",
                       help="Output six-level hierarchy file")
    parser.add_argument("--config", "-c", default="config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--no-foundational", action="store_true",
                       help="Skip foundational content injection")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM synonym normalization")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    if args.no_foundational:
        config.include_foundational_content = False
    if args.no_llm:
        config.openai_api_key = ""
        import os
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    logger = CurriculumLogger("step4_hierarchy", "DEBUG" if args.verbose else "INFO")
    file_manager = FileManager(logger)
    
    logger.info("Starting Six-Level Hierarchy Builder")
    logger.info(f"Foundational content: {'enabled' if config.include_foundational_content else 'disabled'}")
    logger.info(f"LLM normalization: {'enabled' if config.openai_api_key else 'disabled'}")
    
    # Load input data
    logger.start_timer("data_loading")
    classified_data = file_manager.load_json(args.input)
    if not classified_data:
        logger.error(f"Failed to load input file: {args.input}")
        return 1
    
    logger.end_timer("data_loading")
    
    # Build hierarchy
    builder = SixLevelHierarchyBuilder(config, logger)
    
    try:
        # Process academic levels and build hierarchy
        hierarchy = builder.process_academic_levels(classified_data)
        
        # Create output with metadata
        output_data = builder.create_output(hierarchy)
        
        # Save results
        logger.start_timer("output_saving")
        if file_manager.save_json(output_data, args.output):
            logger.info(f"Six-level hierarchy saved to: {args.output}")
        else:
            logger.error("Failed to save output")
            return 1
        logger.end_timer("output_saving")
        
        # Performance summary
        logger.log_performance_summary()
        
        # Summary statistics
        stats = output_data["metadata"]["statistics"]
        logger.info("Hierarchy Statistics:")
        logger.info(f"  Total domains: {stats['total_domains']}")
        logger.info(f"  Total categories: {stats['total_categories']}")
        logger.info(f"  Total concepts: {stats['total_concepts']}")
        logger.info(f"  Total topics: {stats['total_topics']}")
        logger.info(f"  Total subtopics: {stats['total_subtopics']}")
        logger.info(f"  Total learning elements: {stats['total_learning_elements']}")
        logger.info(f"  Hierarchy valid: {output_data['metadata']['hierarchy_valid']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Hierarchy building failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())