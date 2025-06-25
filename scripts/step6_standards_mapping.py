#!/usr/bin/env python3
"""
Step 6: Comprehensive Standards Mapping

This module maps curriculum content to educational standards including MCAT, IB Physics,
A-Level Physics, IGCSE Physics, ABET outcomes, ISO standards, and UNESCO competencies.

Features:
- Auto-discovery and caching of standards documents
- Comprehensive mapping to multiple standards frameworks
- Bloom's taxonomy classification
- Difficulty level assessment
- Application domain annotations
- Confidence scoring for mappings
"""

import sys
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import argparse
from urllib.parse import urljoin, urlparse
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.curriculum_utils import (
    CurriculumConfig, CurriculumLogger, LLMClient, FileManager, 
    DataValidator, load_config, CacheManager
)


@dataclass
class StandardsMapping:
    """Represents a mapping to an educational standard."""
    topic_id: str
    topic_title: str
    standard_type: str  # MCAT, IB_HL, A_Level, etc.
    standard_code: str  # e.g., "C/P", "HL.2.3"
    standard_description: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    bloom_level: str  # Remember, Understand, Apply, Analyze, Evaluate, Create
    difficulty_level: str  # Introductory, Intermediate, Advanced
    application_domains: List[str]  # Engineering, Medicine, Research, etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)


class StandardsRepository:
    """Repository for managing educational standards documents and mappings."""
    
    def __init__(self, cache_manager: CacheManager, logger: CurriculumLogger):
        self.cache_manager = cache_manager
        self.logger = logger
        
        # Standards definitions
        self.standards_definitions = self._define_standards_frameworks()
        
        # Content patterns for mapping
        self.content_patterns = self._define_content_patterns()
    
    def _define_standards_frameworks(self) -> Dict[str, Dict]:
        """Define the structure and content of various standards frameworks."""
        return {
            "MCAT": {
                "full_name": "Medical College Admission Test",
                "sections": {
                    "C/P": {
                        "name": "Chemical and Physical Foundations of Biological Systems",
                        "physics_topics": [
                            "translational motion", "force", "work", "energy", "periodic motion",
                            "sound", "fluids", "electrostatics", "circuits", "magnetism",
                            "electromagnetic radiation", "atomic nucleus", "electronic structure"
                        ]
                    },
                    "CARS": {
                        "name": "Critical Analysis and Reasoning Skills",
                        "physics_topics": []  # No physics content
                    },
                    "B/B": {
                        "name": "Biological and Biochemical Foundations of Living Systems",
                        "physics_topics": [
                            "energy transformations", "thermodynamics in biological systems"
                        ]
                    },
                    "P/S": {
                        "name": "Psychological, Social, and Biological Foundations of Behavior",
                        "physics_topics": [
                            "sensory processing", "vision", "hearing", "electromagnetic spectrum"
                        ]
                    }
                }
            },
            "IB_HL": {
                "full_name": "International Baccalaureate Physics Higher Level",
                "topics": {
                    "1": "Measurements and uncertainties",
                    "2": "Mechanics", 
                    "3": "Thermal physics",
                    "4": "Waves",
                    "5": "Electricity and magnetism",
                    "6": "Circular motion and gravitation",
                    "7": "Atomic, nuclear and particle physics",
                    "8": "Energy production",
                    "9": "Wave phenomena (HL)",
                    "10": "Fields (HL)",
                    "11": "Electromagnetic induction (HL)",
                    "12": "Quantum and nuclear physics (HL)"
                }
            },
            "IB_SL": {
                "full_name": "International Baccalaureate Physics Standard Level",
                "topics": {
                    "1": "Measurements and uncertainties",
                    "2": "Mechanics",
                    "3": "Thermal physics", 
                    "4": "Waves",
                    "5": "Electricity and magnetism",
                    "6": "Circular motion and gravitation",
                    "7": "Atomic, nuclear and particle physics",
                    "8": "Energy production"
                }
            },
            "A_Level": {
                "full_name": "A-Level Physics (UK)",
                "modules": {
                    "AS1": "Forces, energy and momentum",
                    "AS2": "Electrons, waves and photons", 
                    "A21": "Further mechanics and thermal physics",
                    "A22": "Fields and their consequences",
                    "A23": "Nuclear physics"
                }
            },
            "IGCSE": {
                "full_name": "International General Certificate of Secondary Education Physics",
                "topics": {
                    "1": "General physics",
                    "2": "Thermal physics",
                    "3": "Properties of waves",
                    "4": "Electricity and magnetism", 
                    "5": "Atomic physics"
                }
            },
            "ABET": {
                "full_name": "Accreditation Board for Engineering and Technology",
                "outcomes": {
                    "1": "Engineering knowledge",
                    "2": "Problem analysis", 
                    "3": "Design/development of solutions",
                    "4": "Investigation",
                    "5": "Modern tool usage",
                    "6": "The engineer and society",
                    "7": "Environment and sustainability"
                }
            },
            "ISO_21001": {
                "full_name": "ISO 21001 Educational Organizations Management Systems",
                "principles": {
                    "learner_focus": "Focus on learners and other beneficiaries",
                    "visionary_leadership": "Visionary leadership",
                    "engagement": "Engagement of people",
                    "process_approach": "Process approach",
                    "improvement": "Improvement",
                    "evidence_decisions": "Evidence-based decision making",
                    "relationship_management": "Relationship management"
                }
            },
            "UNESCO": {
                "full_name": "UNESCO Education Framework",
                "competencies": {
                    "scientific_literacy": "Scientific literacy and inquiry",
                    "critical_thinking": "Critical thinking and problem solving",
                    "communication": "Communication and collaboration",
                    "creativity": "Creativity and innovation",
                    "global_citizenship": "Global citizenship and sustainability"
                }
            }
        }
    
    def _define_content_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Define patterns for matching content to standards."""
        return {
            "MCAT_C/P": {
                "mechanics": ["motion", "force", "energy", "momentum", "rotation", "oscillation"],
                "thermodynamics": ["heat", "temperature", "entropy", "gas laws"],
                "electricity": ["electric field", "potential", "current", "resistance", "circuit"],
                "magnetism": ["magnetic field", "induction", "electromagnetic"],
                "waves": ["wave", "sound", "electromagnetic radiation", "optics"],
                "atomic": ["atom", "nucleus", "electron", "quantum", "photoelectric"]
            },
            "IB_HL": {
                "measurements": ["uncertainty", "error", "significant figures", "precision"],
                "mechanics": ["kinematics", "dynamics", "energy", "momentum", "circular motion"],
                "thermal": ["heat", "temperature", "thermal expansion", "specific heat"],
                "waves": ["wave properties", "sound", "electromagnetic spectrum"],
                "electricity": ["current", "resistance", "circuits", "magnetic field"],
                "fields": ["gravitational field", "electric field", "magnetic field", "potential"]
            },
            "A_Level": {
                "forces": ["force", "newton", "equilibrium", "friction"],
                "energy": ["kinetic energy", "potential energy", "conservation"],
                "momentum": ["momentum", "collision", "impulse"],
                "electrons": ["electron", "photoelectric", "electronic"],
                "waves": ["wave", "interference", "diffraction", "polarization"],
                "fields": ["field", "potential", "flux"]
            },
            "IGCSE": {
                "general": ["measurement", "motion", "mass", "weight", "density"],
                "thermal": ["heat", "temperature", "expansion", "change of state"],
                "waves": ["wave", "sound", "light", "electromagnetic spectrum"],
                "electricity": ["current", "voltage", "resistance", "power"],
                "magnetism": ["magnet", "magnetic field", "electromagnet"]
            }
        }
    
    def get_standards_document(self, standard_type: str) -> Optional[Dict]:
        """Retrieve or create standards document."""
        # First check cache
        cached = self.cache_manager.get_standards_document(standard_type)
        if cached:
            return cached
        
        # Use built-in definitions
        if standard_type in self.standards_definitions:
            document = self.standards_definitions[standard_type]
            
            # Cache the document
            self.cache_manager.store_standards_document(standard_type, document)
            return document
        
        # Could add web scraping here for live standards
        # For now, return None for unknown standards
        return None


class StandardsMapper:
    """Main class for mapping curriculum content to educational standards."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.file_manager = FileManager(logger)
        self.cache_manager = CacheManager(config.cache_directory, logger)
        
        self.standards_repo = StandardsRepository(self.cache_manager, logger)
        
        # Bloom's taxonomy levels
        self.bloom_levels = [
            "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
        ]
        
        # Difficulty levels
        self.difficulty_levels = ["Introductory", "Intermediate", "Advanced"]
        
        # Application domains
        self.application_domains = [
            "Engineering", "Medicine", "Research", "Technology", "Education",
            "Industry", "Environmental", "Space", "Energy", "Communication"
        ]
    
    def classify_bloom_level(self, topic_title: str, learning_elements: List[str]) -> str:
        """Classify the Bloom's taxonomy level of a topic."""
        # Keyword-based classification
        bloom_keywords = {
            "Remember": ["define", "list", "identify", "recall", "recognize", "name"],
            "Understand": ["explain", "describe", "interpret", "summarize", "compare", "classify"],
            "Apply": ["calculate", "solve", "demonstrate", "use", "implement", "apply"],
            "Analyze": ["analyze", "examine", "investigate", "distinguish", "differentiate"],
            "Evaluate": ["evaluate", "assess", "critique", "judge", "argue", "defend"],
            "Create": ["design", "create", "develop", "construct", "formulate", "synthesize"]
        }
        
        # Check learning elements for keywords
        all_text = (topic_title + " " + " ".join(learning_elements)).lower()
        
        level_scores = {}
        for level, keywords in bloom_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            level_scores[level] = score
        
        # Return highest scoring level, default to "Understand"
        if level_scores and max(level_scores.values()) > 0:
            return max(level_scores.items(), key=lambda x: x[1])[0]
        
        # Default classification based on common patterns
        if any(word in all_text for word in ["problem", "calculation", "solve"]):
            return "Apply"
        elif any(word in all_text for word in ["definition", "concept", "theory"]):
            return "Understand"
        else:
            return "Understand"
    
    def assess_difficulty_level(self, topic_path: List[str], prerequisites_count: int) -> str:
        """Assess the difficulty level of a topic."""
        # Factors for difficulty assessment
        level = len(topic_path)  # Depth in hierarchy
        
        # Topic-based difficulty indicators
        topic_text = " ".join(topic_path).lower()
        
        advanced_indicators = [
            "quantum", "relativity", "nuclear", "particle", "advanced",
            "theoretical", "graduate", "research", "complex"
        ]
        
        introductory_indicators = [
            "introduction", "basic", "fundamental", "elementary",
            "overview", "concepts", "simple"
        ]
        
        # Count indicators
        advanced_score = sum(1 for indicator in advanced_indicators if indicator in topic_text)
        intro_score = sum(1 for indicator in introductory_indicators if indicator in topic_text)
        
        # Classification logic
        if advanced_score > 0 or prerequisites_count > 5 or level > 4:
            return "Advanced"
        elif intro_score > 0 or prerequisites_count == 0 or level <= 2:
            return "Introductory"
        else:
            return "Intermediate"
    
    def identify_application_domains(self, topic_title: str, topic_path: List[str]) -> List[str]:
        """Identify application domains for a topic."""
        domain_keywords = {
            "Engineering": ["force", "stress", "strain", "material", "design", "structure"],
            "Medicine": ["radiation", "imaging", "ultrasound", "nuclear medicine", "therapy"],
            "Research": ["quantum", "particle", "theoretical", "experimental", "advanced"],
            "Technology": ["electronics", "circuit", "device", "sensor", "communication"],
            "Environmental": ["climate", "atmosphere", "earth", "environmental", "sustainability"],
            "Space": ["astronomy", "astrophysics", "cosmology", "satellite", "space"],
            "Energy": ["power", "energy", "renewable", "nuclear", "thermal", "solar"],
            "Communication": ["wave", "electromagnetic", "fiber optic", "signal", "transmission"]
        }
        
        all_text = (topic_title + " " + " ".join(topic_path)).lower()
        
        applicable_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                applicable_domains.append(domain)
        
        # Default domains if none found
        if not applicable_domains:
            applicable_domains = ["Education"]
        
        return applicable_domains
    
    def map_to_mcat(self, topic_title: str, topic_path: List[str], 
                   learning_elements: List[str]) -> List[StandardsMapping]:
        """Map topic to MCAT standards."""
        mappings = []
        mcat_doc = self.standards_repo.get_standards_document("MCAT")
        
        if not mcat_doc:
            return mappings
        
        topic_text = (topic_title + " " + " ".join(topic_path) + " " + " ".join(learning_elements)).lower()
        
        # Check each MCAT section
        for section_code, section_info in mcat_doc["sections"].items():
            if not section_info.get("physics_topics"):
                continue
            
            # Calculate relevance score
            relevance_score = 0
            matched_topics = []
            
            for physics_topic in section_info["physics_topics"]:
                if physics_topic.lower() in topic_text:
                    relevance_score += 1
                    matched_topics.append(physics_topic)
            
            if relevance_score > 0:
                confidence = min(0.9, relevance_score / 3.0)  # Normalize to max 0.9
                
                mappings.append(StandardsMapping(
                    topic_id=" → ".join(topic_path),
                    topic_title=topic_title,
                    standard_type="MCAT",
                    standard_code=section_code,
                    standard_description=section_info["name"],
                    confidence=confidence,
                    reasoning=f"Matches MCAT topics: {', '.join(matched_topics)}",
                    bloom_level=self.classify_bloom_level(topic_title, learning_elements),
                    difficulty_level=self.assess_difficulty_level(topic_path, 0),
                    application_domains=["Medicine"]
                ))
        
        return mappings
    
    def map_to_ib(self, topic_title: str, topic_path: List[str], 
                 learning_elements: List[str], level: str = "HL") -> List[StandardsMapping]:
        """Map topic to IB Physics standards."""
        mappings = []
        ib_doc = self.standards_repo.get_standards_document(f"IB_{level}")
        
        if not ib_doc:
            return mappings
        
        topic_text = (topic_title + " " + " ".join(topic_path)).lower()
        
        # Check IB topics
        for topic_num, topic_desc in ib_doc["topics"].items():
            topic_desc_lower = topic_desc.lower()
            
            # Simple keyword matching
            relevance_score = 0
            if any(word in topic_text for word in topic_desc_lower.split()):
                relevance_score = 1
            
            # More sophisticated matching
            if "mechanic" in topic_text and "mechanic" in topic_desc_lower:
                relevance_score = 2
            elif "wave" in topic_text and "wave" in topic_desc_lower:
                relevance_score = 2
            elif "electric" in topic_text and "electric" in topic_desc_lower:
                relevance_score = 2
            elif "atomic" in topic_text and "atomic" in topic_desc_lower:
                relevance_score = 2
            
            if relevance_score > 0:
                confidence = min(0.8, relevance_score / 2.0)
                
                mappings.append(StandardsMapping(
                    topic_id=" → ".join(topic_path),
                    topic_title=topic_title,
                    standard_type=f"IB_{level}",
                    standard_code=f"{level}.{topic_num}",
                    standard_description=topic_desc,
                    confidence=confidence,
                    reasoning=f"Maps to IB {level} topic {topic_num}: {topic_desc}",
                    bloom_level=self.classify_bloom_level(topic_title, learning_elements),
                    difficulty_level="Intermediate" if level == "SL" else "Advanced",
                    application_domains=self.identify_application_domains(topic_title, topic_path)
                ))
        
        return mappings
    
    def map_to_a_level(self, topic_title: str, topic_path: List[str], 
                      learning_elements: List[str]) -> List[StandardsMapping]:
        """Map topic to A-Level Physics standards."""
        mappings = []
        a_level_doc = self.standards_repo.get_standards_document("A_Level")
        
        if not a_level_doc:
            return mappings
        
        topic_text = (topic_title + " " + " ".join(topic_path)).lower()
        
        # Check A-Level modules
        for module_code, module_desc in a_level_doc["modules"].items():
            module_desc_lower = module_desc.lower()
            
            relevance_score = 0
            matched_keywords = []
            
            # Keyword matching
            for word in module_desc_lower.split():
                if len(word) > 3 and word in topic_text:
                    relevance_score += 1
                    matched_keywords.append(word)
            
            if relevance_score > 0:
                confidence = min(0.8, relevance_score / 3.0)
                
                mappings.append(StandardsMapping(
                    topic_id=" → ".join(topic_path),
                    topic_title=topic_title,
                    standard_type="A_Level",
                    standard_code=module_code,
                    standard_description=module_desc,
                    confidence=confidence,
                    reasoning=f"Matches A-Level keywords: {', '.join(matched_keywords)}",
                    bloom_level=self.classify_bloom_level(topic_title, learning_elements),
                    difficulty_level="Intermediate",
                    application_domains=self.identify_application_domains(topic_title, topic_path)
                ))
        
        return mappings
    
    def map_to_igcse(self, topic_title: str, topic_path: List[str], 
                    learning_elements: List[str]) -> List[StandardsMapping]:
        """Map topic to IGCSE Physics standards."""
        mappings = []
        igcse_doc = self.standards_repo.get_standards_document("IGCSE")
        
        if not igcse_doc:
            return mappings
        
        topic_text = (topic_title + " " + " ".join(topic_path)).lower()
        
        # Check IGCSE topics
        for topic_num, topic_desc in igcse_doc["topics"].items():
            topic_desc_lower = topic_desc.lower()
            
            relevance_score = 0
            if any(word in topic_text for word in topic_desc_lower.split() if len(word) > 3):
                relevance_score = 1
            
            if relevance_score > 0:
                confidence = 0.7  # IGCSE is broad, so moderate confidence
                
                mappings.append(StandardsMapping(
                    topic_id=" → ".join(topic_path),
                    topic_title=topic_title,
                    standard_type="IGCSE",
                    standard_code=f"IGCSE.{topic_num}",
                    standard_description=topic_desc,
                    confidence=confidence,
                    reasoning=f"Maps to IGCSE topic {topic_num}: {topic_desc}",
                    bloom_level=self.classify_bloom_level(topic_title, learning_elements),
                    difficulty_level="Introductory",
                    application_domains=["Education"]
                ))
        
        return mappings
    
    def llm_enhanced_mapping(self, topic_title: str, topic_path: List[str], 
                           learning_elements: List[str]) -> List[StandardsMapping]:
        """Use LLM to enhance standards mapping."""
        if not self.llm_client.is_available():
            return []
        
        prompt = f"""
        You are an expert in physics education standards. Analyze this physics topic and map it to relevant educational standards.

        Topic: {topic_title}
        Path: {' → '.join(topic_path)}
        Learning Elements: {', '.join(learning_elements[:5])}  # First 5 elements

        Map this topic to appropriate standards from:
        1. MCAT (sections: C/P, B/B, P/S)
        2. IB Physics HL/SL
        3. A-Level Physics 
        4. IGCSE Physics

        For each mapping, consider:
        - Relevance and appropriateness
        - Bloom's taxonomy level (Remember, Understand, Apply, Analyze, Evaluate, Create)
        - Difficulty level (Introductory, Intermediate, Advanced)
        - Application domains

        Respond with JSON format:
        {{
            "mappings": [
                {{
                    "standard_type": "MCAT",
                    "standard_code": "C/P",
                    "confidence": 0.8,
                    "reasoning": "Brief explanation",
                    "bloom_level": "Apply",
                    "difficulty_level": "Intermediate",
                    "application_domains": ["Medicine", "Research"]
                }}
            ]
        }}
        """
        
        cache_key = f"standards_{hash(topic_title)}_{hash(str(topic_path))}"
        
        response = self.llm_client.generate_completion(
            prompt=prompt,
            cache_key=cache_key,
            temperature=0.1
        )
        
        mappings = []
        
        if response:
            try:
                result = json.loads(response)
                llm_mappings = result.get("mappings", [])
                
                for mapping_data in llm_mappings:
                    mappings.append(StandardsMapping(
                        topic_id=" → ".join(topic_path),
                        topic_title=topic_title,
                        standard_type=mapping_data.get("standard_type", ""),
                        standard_code=mapping_data.get("standard_code", ""),
                        standard_description=mapping_data.get("standard_description", ""),
                        confidence=float(mapping_data.get("confidence", 0.5)),
                        reasoning=mapping_data.get("reasoning", "LLM-identified mapping"),
                        bloom_level=mapping_data.get("bloom_level", "Understand"),
                        difficulty_level=mapping_data.get("difficulty_level", "Intermediate"),
                        application_domains=mapping_data.get("application_domains", ["Education"])
                    ))
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                self.logger.warning(f"Failed to parse LLM standards response: {e}")
        
        return mappings
    
    def process_topic(self, topic_id: str, topic_data: Dict, 
                     prerequisites_count: int = 0) -> List[StandardsMapping]:
        """Process a single topic and generate all standards mappings."""
        title = topic_data.get("title", "")
        path = topic_data.get("path", [])
        
        # Extract learning elements (assume they're at the deepest level)
        learning_elements = []
        if isinstance(topic_data.get("learning_elements"), list):
            learning_elements = topic_data["learning_elements"]
        
        all_mappings = []
        
        # Apply all mapping methods
        all_mappings.extend(self.map_to_mcat(title, path, learning_elements))
        all_mappings.extend(self.map_to_ib(title, path, learning_elements, "HL"))
        all_mappings.extend(self.map_to_ib(title, path, learning_elements, "SL"))
        all_mappings.extend(self.map_to_a_level(title, path, learning_elements))
        all_mappings.extend(self.map_to_igcse(title, path, learning_elements))
        
        # LLM enhancement for high-value topics
        if self.llm_client.is_available() and len(path) >= 3:
            llm_mappings = self.llm_enhanced_mapping(title, path, learning_elements)
            all_mappings.extend(llm_mappings)
        
        return all_mappings
    
    def process_curriculum(self, curriculum_data: Dict) -> Dict[str, Any]:
        """Process entire curriculum and generate standards mappings."""
        self.logger.start_timer("standards_mapping")
        
        topics = curriculum_data.get("topics", {})
        prerequisites = curriculum_data.get("prerequisites", [])
        
        # Build prerequisites count lookup
        prereq_counts = defaultdict(int)
        for prereq in prerequisites:
            prereq_counts[prereq["target"]] += 1
        
        all_mappings = []
        processed_count = 0
        
        # Process topics (sample for large datasets)
        topic_items = list(topics.items())
        if len(topic_items) > 500:  # Sample for performance
            import random
            topic_items = random.sample(topic_items, 500)
            self.logger.info(f"Sampling {len(topic_items)} topics for standards mapping")
        
        for topic_id, topic_data in topic_items:
            prereq_count = prereq_counts.get(topic_id, 0)
            topic_mappings = self.process_topic(topic_id, topic_data, prereq_count)
            all_mappings.extend(topic_mappings)
            
            processed_count += 1
            if processed_count % 100 == 0:
                self.logger.info(f"Processed {processed_count} topics...")
        
        self.logger.info(f"Generated {len(all_mappings)} standards mappings")
        self.logger.end_timer("standards_mapping")
        
        return {
            "mappings": all_mappings,
            "processed_topics": processed_count,
            "total_mappings": len(all_mappings)
        }
    
    def create_output(self, processing_results: Dict, input_data: Dict) -> Dict[str, Any]:
        """Create structured output with standards mappings."""
        mappings = processing_results["mappings"]
        
        # Organize mappings by standard type
        mappings_by_standard = defaultdict(list)
        for mapping in mappings:
            mappings_by_standard[mapping.standard_type].append(mapping.to_dict())
        
        # Calculate statistics
        stats = {
            "total_mappings": len(mappings),
            "standards_covered": len(mappings_by_standard),
            "mappings_by_standard": {k: len(v) for k, v in mappings_by_standard.items()},
            "avg_confidence": sum(m.confidence for m in mappings) / len(mappings) if mappings else 0,
            "bloom_distribution": {},
            "difficulty_distribution": {},
            "application_domains": set()
        }
        
        # Analyze distributions
        bloom_counts = Counter(m.bloom_level for m in mappings)
        difficulty_counts = Counter(m.difficulty_level for m in mappings)
        
        for mapping in mappings:
            stats["application_domains"].update(mapping.application_domains)
        
        stats["bloom_distribution"] = dict(bloom_counts)
        stats["difficulty_distribution"] = dict(difficulty_counts)
        stats["application_domains"] = list(stats["application_domains"])
        
        # Preserve input data
        output = input_data.copy()
        
        # Add standards mapping results
        output["standards_mappings"] = dict(mappings_by_standard)
        output["metadata"]["standards_mapping"] = {
            "statistics": stats,
            "processed_topics": processing_results["processed_topics"],
            "supported_standards": list(mappings_by_standard.keys()),
            "timestamp": self.logger.logger.handlers[0].formatter.formatTime(
                self.logger.logger.makeRecord("", 0, "", 0, "", (), None)
            )
        }
        
        return output


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Comprehensive Standards Mapping")
    parser.add_argument("--input", "-i", default="curriculum_with_prerequisites.json",
                       help="Input curriculum with prerequisites file")
    parser.add_argument("--output", "-o", default="standards_mapped_curriculum.json",
                       help="Output standards-mapped curriculum file")
    parser.add_argument("--config", "-c", default="config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM-enhanced standards mapping")
    parser.add_argument("--standards", nargs="+", 
                       choices=["MCAT", "IB_HL", "IB_SL", "A_Level", "IGCSE", "ABET", "ISO", "UNESCO"],
                       help="Specific standards to map to")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    if args.no_llm:
        config.openai_api_key = ""
        import os
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
    
    if args.standards:
        config.supported_standards = args.standards
    
    logger = CurriculumLogger("step6_standards", "DEBUG" if args.verbose else "INFO")
    file_manager = FileManager(logger)
    
    logger.info("Starting Comprehensive Standards Mapping")
    logger.info(f"Supported standards: {config.supported_standards}")
    logger.info(f"LLM enhancement: {'enabled' if config.openai_api_key else 'disabled'}")
    
    # Load input data
    logger.start_timer("data_loading")
    curriculum_data = file_manager.load_json(args.input)
    if not curriculum_data:
        logger.error(f"Failed to load input file: {args.input}")
        return 1
    
    logger.end_timer("data_loading")
    
    # Process standards mapping
    mapper = StandardsMapper(config, logger)
    
    try:
        processing_results = mapper.process_curriculum(curriculum_data)
        output_data = mapper.create_output(processing_results, curriculum_data)
        
        # Save results
        logger.start_timer("output_saving")
        if file_manager.save_json(output_data, args.output):
            logger.info(f"Standards-mapped curriculum saved to: {args.output}")
        else:
            logger.error("Failed to save output")
            return 1
        logger.end_timer("output_saving")
        
        # Performance summary
        logger.log_performance_summary()
        
        # Summary statistics
        standards_metadata = output_data["metadata"]["standards_mapping"]
        stats = standards_metadata["statistics"]
        logger.info("Standards Mapping Summary:")
        logger.info(f"  Topics processed: {standards_metadata['processed_topics']}")
        logger.info(f"  Total mappings: {stats['total_mappings']}")
        logger.info(f"  Standards covered: {stats['standards_covered']}")
        logger.info(f"  Average confidence: {stats['avg_confidence']:.3f}")
        logger.info(f"  Application domains: {len(stats['application_domains'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Standards mapping failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())