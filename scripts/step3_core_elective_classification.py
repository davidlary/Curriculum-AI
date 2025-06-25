#!/usr/bin/env python3
"""
Step 3: Core/Elective Classification Engine

This module classifies textbooks and their content into core curriculum vs elective domains
using keyword-based classification, frequency analysis, and academic level exclusivity detection.

Features:
- Configurable keyword patterns for elective detection
- Frequency analysis with adjustable threshold (default 20%)
- Academic level exclusivity detection
- Automatic elective domain grouping
- LLM-enhanced classification validation
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
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
class ClassificationResult:
    """Result of core/elective classification."""
    book_id: str
    book_title: str
    classification: str  # "core" or "elective"
    confidence: float
    reasons: List[str]
    elective_domain: Optional[str] = None


class CoreElectiveClassifier:
    """Main classifier for separating core curriculum from elective content."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.file_manager = FileManager(logger)
        
        # Statistics for frequency analysis
        self.topic_frequencies = Counter()
        self.book_count = 0
        self.level_distributions = defaultdict(list)
        
        # Classification patterns
        self.elective_patterns = self._build_elective_patterns()
        self.elective_domains = self._define_elective_domains()
    
    def _build_elective_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for identifying elective content."""
        return {
            "astronomy": [
                "astronomy", "astrophysics", "cosmology", "stellar", "galactic",
                "planetary", "solar system", "universe", "telescope", "observatory"
            ],
            "biophysics": [
                "biophysics", "biomechanics", "biomedical", "biological physics",
                "molecular biology", "cellular", "protein", "dna", "genetics"
            ],
            "environmental": [
                "environmental", "climate", "atmospheric", "meteorology",
                "earth science", "geology", "oceanography", "sustainability"
            ],
            "nuclear": [
                "nuclear physics", "radioactivity", "radiation", "nuclear reactor",
                "fission", "fusion", "isotope", "radioactive decay"
            ],
            "particle": [
                "particle physics", "elementary particles", "quark", "lepton",
                "standard model", "particle accelerator", "collider", "cern"
            ],
            "quantum_advanced": [
                "quantum field theory", "quantum chromodynamics", "quantum electrodynamics",
                "many-body", "condensed matter", "superconductivity"
            ],
            "medical": [
                "medical physics", "radiology", "medical imaging", "radiation therapy",
                "nuclear medicine", "mri", "ct scan", "ultrasound"
            ],
            "geophysics": [
                "geophysics", "seismology", "plate tectonics", "earthquakes",
                "magnetic field", "gravity field", "geodynamics"
            ],
            "nanotechnology": [
                "nanotechnology", "nanoscale", "nanoparticles", "quantum dots",
                "molecular electronics", "nanostructures"
            ]
        }
    
    def _define_elective_domains(self) -> Dict[str, str]:
        """Map elective patterns to domain names."""
        return {
            "astronomy": "Astronomy and Astrophysics",
            "biophysics": "Biophysics and Medical Physics",
            "environmental": "Environmental and Earth Physics",
            "nuclear": "Nuclear and Particle Physics",
            "particle": "Nuclear and Particle Physics",
            "quantum_advanced": "Advanced Quantum Physics",
            "medical": "Biophysics and Medical Physics",
            "geophysics": "Environmental and Earth Physics",
            "nanotechnology": "Nanotechnology and Materials Physics"
        }
    
    def analyze_topic_frequencies(self, toc_data: Dict) -> None:
        """Analyze topic frequencies across all books."""
        self.logger.start_timer("frequency_analysis")
        
        self.topic_frequencies.clear()
        self.book_count = 0
        self.level_distributions.clear()
        
        # Extract topics from all books
        tocs = toc_data.get("tocs_by_level", toc_data.get("tocs", {}))
        
        for level, books in tocs.items():
            if isinstance(books, list):
                self.level_distributions[level] = books
                self.book_count += len(books)
                
                for book in books:
                    if "toc_entries" in book:
                        for entry in book["toc_entries"]:
                            topic = entry.get("title", "").lower().strip()
                            if topic:
                                self.topic_frequencies[topic] += 1
        
        self.logger.info(f"Analyzed {len(self.topic_frequencies)} unique topics across {self.book_count} books")
        self.logger.end_timer("frequency_analysis")
    
    def classify_by_keywords(self, book: Dict) -> Tuple[str, float, List[str], Optional[str]]:
        """Classify book using keyword patterns."""
        book_title = book.get("book_title", "").lower()
        
        # Check title keywords
        elective_matches = []
        for domain, patterns in self.elective_patterns.items():
            for pattern in patterns:
                if pattern in book_title:
                    elective_matches.append((domain, pattern))
        
        # Check TOC content
        toc_matches = []
        if "toc_entries" in book:
            toc_text = " ".join([entry.get("title", "").lower() 
                               for entry in book["toc_entries"]]).lower()
            
            for domain, patterns in self.elective_patterns.items():
                pattern_count = sum(1 for pattern in patterns if pattern in toc_text)
                if pattern_count > 0:
                    toc_matches.append((domain, pattern_count))
        
        # Determine classification
        if elective_matches or toc_matches:
            # Find dominant elective domain
            domain_scores = defaultdict(int)
            
            for domain, _ in elective_matches:
                domain_scores[domain] += 3  # Title matches weighted higher
            
            for domain, count in toc_matches:
                domain_scores[domain] += count
            
            if domain_scores:
                best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
                confidence = min(0.9, domain_scores[best_domain] / 10.0)
                
                reasons = []
                if elective_matches:
                    reasons.append(f"Title contains elective keywords: {[m[1] for m in elective_matches]}")
                if toc_matches:
                    reasons.append(f"TOC contains elective content: {dict(toc_matches)}")
                
                return "elective", confidence, reasons, self.elective_domains.get(best_domain, best_domain)
        
        # Default to core
        return "core", 0.8, ["No elective keywords found"], None
    
    def classify_by_frequency(self, book: Dict) -> Tuple[str, float, List[str]]:
        """Classify book based on topic frequency analysis."""
        if not self.topic_frequencies or self.book_count == 0:
            return "core", 0.5, ["Frequency analysis not available"]
        
        rare_topics = []
        total_topics = 0
        
        if "toc_entries" in book:
            for entry in book["toc_entries"]:
                topic = entry.get("title", "").lower().strip()
                if topic and topic in self.topic_frequencies:
                    total_topics += 1
                    frequency = self.topic_frequencies[topic] / self.book_count
                    
                    if frequency < self.config.elective_frequency_threshold:
                        rare_topics.append((topic, frequency))
        
        if total_topics == 0:
            return "core", 0.5, ["No topics found for frequency analysis"]
        
        rare_ratio = len(rare_topics) / total_topics
        
        if rare_ratio > 0.3:  # More than 30% rare topics
            confidence = min(0.8, rare_ratio)
            reasons = [f"Contains {len(rare_topics)} rare topics (appearing in <{self.config.elective_frequency_threshold*100:.0f}% of books)"]
            return "elective", confidence, reasons
        else:
            confidence = 0.7
            reasons = [f"Only {len(rare_topics)} rare topics out of {total_topics} total topics"]
            return "core", confidence, reasons
    
    def classify_by_academic_level(self, book: Dict, level: str) -> Tuple[str, float, List[str]]:
        """Classify based on academic level exclusivity."""
        # Graduate-only content is more likely to be elective
        if level == "graduate":
            # Check if content appears only at graduate level
            book_title = book.get("book_title", "").lower()
            
            # Look for graduate-specific indicators
            graduate_indicators = [
                "advanced", "graduate", "research", "quantum field",
                "many-body", "statistical mechanics", "theoretical"
            ]
            
            indicator_count = sum(1 for indicator in graduate_indicators 
                                if indicator in book_title)
            
            if indicator_count > 0:
                confidence = min(0.7, indicator_count / 3.0)
                reasons = [f"Graduate-level content with indicators: {[i for i in graduate_indicators if i in book_title]}"]
                return "elective", confidence, reasons
        
        return "core", 0.6, [f"Standard {level} level content"]
    
    def llm_enhanced_classification(self, book: Dict, initial_classification: str, 
                                  confidence: float) -> Tuple[str, float, List[str]]:
        """Use LLM to validate and refine classification."""
        if not self.llm_client.is_available() or confidence > 0.85:
            return initial_classification, confidence, ["LLM validation skipped"]
        
        book_title = book.get("book_title", "")
        toc_sample = []
        
        if "toc_entries" in book:
            # Get first 10 TOC entries as sample
            for entry in book["toc_entries"][:10]:
                toc_sample.append(entry.get("title", ""))
        
        prompt = f"""
        Analyze this physics textbook and determine if it should be classified as CORE curriculum or ELECTIVE content.

        Book Title: {book_title}
        
        Table of Contents Sample:
        {chr(10).join([f"- {title}" for title in toc_sample if title])}
        
        CORE curriculum includes fundamental physics topics that are essential for all physics students:
        - Mechanics, Thermodynamics, Electricity & Magnetism, Waves, Optics
        - Mathematical methods, Problem-solving, Units and measurements
        - Standard undergraduate and graduate physics courses
        
        ELECTIVE content includes specialized or advanced topics:
        - Astronomy, Astrophysics, Biophysics, Medical Physics
        - Environmental Physics, Geophysics, Nuclear/Particle Physics
        - Advanced quantum field theory, Nanotechnology
        - Highly specialized research topics
        
        Current classification: {initial_classification} (confidence: {confidence:.2f})
        
        Respond with:
        1. CLASSIFICATION: core or elective
        2. CONFIDENCE: 0.0 to 1.0
        3. REASONING: Brief explanation
        4. DOMAIN: If elective, specify the specialized domain
        
        Format your response as JSON:
        {{"classification": "core/elective", "confidence": 0.0, "reasoning": "...", "domain": "..."}}
        """
        
        cache_key = f"classification_{hash(book_title)}_{hash(str(toc_sample))}"
        
        response = self.llm_client.generate_completion(
            prompt=prompt,
            cache_key=cache_key,
            temperature=0.1
        )
        
        if response:
            try:
                result = json.loads(response)
                llm_classification = result.get("classification", initial_classification)
                llm_confidence = float(result.get("confidence", confidence))
                llm_reasoning = result.get("reasoning", "LLM analysis")
                
                # Combine with initial confidence
                final_confidence = (confidence + llm_confidence) / 2
                
                return llm_classification, final_confidence, [llm_reasoning]
                
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse LLM response: {e}")
        
        return initial_classification, confidence, ["LLM validation failed"]
    
    def classify_book(self, book: Dict, level: str) -> ClassificationResult:
        """Classify a single book using all methods."""
        book_id = book.get("book_id", book.get("file_path", "unknown"))
        book_title = book.get("book_title", "Untitled")
        
        # Apply classification methods
        keyword_result = self.classify_by_keywords(book)
        frequency_result = self.classify_by_frequency(book)
        level_result = self.classify_by_academic_level(book, level)
        
        # Combine results (weighted voting)
        classifications = [keyword_result[0], frequency_result[0], level_result[0]]
        confidences = [keyword_result[1], frequency_result[1], level_result[1]]
        all_reasons = keyword_result[2] + frequency_result[2] + level_result[2]
        
        # Weighted decision
        elective_votes = sum(1 for c in classifications if c == "elective")
        core_votes = sum(1 for c in classifications if c == "core")
        
        if elective_votes > core_votes:
            initial_classification = "elective"
            initial_confidence = sum(c for i, c in enumerate(confidences) 
                                   if classifications[i] == "elective") / elective_votes
        else:
            initial_classification = "core"
            initial_confidence = sum(c for i, c in enumerate(confidences) 
                                   if classifications[i] == "core") / core_votes
        
        # LLM validation
        final_classification, final_confidence, llm_reasons = self.llm_enhanced_classification(
            book, initial_classification, initial_confidence
        )
        
        all_reasons.extend(llm_reasons)
        
        # Determine elective domain
        elective_domain = None
        if final_classification == "elective" and keyword_result[3]:
            elective_domain = keyword_result[3]
        
        return ClassificationResult(
            book_id=book_id,
            book_title=book_title,
            classification=final_classification,
            confidence=final_confidence,
            reasons=all_reasons,
            elective_domain=elective_domain
        )
    
    def classify_all_books(self, toc_data: Dict) -> Dict[str, List[ClassificationResult]]:
        """Classify all books in the TOC data."""
        self.logger.start_timer("book_classification")
        
        # First pass: analyze frequencies
        self.analyze_topic_frequencies(toc_data)
        
        # Second pass: classify books
        results = {"core": [], "elective": []}
        
        tocs = toc_data.get("tocs_by_level", toc_data.get("tocs", {}))
        
        for level, books in tocs.items():
            if isinstance(books, list):
                self.logger.info(f"Classifying {len(books)} books at {level} level")
                
                for book in books:
                    result = self.classify_book(book, level)
                    results[result.classification].append(result)
                    
                    self.logger.debug(f"Classified '{result.book_title}' as {result.classification} "
                                    f"(confidence: {result.confidence:.2f})")
        
        self.logger.info(f"Classification complete: {len(results['core'])} core, "
                        f"{len(results['elective'])} elective books")
        
        self.logger.end_timer("book_classification")
        return results
    
    def organize_elective_domains(self, elective_results: List[ClassificationResult]) -> Dict[str, List[ClassificationResult]]:
        """Organize elective books by domain."""
        domains = defaultdict(list)
        
        for result in elective_results:
            domain = result.elective_domain or "Other Electives"
            domains[domain].append(result)
        
        return dict(domains)
    
    def create_output(self, classification_results: Dict, toc_data: Dict) -> Dict:
        """Create structured output with classified curriculum."""
        core_books = classification_results["core"]
        elective_books = classification_results["elective"]
        elective_domains = self.organize_elective_domains(elective_books)
        
        # Build output structure
        output = {
            "metadata": {
                "total_books": len(core_books) + len(elective_books),
                "core_books": len(core_books),
                "elective_books": len(elective_books),
                "elective_domains": list(elective_domains.keys()),
                "classification_threshold": self.config.elective_frequency_threshold,
                "timestamp": self.logger.logger.handlers[0].formatter.formatTime(
                    self.logger.logger.makeRecord("", 0, "", 0, "", (), None)
                )
            },
            "core": {},
            "electives": {}
        }
        
        # Add core books organized by level
        tocs = toc_data.get("tocs_by_level", toc_data.get("tocs", {}))
        
        for level in tocs.keys():
            core_books_at_level = [r for r in core_books 
                                 if any(book.get("book_title") == r.book_title 
                                       for book in tocs[level])]
            
            if core_books_at_level:
                output["core"][level] = []
                for result in core_books_at_level:
                    # Find original book data
                    original_book = next(
                        (book for book in tocs[level] 
                         if book.get("book_title") == result.book_title), 
                        {}
                    )
                    
                    book_data = original_book.copy()
                    book_data["classification_result"] = {
                        "classification": result.classification,
                        "confidence": result.confidence,
                        "reasons": result.reasons
                    }
                    output["core"][level].append(book_data)
        
        # Add elective books organized by domain
        for domain, domain_results in elective_domains.items():
            output["electives"][domain] = []
            
            for result in domain_results:
                # Find original book data
                original_book = None
                for level_books in tocs.values():
                    for book in level_books:
                        if book.get("book_title") == result.book_title:
                            original_book = book
                            break
                    if original_book:
                        break
                
                if original_book:
                    book_data = original_book.copy()
                    book_data["classification_result"] = {
                        "classification": result.classification,
                        "confidence": result.confidence,
                        "reasons": result.reasons,
                        "elective_domain": result.elective_domain
                    }
                    output["electives"][domain].append(book_data)
        
        return output


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Core/Elective Classification Engine")
    parser.add_argument("--input", "-i", default="../TOCs/extracted_tocs.json",
                       help="Input TOC data file")
    parser.add_argument("--output", "-o", default="../classified_curriculum.json",
                       help="Output classified curriculum file")
    parser.add_argument("--config", "-c", default="../config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--threshold", "-t", type=float,
                       help="Elective frequency threshold (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    if args.threshold:
        config.elective_frequency_threshold = args.threshold
    
    logger = CurriculumLogger("step3_classification", "DEBUG" if args.verbose else "INFO")
    file_manager = FileManager(logger)
    
    logger.info("Starting Core/Elective Classification Engine")
    logger.info(f"Elective threshold: {config.elective_frequency_threshold}")
    
    # Load input data
    logger.start_timer("data_loading")
    toc_data = file_manager.load_json(args.input)
    if not toc_data:
        logger.error(f"Failed to load input file: {args.input}")
        return 1
    
    # Validate input
    is_valid, errors = DataValidator.validate_toc_data(toc_data)
    if not is_valid:
        logger.error(f"Invalid input data: {errors}")
        return 1
    
    logger.end_timer("data_loading")
    
    # Run classification
    classifier = CoreElectiveClassifier(config, logger)
    
    try:
        classification_results = classifier.classify_all_books(toc_data)
        output_data = classifier.create_output(classification_results, toc_data)
        
        # Save results
        logger.start_timer("output_saving")
        if file_manager.save_json(output_data, args.output):
            logger.info(f"Classified curriculum saved to: {args.output}")
        else:
            logger.error("Failed to save output")
            return 1
        logger.end_timer("output_saving")
        
        # Performance summary
        logger.log_performance_summary()
        
        # Summary statistics
        logger.info("Classification Summary:")
        logger.info(f"  Core books: {output_data['metadata']['core_books']}")
        logger.info(f"  Elective books: {output_data['metadata']['elective_books']}")
        logger.info(f"  Elective domains: {len(output_data['metadata']['elective_domains'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())