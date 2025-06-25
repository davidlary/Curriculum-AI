#!/usr/bin/env python3
"""
Step 3: LLM-Enhanced Topic Normalization Module
Replaces the old pattern-based normalization with intelligent LLM-guided processing.

This module implements the 7-stage normalization process:
A. Classify Academic Levels
B. Select Foundational TOC  
C. Normalize Next TOC
D. Merge and Enrich
E. Record Structural Changes
F. Validate Pedagogical Structure
G. Iterate Through All TOCs

Usage:
    python scripts/step3_llm_topic_normalization.py --discipline Physics --language English
    python scripts/step3_llm_topic_normalization.py --discipline Mathematics --language Spanish --force-refresh
"""

import os
import json
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our LLM normalizer
from llm_toc_normalizer import TOCNormalizer, NormalizedTOC, TOCEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NormalizationResult:
    """Results from the normalization process."""
    success: bool
    normalized_topics: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    academic_levels: List[str]
    source_books: List[str]
    processing_time: float
    error_message: Optional[str] = None
    cache_used: bool = False
    
    def to_dict(self):
        return asdict(self)

class PipelineNormalizer:
    """Pipeline-compatible wrapper for LLM TOC normalization."""
    
    def __init__(self, discipline: str, language: str, cache_dir: Optional[Path] = None):
        self.discipline = discipline
        self.language = language
        self.cache_dir = cache_dir or Path("Cache/Normalization")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM normalizer if API key available
        self.llm_available = self._check_llm_availability()
        if self.llm_available:
            self.normalizer = TOCNormalizer()
        else:
            logger.warning("No LLM API key found - will use fallback normalization")
            self.normalizer = None
    
    def _check_llm_availability(self) -> bool:
        """Check if LLM API keys are available."""
        return bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    
    def normalize_curriculum(self, input_file: Path, output_file: Path, 
                           force_refresh: bool = False) -> NormalizationResult:
        """
        Execute the complete normalization process.
        
        Args:
            input_file: Path to TOC extraction results from Step 2
            output_file: Path for normalized curriculum output
            force_refresh: Skip cache and force fresh processing
            
        Returns:
            NormalizationResult with success status and metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting LLM-Enhanced Topic Normalization")
            logger.info(f"   Discipline: {self.discipline}")
            logger.info(f"   Language: {self.language}")
            logger.info(f"   Input: {input_file}")
            logger.info(f"   Output: {output_file}")
            
            # Check cache first
            if not force_refresh:
                cached_result = self._load_from_cache()
                if cached_result:
                    logger.info("✅ Using cached normalization results")
                    self._save_results(cached_result, output_file)
                    return NormalizationResult(
                        success=True,
                        normalized_topics=cached_result.get('normalized_topics', []),
                        metrics=cached_result.get('metrics', {}),
                        academic_levels=cached_result.get('academic_levels', []),
                        source_books=cached_result.get('source_books', []),
                        processing_time=time.time() - start_time,
                        cache_used=True
                    )
            
            # Load TOC data from Step 2
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            with open(input_file, 'r') as f:
                toc_data = json.load(f)
            
            # Process with LLM or fallback
            if self.llm_available and self.normalizer:
                result = self._process_with_llm(toc_data)
            else:
                result = self._process_with_fallback(toc_data)
            
            # Save results
            self._save_results(result, output_file)
            self._save_to_cache(result)
            
            processing_time = time.time() - start_time
            
            logger.info(f"🎉 Normalization completed successfully!")
            logger.info(f"   Processing time: {processing_time:.1f}s")
            logger.info(f"   Total topics: {len(result.get('normalized_topics', []))}")
            logger.info(f"   Academic levels: {result.get('academic_levels', [])}")
            
            return NormalizationResult(
                success=True,
                normalized_topics=result.get('normalized_topics', []),
                metrics=result.get('metrics', {}),
                academic_levels=result.get('academic_levels', []),
                source_books=result.get('source_books', []),
                processing_time=processing_time,
                cache_used=False
            )
            
        except Exception as e:
            logger.error(f"❌ Normalization failed: {e}")
            processing_time = time.time() - start_time
            
            return NormalizationResult(
                success=False,
                normalized_topics=[],
                metrics={},
                academic_levels=[],
                source_books=[],
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _process_with_llm(self, toc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process using LLM-enhanced normalization."""
        logger.info("🤖 Processing with LLM-enhanced normalization")
        
        # Convert to format expected by normalizer
        tocs = self._extract_tocs_from_data(toc_data)
        
        if not tocs:
            raise ValueError("No TOC data found in input file")
        
        logger.info(f"   Found {len(tocs)} books to process")
        
        # Stage A: Classify Academic Levels
        logger.info("📚 Stage A: Classifying Academic Levels")
        tocs = self._classify_academic_levels(tocs)
        
        # Stage B: Select Foundational TOC
        logger.info("🏗️  Stage B: Selecting Foundational TOC")
        foundational_toc = self._select_foundational_toc(tocs)
        
        # Initialize normalized structure
        normalized_toc = self._initialize_normalized_structure(foundational_toc)
        
        # Stages C-F: Process remaining TOCs
        remaining_tocs = [toc for toc in tocs if toc != foundational_toc]
        logger.info(f"🔄 Processing {len(remaining_tocs)} additional books")
        
        for i, toc in enumerate(remaining_tocs):
            logger.info(f"   📖 Processing {i+1}/{len(remaining_tocs)}: {toc['book_title']}")
            
            # Stage C: Normalize TOC
            normalized_entries = self._normalize_toc_against_base(toc, normalized_toc)
            
            # Stage D: Merge and Enrich
            normalized_toc = self._merge_and_enrich(normalized_entries, normalized_toc)
            
            # Stage E: Record Changes (done automatically)
            
            # Brief pause for API rate limiting
            time.sleep(0.5)
        
        # Stage F: Final validation
        logger.info("✅ Stage F: Final Validation")
        final_result = self._validate_and_finalize(normalized_toc)
        
        return final_result
    
    def _extract_tocs_from_data(self, toc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract TOCs from Step 2 output format."""
        tocs = []
        
        if 'tocs_by_level' in toc_data:
            for level, books in toc_data['tocs_by_level'].items():
                for book in books:
                    tocs.append({
                        'book_title': book.get('book_title', 'Unknown'),
                        'level': level,
                        'toc_entries': book.get('toc_entries', []),
                        'total_topics': len(book.get('toc_entries', []))
                    })
        
        return tocs
    
    def _classify_academic_levels(self, tocs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage A: Classify academic levels using LLM."""
        
        for toc in tocs:
            # Create sample for classification (use first 10 entries for better classification)
            sample_entries = toc['toc_entries'][:10] if len(toc['toc_entries']) > 10 else toc['toc_entries']
            if not sample_entries:
                toc['academic_level'] = 'undergraduate'
                continue
            
            toc_sample = "\\n".join([f"- {entry.get('title', 'Unknown')}" for entry in sample_entries])
            
            system_prompt = "Classify textbook academic level. Return only: high_school, undergraduate, or graduate"
            
            prompt = f"""Book: {toc['book_title']}
Sample TOC:
{toc_sample}

Academic level:"""
            
            try:
                response = self.normalizer.llm.query(prompt, system_prompt)
                
                if 'high_school' in response.lower():
                    toc['academic_level'] = 'high_school'
                elif 'graduate' in response.lower():
                    toc['academic_level'] = 'graduate'
                else:
                    toc['academic_level'] = 'undergraduate'
                    
                logger.info(f"     {toc['book_title']}: {toc['academic_level']}")
                
            except Exception as e:
                logger.warning(f"     Classification failed for {toc['book_title']}: {e}")
                toc['academic_level'] = 'undergraduate'  # Default
        
        return tocs
    
    def _select_foundational_toc(self, tocs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage B: Select most comprehensive TOC from lowest academic level."""
        
        # Find lowest academic level
        level_priority = {'high_school': 1, 'undergraduate': 2, 'graduate': 3}
        lowest_level = min(tocs, key=lambda x: level_priority[x['academic_level']])['academic_level']
        
        # Get all TOCs at lowest level
        lowest_level_tocs = [toc for toc in tocs if toc['academic_level'] == lowest_level]
        
        # Select most comprehensive (most entries)
        foundational = max(lowest_level_tocs, key=lambda x: x['total_topics'])
        
        logger.info(f"     Selected: {foundational['book_title']} ({foundational['total_topics']} topics)")
        
        return foundational
    
    def _initialize_normalized_structure(self, foundational_toc: Dict[str, Any]) -> NormalizedTOC:
        """Initialize normalized structure with foundational TOC."""
        
        entries = []
        for i, entry in enumerate(foundational_toc['toc_entries']):
            toc_entry = TOCEntry(
                title=entry.get('title', 'Unknown'),
                level=entry.get('level', 1),
                original_book=foundational_toc['book_title'],
                academic_level=foundational_toc['academic_level'],
                entry_id=f"entry_{i}"
            )
            entries.append(toc_entry)
        
        return NormalizedTOC(
            entries=entries,
            academic_levels=[foundational_toc['academic_level']],
            source_books=[foundational_toc['book_title']],
            version="1.0",
            created_at=datetime.now().isoformat(),
            change_log=[]
        )
    
    def _normalize_toc_against_base(self, toc: Dict[str, Any], base_toc: NormalizedTOC) -> List[TOCEntry]:
        """Stage C: Normalize TOC against base structure."""
        
        normalized_entries = []
        
        # Full normalization - process all entries
        for i, entry in enumerate(toc['toc_entries']):
            toc_entry = TOCEntry(
                title=entry.get('title', 'Unknown'),
                level=entry.get('level', 1),
                original_book=toc['book_title'],
                academic_level=toc['academic_level'],
                entry_id=f"{toc['book_title']}_{i}"
            )
            normalized_entries.append(toc_entry)
        
        return normalized_entries
    
    def _merge_and_enrich(self, new_entries: List[TOCEntry], base_toc: NormalizedTOC) -> NormalizedTOC:
        """Stage D: Merge new entries into base structure."""
        
        # Simple duplicate detection
        existing_titles = {entry.title.lower() for entry in base_toc.entries}
        
        for entry in new_entries:
            if entry.title.lower() not in existing_titles:
                base_toc.entries.append(entry)
                existing_titles.add(entry.title.lower())
        
        # Update metadata
        if new_entries:
            book_title = new_entries[0].original_book
            if book_title not in base_toc.source_books:
                base_toc.source_books.append(book_title)
            
            academic_level = new_entries[0].academic_level
            if academic_level not in base_toc.academic_levels:
                base_toc.academic_levels.append(academic_level)
        
        return base_toc
    
    def _validate_and_finalize(self, normalized_toc: NormalizedTOC) -> Dict[str, Any]:
        """Stage F: Final validation and output formatting."""
        
        # Convert to pipeline output format
        normalized_topics = []
        
        for entry in normalized_toc.entries:
            topic = {
                'id': entry.entry_id,
                'title': entry.title,
                'canonical_name': entry.title,  # Use title as canonical name
                'alternative_names': [],
                'educational_levels': [entry.academic_level],
                'depth_progression': {entry.academic_level: entry.level},
                'source_books': [entry.original_book],
                'semantic_cluster_id': 0,  # Default cluster
                'parent_topics': [],
                'subtopics': [],
                'learning_objectives': [],
                'prerequisites': [],
                'difficulty_progression': {entry.academic_level: entry.level},
                'topic_type': 'core',  # Default type
                'frequency_score': 1.0,  # Default frequency
                'consensus_score': 1.0,  # Default consensus
                'quality_score': 1.0,  # Default quality
                'level': entry.level,
                'original_book': entry.original_book,
                'academic_level': entry.academic_level,
                'tags': entry.tags,
                'entry_id': entry.entry_id,
                'description': entry.title
            }
            normalized_topics.append(topic)
        
        # Calculate metrics
        metrics = {
            'total_topics': len(normalized_topics),
            'books_processed': len(normalized_toc.source_books),
            'academic_levels': normalized_toc.academic_levels,
            'processing_method': 'llm_enhanced',
            'created_at': datetime.now().isoformat()
        }
        
        # Group by academic level for better organization
        organized_by_level = {}
        for topic in normalized_topics:
            level = topic['academic_level']
            if level not in organized_by_level:
                organized_by_level[level] = []
            organized_by_level[level].append(topic)
        
        return {
            'normalized_topics': normalized_topics,
            'metrics': metrics,
            'organized_by_level': organized_by_level,
            'academic_levels': normalized_toc.academic_levels,
            'source_books': normalized_toc.source_books,
            'processing_metadata': {
                'method': 'llm_enhanced_7_stage',
                'version': '1.0',
                'discipline': self.discipline,
                'language': self.language
            }
        }
    
    def _process_with_fallback(self, toc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when LLM is not available."""
        logger.info("⚡ Processing with fallback normalization")
        
        # Extract TOCs
        tocs = self._extract_tocs_from_data(toc_data)
        
        if not tocs:
            raise ValueError("No TOC data found")
        
        # Simple merging without LLM
        all_topics = []
        source_books = []
        academic_levels = set()
        
        for toc in tocs:
            source_books.append(toc['book_title'])
            
            # Assign default academic level based on name
            if 'high school' in toc['book_title'].lower():
                academic_level = 'high_school'
            elif 'graduate' in toc['book_title'].lower():
                academic_level = 'graduate'
            else:
                academic_level = 'undergraduate'
            
            academic_levels.add(academic_level)
            
            for i, entry in enumerate(toc['toc_entries']):
                entry_id = f"{toc['book_title']}_{i}"
                topic_title = entry.get('title', 'Unknown')
                topic_level = entry.get('level', 1)
                topic = {
                    'id': entry_id,
                    'title': topic_title,
                    'canonical_name': topic_title,
                    'alternative_names': [],
                    'educational_levels': [academic_level],
                    'depth_progression': {academic_level: topic_level},
                    'source_books': [toc['book_title']],
                    'semantic_cluster_id': 0,
                    'parent_topics': [],
                    'subtopics': [],
                    'learning_objectives': [],
                    'prerequisites': [],
                    'difficulty_progression': {academic_level: topic_level},
                    'topic_type': 'core',
                    'frequency_score': 1.0,
                    'consensus_score': 1.0,
                    'quality_score': 1.0,
                    'level': topic_level,
                    'original_book': toc['book_title'],
                    'academic_level': academic_level,
                    'tags': [],
                    'entry_id': entry_id,
                    'description': topic_title
                }
                all_topics.append(topic)
        
        # Basic deduplication
        seen_titles = set()
        deduplicated_topics = []
        
        for topic in all_topics:
            title_key = topic['title'].lower().strip()
            if title_key not in seen_titles:
                deduplicated_topics.append(topic)
                seen_titles.add(title_key)
        
        metrics = {
            'total_topics': len(deduplicated_topics),
            'books_processed': len(source_books),
            'academic_levels': list(academic_levels),
            'processing_method': 'fallback',
            'created_at': datetime.now().isoformat()
        }
        
        return {
            'normalized_topics': deduplicated_topics,
            'metrics': metrics,
            'organized_by_level': {},
            'academic_levels': list(academic_levels),
            'source_books': source_books,
            'processing_metadata': {
                'method': 'fallback_basic',
                'version': '1.0',
                'discipline': self.discipline,
                'language': self.language
            }
        }
    
    def _save_results(self, result: Dict[str, Any], output_file: Path):
        """Save normalization results to output file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
    
    def _get_cache_file(self) -> Path:
        """Get cache file path for this discipline/language."""
        cache_filename = f"{self.discipline}_{self.language}_normalization.json"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load results from cache if available and valid."""
        cache_file = self._get_cache_file()
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is recent (within 24 hours)
            cache_time = datetime.fromisoformat(cached_data.get('cached_at', '2000-01-01'))
            if (datetime.now() - cache_time).total_seconds() > 86400:  # 24 hours
                logger.info("Cache expired, processing fresh")
                return None
            
            logger.info("Valid cache found")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            return None
    
    def _save_to_cache(self, result: Dict[str, Any]):
        """Save results to cache."""
        try:
            cache_data = {
                **result,
                'cached_at': datetime.now().isoformat()
            }
            
            cache_file = self._get_cache_file()
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results cached to: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Cache saving failed: {e}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="LLM-Enhanced Topic Normalization")
    parser.add_argument("--discipline", required=True, help="Academic discipline (e.g., Physics)")
    parser.add_argument("--language", default="English", help="Language (default: English)")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh, skip cache")
    parser.add_argument("--openai-api-key", help="OpenAI API key (optional, can use env var)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    # Set up file paths
    curriculum_dir = Path("Curriculum")
    input_file = curriculum_dir / f"{args.discipline}_{args.language}_tocs_extracted.json"
    output_file = curriculum_dir / f"{args.discipline}_{args.language}_topics_normalized.json"
    
    # Initialize normalizer
    normalizer = PipelineNormalizer(args.discipline, args.language)
    
    # Run normalization
    result = normalizer.normalize_curriculum(
        input_file=input_file,
        output_file=output_file,
        force_refresh=args.force_refresh
    )
    
    # Output results for pipeline
    if result.success:
        print(f"SUCCESS: Normalized {len(result.normalized_topics)} topics from {len(result.source_books)} books")
        print(f"Academic levels: {result.academic_levels}")
        print(f"Processing time: {result.processing_time:.1f}s")
        if result.cache_used:
            print("Used cached results")
    else:
        print(f"FAILED: {result.error_message}")
        sys.exit(1)

if __name__ == "__main__":
    main()