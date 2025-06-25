#!/usr/bin/env python3
"""
Step 3: Topic Normalization Module
Unifies cross-book content across all educational levels through semantic alignment and concept clustering.

This module:
1. Processes TOCs from Step 2 to identify common topics across books/levels
2. Uses semantic similarity for topic alignment and clustering
3. Handles cross-level topic merging with depth progression
4. Removes duplicates while preserving topic diversity
5. Creates normalized topic hierarchy for curriculum sequencing

Usage:
    python scripts/step3_topic_normalization.py --discipline Physics --language English
    python scripts/step3_topic_normalization.py --discipline Mathematics --language Spanish --similarity-threshold 0.85
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import re

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import existing core modules and AI capabilities
try:
    from core.enhanced_logging import setup_logging
    from core.data_models import SubtopicEntry, EducationalLevel, BloomLevel
    from scripts.adaptive_curriculum_system import AdaptiveJSONParser
    logger.info("Successfully imported core modules")
except ImportError as e:
    logger.warning(f"Some core modules not available: {e}")
    logger.info("Using simplified functionality")

# Try to import embedding capabilities
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: OpenAI embeddings not available. Using TF-IDF for similarity.")

# Configure logging
logger = logging.getLogger(__name__)

# Directory structure
BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "Cache" / "Normalization"
OUTPUT_DIR = BASE_DIR / "Curriculum"
TOCS_DIR = BASE_DIR / "Cache" / "TOCs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class NormalizedTopic:
    """Represents a normalized topic across multiple books and levels."""
    id: str
    canonical_name: str
    alternative_names: List[str]
    educational_levels: List[str]  # Levels where this topic appears
    depth_progression: Dict[str, int]  # Level -> depth mapping
    source_books: List[str]
    semantic_cluster_id: int
    parent_topics: List[str]
    subtopics: List[str]
    learning_objectives: List[str]
    prerequisites: List[str]
    difficulty_progression: Dict[str, int]  # Level -> difficulty mapping
    topic_type: str  # core, elective, foundational, advanced
    frequency_score: float  # How often this topic appears across books
    consensus_score: float  # Agreement level across sources
    quality_score: float

@dataclass
class TopicCluster:
    """Represents a cluster of semantically similar topics."""
    cluster_id: int
    central_topic: str
    member_topics: List[str]
    similarity_scores: List[float]
    educational_span: List[str]
    consensus_level: float

@dataclass
class NormalizationMetrics:
    """Metrics for the normalization process."""
    total_raw_topics: int
    normalized_topics: int
    reduction_ratio: float
    cross_level_alignments: int
    semantic_clusters: int
    avg_consensus_score: float
    coverage_by_level: Dict[str, int]
    processing_time: float
    quality_distribution: Dict[str, int]
    duplicate_removal_count: int

class TopicNormalizationEngine:
    """
    Main engine for normalizing topics across books and educational levels.
    Uses semantic similarity and clustering for intelligent topic unification.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, openai_api_key: Optional[str] = None):
        """Initialize the normalization engine."""
        self.similarity_threshold = similarity_threshold
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize embeddings if available
        if EMBEDDINGS_AVAILABLE and self.openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            self.use_embeddings = True
        else:
            self.embeddings = None
            self.use_embeddings = False
            logger.warning("Using TF-IDF similarity instead of embeddings")
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=10000,
            lowercase=True
        )
        
        # Cache configuration
        self.cache_ttl_hours = 72  # Cache valid for 3 days
        self.cache_version = "1.0"
        
        # Quality thresholds
        self.min_consensus_score = 0.6
        self.min_frequency_score = 0.3
        
        # Educational level hierarchy
        self.level_hierarchy = {
            'high_school': 1,
            'undergraduate': 2,
            'graduate': 3,
            'professional': 4
        }
        
        # Physics-specific domain knowledge
        self.physics_domains = {
            'mechanics': ['mechanics', 'motion', 'force', 'energy', 'momentum', 'dynamics', 'kinematics', 'statics'],
            'thermodynamics': ['thermodynamics', 'heat', 'temperature', 'entropy', 'statistical mechanics', 'thermal'],
            'electromagnetism': ['electromagnetism', 'electricity', 'magnetism', 'electromagnetic', 'maxwell', 'field'],
            'waves_optics': ['waves', 'optics', 'light', 'interference', 'diffraction', 'refraction', 'reflection'],
            'quantum': ['quantum', 'atomic', 'molecular', 'particle', 'photon', 'electron', 'orbital'],
            'modern_physics': ['relativity', 'nuclear', 'radioactivity', 'fusion', 'fission', 'particle physics'],
            'condensed_matter': ['solid state', 'crystal', 'semiconductor', 'conductor', 'superconductor', 'materials'],
            'astrophysics': ['astrophysics', 'cosmology', 'stellar', 'galaxy', 'universe', 'astronomy', 'planetary']
        }
        
        logger.info(f"TopicNormalizationEngine initialized (embeddings: {self.use_embeddings})")

    def normalize_topics_from_tocs(self, discipline: str, language: str = "English", 
                                  force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main normalization method that processes TOCs from Step 2.
        
        Args:
            discipline: Target discipline
            language: Target language
            force_refresh: Force refresh of cached results
            
        Returns:
            Dictionary containing normalized topics and metrics
        """
        start_time = time.time()
        logger.info(f"Starting topic normalization for {discipline} in {language}")
        
        # Check cache first
        cache_key = self._generate_cache_key(discipline, language)
        if not force_refresh:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached normalization results for {discipline}")
                return cached_result
        
        # Load TOC extraction results from Step 2
        tocs_file = OUTPUT_DIR / f"{discipline}_{language}_tocs_extracted.json"
        if not tocs_file.exists():
            raise FileNotFoundError(f"TOC extraction file not found: {tocs_file}")
        
        with open(tocs_file, 'r', encoding='utf-8') as f:
            toc_data = json.load(f)
        
        logger.info(f"Loaded TOC data with {toc_data['successful_extractions']} successful extractions")
        
        # Extract all topics from TOCs
        raw_topics = self._extract_all_topics(toc_data)
        logger.info(f"Extracted {len(raw_topics)} raw topics")
        
        # Calculate semantic similarities
        similarity_matrix = self._calculate_topic_similarities(raw_topics)
        
        # Perform clustering
        topic_clusters = self._cluster_topics(raw_topics, similarity_matrix)
        logger.info(f"Created {len(topic_clusters)} topic clusters")
        
        # Normalize topics within clusters
        normalized_topics = self._normalize_clustered_topics(topic_clusters, raw_topics)
        logger.info(f"Normalized to {len(normalized_topics)} topics")
        
        # Enhance with cross-level analysis
        enhanced_topics = self._enhance_with_cross_level_analysis(normalized_topics)
        
        # Automated elective detection based on discipline knowledge and TOC analysis
        topics_with_electives = self._detect_elective_topics(enhanced_topics, discipline)
        
        # Remove duplicates and validate
        final_topics = self._remove_duplicates_and_validate(topics_with_electives)
        
        # Calculate metrics
        metrics = self._calculate_normalization_metrics(raw_topics, final_topics, start_time)
        
        # Organize by educational level and domain
        organized_topics = self._organize_topics(final_topics)
        
        # Prepare result
        result = {
            'discipline': discipline,
            'language': language,
            'normalization_timestamp': datetime.now().isoformat(),
            'normalized_topics': [asdict(topic) for topic in final_topics],
            'organized_by_level': organized_topics['by_level'],
            'organized_by_domain': organized_topics['by_domain'],
            'topic_clusters': [asdict(cluster) for cluster in topic_clusters],
            'metrics': asdict(metrics),
            'cache_version': self.cache_version,
            'total_normalized_topics': len(final_topics),
            'elective_classification': getattr(self, 'elective_classification_summary', {}),
            'preprocessing_summary': {
                'raw_topics': len(raw_topics),
                'clusters_formed': len(topic_clusters),
                'reduction_ratio': metrics.reduction_ratio
            }
        }
        
        # Cache the result
        self._save_to_cache(cache_key, result)
        
        # Save to output directory
        output_file = OUTPUT_DIR / f"{discipline}_{language}_topics_normalized.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        converted_result = convert_numpy_types(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Normalization completed: {len(raw_topics)} -> {len(final_topics)} topics in {metrics.processing_time:.2f}s")
        return result

    def _extract_all_topics(self, toc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all topics from TOC data across all books and levels."""
        raw_topics = []
        topic_id_counter = 0
        
        # Process TOCs by level
        tocs_by_level = toc_data.get('tocs_by_level', {})
        
        for level, level_tocs in tocs_by_level.items():
            for book_toc in level_tocs:
                book_id = book_toc['book_id']
                book_title = book_toc['book_title']
                
                for toc_entry in book_toc['toc_entries']:
                    topic_id_counter += 1
                    
                    raw_topic = {
                        'id': f"topic_{topic_id_counter}",
                        'title': toc_entry.get('title', ''),
                        'level': toc_entry.get('level', 1),
                        'educational_level': level,
                        'source_book_id': book_id,
                        'source_book_title': book_title,
                        'page_number': toc_entry.get('page_number'),
                        'section_number': toc_entry.get('section_number'),
                        'parent_id': toc_entry.get('parent_id'),
                        'hierarchy_level': toc_entry.get('level', 1)
                    }
                    
                    # Clean and enhance the topic title
                    cleaned_title = self._clean_topic_title(raw_topic['title'])
                    if cleaned_title and len(cleaned_title) > 2:
                        raw_topic['cleaned_title'] = cleaned_title
                        raw_topic['domain'] = self._classify_domain(cleaned_title)
                        raw_topics.append(raw_topic)
        
        return raw_topics

    def _clean_topic_title(self, title: str) -> str:
        """Clean and standardize topic titles."""
        if not title:
            return ""
        
        # Remove common prefixes and suffixes
        cleaned = title.strip()
        
        # Remove chapter/section numbers
        cleaned = re.sub(r'^\d+\.?\d*\s*', '', cleaned)
        cleaned = re.sub(r'^Chapter\s+\d+\s*:?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Section\s+\d+\.?\d*\s*:?\s*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove common formatting
        cleaned = re.sub(r'\s*\.\s*$', '', cleaned)  # Trailing periods
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces
        
        # Standardize case
        cleaned = cleaned.strip()
        
        # Don't modify if it's already well-formatted
        if cleaned and not cleaned.islower() and not cleaned.isupper():
            return cleaned
        
        # Title case for simple titles
        return cleaned.title() if cleaned else ""

    def _classify_domain(self, title: str) -> str:
        """Classify topic into physics domain based on title."""
        title_lower = title.lower()
        
        for domain, keywords in self.physics_domains.items():
            if any(keyword in title_lower for keyword in keywords):
                return domain
        
        return 'general'

    def _calculate_topic_similarities(self, topics: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate semantic similarities between all topics."""
        topic_texts = [topic['cleaned_title'] for topic in topics]
        
        if self.use_embeddings:
            return self._calculate_embedding_similarities(topic_texts)
        else:
            return self._calculate_tfidf_similarities(topic_texts)

    def _calculate_embedding_similarities(self, topic_texts: List[str]) -> np.ndarray:
        """Calculate similarities using OpenAI embeddings."""
        try:
            logger.info("Calculating embedding similarities...")
            embeddings_list = self.embeddings.embed_documents(topic_texts)
            embeddings_array = np.array(embeddings_list)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(embeddings_array)
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error calculating embeddings: {e}")
            logger.info("Falling back to TF-IDF similarities")
            return self._calculate_tfidf_similarities(topic_texts)

    def _calculate_tfidf_similarities(self, topic_texts: List[str]) -> np.ndarray:
        """Calculate similarities using TF-IDF vectors."""
        logger.info("Calculating TF-IDF similarities...")
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(topic_texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix

    def _cluster_topics(self, topics: List[Dict[str, Any]], similarity_matrix: np.ndarray) -> List[TopicCluster]:
        """Cluster topics based on semantic similarity."""
        logger.info("Clustering topics...")
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Perform hierarchical clustering
        n_clusters = min(len(topics) // 3, 100)  # Adaptive cluster count
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group topics by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
        
        # Create TopicCluster objects
        topic_clusters = []
        for cluster_id, topic_indices in clusters.items():
            if len(topic_indices) >= 2:  # Only meaningful clusters
                # Find central topic (highest average similarity to others in cluster)
                avg_similarities = []
                for i in topic_indices:
                    avg_sim = np.mean([similarity_matrix[i][j] for j in topic_indices if i != j])
                    avg_similarities.append(avg_sim)
                
                central_idx = topic_indices[np.argmax(avg_similarities)]
                central_topic = topics[central_idx]['cleaned_title']
                
                member_topics = [topics[i]['cleaned_title'] for i in topic_indices]
                similarities = [similarity_matrix[central_idx][i] for i in topic_indices]
                
                # Get educational span
                educational_levels = list(set(topics[i]['educational_level'] for i in topic_indices))
                
                # Calculate consensus level
                consensus = np.mean(similarities)
                
                cluster = TopicCluster(
                    cluster_id=cluster_id,
                    central_topic=central_topic,
                    member_topics=member_topics,
                    similarity_scores=similarities,
                    educational_span=educational_levels,
                    consensus_level=consensus
                )
                
                topic_clusters.append(cluster)
        
        return topic_clusters

    def _normalize_clustered_topics(self, clusters: List[TopicCluster], raw_topics: List[Dict[str, Any]]) -> List[NormalizedTopic]:
        """Normalize topics within each cluster."""
        logger.info("Normalizing clustered topics...")
        
        normalized_topics = []
        
        for cluster in clusters:
            # Find all raw topics in this cluster
            cluster_topics = []
            for topic in raw_topics:
                if topic['cleaned_title'] in cluster.member_topics:
                    cluster_topics.append(topic)
            
            # Create normalized topic
            normalized = self._create_normalized_topic(cluster, cluster_topics)
            if normalized:
                normalized_topics.append(normalized)
        
        # Handle singleton topics (not in any meaningful cluster)
        clustered_titles = set()
        for cluster in clusters:
            clustered_titles.update(cluster.member_topics)
        
        singleton_topics = [t for t in raw_topics if t['cleaned_title'] not in clustered_titles]
        for topic in singleton_topics:
            normalized = self._create_normalized_topic_from_single(topic)
            if normalized:
                normalized_topics.append(normalized)
        
        return normalized_topics

    def _create_normalized_topic(self, cluster: TopicCluster, cluster_topics: List[Dict[str, Any]]) -> Optional[NormalizedTopic]:
        """Create a normalized topic from a cluster of similar topics."""
        if not cluster_topics:
            return None
        
        # Use central topic as canonical name
        canonical_name = cluster.central_topic
        
        # Collect alternative names
        alternative_names = list(set(t['cleaned_title'] for t in cluster_topics if t['cleaned_title'] != canonical_name))
        
        # Collect educational levels
        educational_levels = list(set(t['educational_level'] for t in cluster_topics))
        
        # Calculate depth progression
        depth_progression = {}
        difficulty_progression = {}
        for level in educational_levels:
            level_topics = [t for t in cluster_topics if t['educational_level'] == level]
            avg_hierarchy = np.mean([t['hierarchy_level'] for t in level_topics])
            depth_progression[level] = int(avg_hierarchy)
            
            # Difficulty based on educational level and hierarchy
            base_difficulty = self.level_hierarchy.get(level, 2)
            difficulty_progression[level] = min(base_difficulty + int(avg_hierarchy) - 1, 5)
        
        # Collect source books
        source_books = list(set(t['source_book_title'] for t in cluster_topics))
        
        # Calculate scores
        frequency_score = len(cluster_topics) / len(source_books) if source_books else 0
        consensus_score = cluster.consensus_level
        quality_score = (frequency_score + consensus_score) / 2
        
        # Determine topic type
        topic_type = self._determine_topic_type(canonical_name, educational_levels, frequency_score)
        
        # Generate learning objectives
        learning_objectives = self._generate_learning_objectives(canonical_name, educational_levels)
        
        return NormalizedTopic(
            id=hashlib.md5(canonical_name.encode()).hexdigest()[:8],
            canonical_name=canonical_name,
            alternative_names=alternative_names,
            educational_levels=educational_levels,
            depth_progression=depth_progression,
            source_books=source_books,
            semantic_cluster_id=cluster.cluster_id,
            parent_topics=[],  # Will be filled later
            subtopics=[],  # Will be filled later
            learning_objectives=learning_objectives,
            prerequisites=[],  # Will be filled in Step 4
            difficulty_progression=difficulty_progression,
            topic_type=topic_type,
            frequency_score=frequency_score,
            consensus_score=consensus_score,
            quality_score=quality_score
        )

    def _create_normalized_topic_from_single(self, topic: Dict[str, Any]) -> Optional[NormalizedTopic]:
        """Create a normalized topic from a single topic (not clustered)."""
        canonical_name = topic['cleaned_title']
        
        # Basic difficulty and depth
        level = topic['educational_level']
        hierarchy_level = topic['hierarchy_level']
        
        base_difficulty = self.level_hierarchy.get(level, 2)
        difficulty = min(base_difficulty + hierarchy_level - 1, 5)
        
        # Lower scores for singleton topics
        frequency_score = 0.3
        consensus_score = 0.5
        quality_score = 0.4
        
        topic_type = self._determine_topic_type(canonical_name, [level], frequency_score)
        learning_objectives = self._generate_learning_objectives(canonical_name, [level])
        
        return NormalizedTopic(
            id=hashlib.md5(canonical_name.encode()).hexdigest()[:8],
            canonical_name=canonical_name,
            alternative_names=[],
            educational_levels=[level],
            depth_progression={level: hierarchy_level},
            source_books=[topic['source_book_title']],
            semantic_cluster_id=-1,  # Not clustered
            parent_topics=[],
            subtopics=[],
            learning_objectives=learning_objectives,
            prerequisites=[],
            difficulty_progression={level: difficulty},
            topic_type=topic_type,
            frequency_score=frequency_score,
            consensus_score=consensus_score,
            quality_score=quality_score
        )

    def _determine_topic_type(self, topic_name: str, levels: List[str], frequency: float) -> str:
        """Determine if topic is core, elective, foundational, or advanced."""
        topic_lower = topic_name.lower()
        
        # Elective indicators
        elective_keywords = ['astrophysics', 'cosmology', 'biophysics', 'geophysics', 'medical', 'applications']
        if any(keyword in topic_lower for keyword in elective_keywords):
            return 'elective'
        
        # Foundational topics (high frequency, early levels)
        if frequency > 0.7 and 'high_school' in levels:
            return 'foundational'
        
        # Advanced topics (graduate level only)
        if set(levels) == {'graduate'} or set(levels) == {'professional'}:
            return 'advanced'
        
        # Core topics (appear across multiple levels)
        if len(levels) >= 2 and frequency > 0.5:
            return 'core'
        
        return 'core'  # Default

    def _generate_learning_objectives(self, topic_name: str, levels: List[str]) -> List[str]:
        """Generate learning objectives for a topic based on educational levels."""
        objectives = []
        
        base_verbs = {
            'high_school': ['understand', 'identify', 'describe', 'explain'],
            'undergraduate': ['analyze', 'apply', 'calculate', 'demonstrate'],
            'graduate': ['evaluate', 'synthesize', 'research', 'derive'],
            'professional': ['design', 'optimize', 'implement', 'innovate']
        }
        
        for level in levels:
            verbs = base_verbs.get(level, ['understand'])
            verb = verbs[0]  # Use first verb for simplicity
            objectives.append(f"{verb.capitalize()} {topic_name.lower()}")
        
        return objectives

    def _enhance_with_cross_level_analysis(self, topics: List[NormalizedTopic]) -> List[NormalizedTopic]:
        """Enhance topics with cross-level analysis and relationships."""
        logger.info("Enhancing with cross-level analysis...")
        
        # Group topics by canonical name across levels
        topic_groups = defaultdict(list)
        for topic in topics:
            base_name = self._get_base_topic_name(topic.canonical_name)
            topic_groups[base_name].append(topic)
        
        enhanced_topics = []
        
        for base_name, group_topics in topic_groups.items():
            if len(group_topics) > 1:
                # Multiple levels for same topic - merge with progression
                merged_topic = self._merge_cross_level_topics(group_topics)
                enhanced_topics.append(merged_topic)
            else:
                # Single level topic
                enhanced_topics.append(group_topics[0])
        
        return enhanced_topics

    def _get_base_topic_name(self, topic_name: str) -> str:
        """Extract base topic name removing level-specific qualifiers."""
        # Remove level-specific words
        level_words = ['introductory', 'basic', 'advanced', 'graduate', 'undergraduate']
        
        base_name = topic_name.lower()
        for word in level_words:
            base_name = base_name.replace(word, '').strip()
        
        # Clean up extra spaces
        base_name = ' '.join(base_name.split())
        
        return base_name

    def _merge_cross_level_topics(self, group_topics: List[NormalizedTopic]) -> NormalizedTopic:
        """Merge topics that appear across multiple educational levels."""
        # Sort by educational level hierarchy
        sorted_topics = sorted(group_topics, key=lambda t: min(self.level_hierarchy.get(level, 0) for level in t.educational_levels))
        
        base_topic = sorted_topics[0]
        
        # Merge all educational levels
        all_levels = []
        for topic in group_topics:
            all_levels.extend(topic.educational_levels)
        unique_levels = list(set(all_levels))
        
        # Merge depth and difficulty progressions
        merged_depth = {}
        merged_difficulty = {}
        for topic in group_topics:
            merged_depth.update(topic.depth_progression)
            merged_difficulty.update(topic.difficulty_progression)
        
        # Merge source books and alternative names
        all_sources = []
        all_alternatives = []
        for topic in group_topics:
            all_sources.extend(topic.source_books)
            all_alternatives.extend(topic.alternative_names)
        
        # Calculate enhanced scores
        avg_frequency = np.mean([t.frequency_score for t in group_topics])
        avg_consensus = np.mean([t.consensus_score for t in group_topics])
        enhanced_quality = (avg_frequency + avg_consensus + len(unique_levels) * 0.1) / 2
        
        # Merge learning objectives
        all_objectives = []
        for topic in group_topics:
            all_objectives.extend(topic.learning_objectives)
        unique_objectives = list(set(all_objectives))
        
        return NormalizedTopic(
            id=base_topic.id,
            canonical_name=base_topic.canonical_name,
            alternative_names=list(set(all_alternatives)),
            educational_levels=unique_levels,
            depth_progression=merged_depth,
            source_books=list(set(all_sources)),
            semantic_cluster_id=base_topic.semantic_cluster_id,
            parent_topics=base_topic.parent_topics,
            subtopics=base_topic.subtopics,
            learning_objectives=unique_objectives,
            prerequisites=base_topic.prerequisites,
            difficulty_progression=merged_difficulty,
            topic_type=base_topic.topic_type,
            frequency_score=avg_frequency,
            consensus_score=avg_consensus,
            quality_score=enhanced_quality
        )

    def _detect_elective_topics(self, topics: List[NormalizedTopic], discipline: str) -> List[NormalizedTopic]:
        """Automatically detect elective topics based on discipline knowledge and TOC analysis."""
        logger.info(f"Detecting elective topics for {discipline}")
        
        # Get discipline-specific knowledge
        core_curriculum = self._get_core_curriculum_topics(discipline)
        specialized_areas = self._get_specialized_areas(discipline)
        
        for topic in topics:
            # Analyze topic characteristics for elective detection
            elective_score = self._calculate_elective_score(
                topic, discipline, core_curriculum, specialized_areas
            )
            
            # Classify topic type based on analysis
            if elective_score > 0.7:
                topic.topic_type = "elective"
            elif elective_score > 0.4:
                topic.topic_type = "specialized"
            elif topic.frequency_score < 0.3:  # Appears in few books
                topic.topic_type = "advanced"
            else:
                topic.topic_type = "core"
        
        # Log classification results
        type_counts = Counter(topic.topic_type for topic in topics)
        logger.info(f"Topic classification: {dict(type_counts)}")
        
        # Store classification summary for output
        self.elective_classification_summary = {
            'total_topics': len(topics),
            'by_type': dict(type_counts),
            'elective_topics': [topic.canonical_name for topic in topics if topic.topic_type in ['elective', 'specialized']],
            'core_topics': [topic.canonical_name for topic in topics if topic.topic_type == 'core']
        }
        
        return topics
    
    def _get_core_curriculum_topics(self, discipline: str) -> Set[str]:
        """Get core curriculum topics for a discipline based on educational standards."""
        core_topics = {
            'Physics': {
                # Mechanics (Core)
                'motion', 'forces', 'newton', 'energy', 'momentum', 'work', 'kinematics', 'dynamics',
                'acceleration', 'velocity', 'gravity', 'friction', 'collision', 'conservation',
                
                # Thermodynamics (Core)
                'heat', 'temperature', 'thermodynamics', 'gas laws', 'entropy', 'thermal',
                
                # Electricity & Magnetism (Core)  
                'electric', 'magnetic', 'current', 'voltage', 'resistance', 'circuits',
                'electromagnetic', 'field', 'charge', 'capacitor', 'inductor',
                
                # Waves & Optics (Core)
                'wave', 'frequency', 'wavelength', 'sound', 'light', 'optics', 'reflection',
                'refraction', 'interference', 'diffraction',
                
                # Modern Physics (Advanced Core)
                'quantum', 'relativity', 'atomic', 'nuclear', 'photon', 'electron'
            },
            'Mathematics': {
                'algebra', 'calculus', 'geometry', 'trigonometry', 'statistics', 'probability',
                'linear algebra', 'differential equations', 'integration', 'derivative'
            },
            'Chemistry': {
                'atomic structure', 'periodic table', 'bonding', 'reactions', 'stoichiometry',
                'acids', 'bases', 'organic', 'inorganic', 'thermochemistry'
            }
        }
        
        return core_topics.get(discipline, set())
    
    def _get_specialized_areas(self, discipline: str) -> Set[str]:
        """Get specialized/elective areas for a discipline."""
        specialized = {
            'Physics': {
                # Astronomy/Astrophysics (Elective)
                'astronomy', 'astrophysics', 'cosmology', 'stellar', 'galactic', 'planetary',
                'solar system', 'universe', 'big bang', 'black hole', 'star formation',
                'exoplanet', 'nebula', 'galaxy', 'comet', 'asteroid',
                
                # Specialized Physics (Elective)
                'particle physics', 'condensed matter', 'plasma physics', 'biophysics',
                'geophysics', 'medical physics', 'laser physics', 'nanotechnology',
                
                # Applied Physics (Specialized)
                'engineering physics', 'materials science', 'semiconductor', 'superconductor',
                'crystallography', 'solid state'
            },
            'Mathematics': {
                'number theory', 'topology', 'abstract algebra', 'real analysis',
                'complex analysis', 'discrete mathematics', 'game theory'
            },
            'Chemistry': {
                'biochemistry', 'analytical chemistry', 'physical chemistry',
                'medicinal chemistry', 'polymer chemistry', 'environmental chemistry'
            }
        }
        
        return specialized.get(discipline, set())
    
    def _calculate_elective_score(self, topic: NormalizedTopic, discipline: str, 
                                 core_topics: Set[str], specialized_areas: Set[str]) -> float:
        """Calculate likelihood that a topic is an elective based on multiple factors."""
        score = 0.0
        topic_text = f"{topic.canonical_name} {' '.join(topic.alternative_names)}".lower()
        
        # Factor 1: Specialized area matching (0.4 weight)
        specialized_matches = sum(1 for term in specialized_areas if term in topic_text)
        if specialized_matches > 0:
            score += 0.4 * min(specialized_matches / 3, 1.0)
        
        # Factor 2: Core curriculum inverse matching (0.3 weight)
        core_matches = sum(1 for term in core_topics if term in topic_text)
        if core_matches == 0:
            score += 0.3  # Not matching core = more likely elective
        else:
            score -= 0.2 * min(core_matches / 3, 1.0)  # Strong core match = less likely elective
        
        # Factor 3: Frequency across books (0.2 weight) 
        # If appears in few books, more likely specialized/elective
        if topic.frequency_score < 0.3:
            score += 0.2
        elif topic.frequency_score > 0.8:
            score -= 0.1  # High frequency suggests core topic
        
        # Factor 4: Educational level distribution (0.1 weight)
        # Electives more common at higher levels
        if 'graduate' in topic.educational_levels:
            score += 0.1
        if len(topic.educational_levels) == 1 and 'undergraduate' in topic.educational_levels:
            score += 0.05  # Only at undergraduate level
        
        return max(0.0, min(1.0, score))

    def _remove_duplicates_and_validate(self, topics: List[NormalizedTopic]) -> List[NormalizedTopic]:
        """Remove duplicates and validate topic quality."""
        logger.info("Removing duplicates and validating...")
        
        # Remove duplicates based on canonical name similarity
        unique_topics = []
        seen_names = set()
        
        for topic in topics:
            # Simple duplicate detection
            name_key = topic.canonical_name.lower().strip()
            if name_key not in seen_names:
                # Quality validation
                if (topic.quality_score >= self.min_consensus_score - 0.1 and  # Slightly relaxed
                    topic.frequency_score >= self.min_frequency_score - 0.1 and
                    len(topic.canonical_name.strip()) > 2):
                    
                    seen_names.add(name_key)
                    unique_topics.append(topic)
        
        logger.info(f"Removed {len(topics) - len(unique_topics)} duplicates/low-quality topics")
        return unique_topics

    def _organize_topics(self, topics: List[NormalizedTopic]) -> Dict[str, Any]:
        """Organize topics by educational level and domain."""
        organized = {
            'by_level': defaultdict(list),
            'by_domain': defaultdict(list),
            'by_type': defaultdict(list)
        }
        
        for topic in topics:
            # By educational level
            for level in topic.educational_levels:
                organized['by_level'][level].append(asdict(topic))
            
            # By domain (classify based on topic name)
            domain = self._classify_domain(topic.canonical_name)
            organized['by_domain'][domain].append(asdict(topic))
            
            # By type
            organized['by_type'][topic.topic_type].append(asdict(topic))
        
        # Convert defaultdicts to regular dicts
        return {
            'by_level': dict(organized['by_level']),
            'by_domain': dict(organized['by_domain']),
            'by_type': dict(organized['by_type'])
        }

    def _calculate_normalization_metrics(self, raw_topics: List[Dict], normalized_topics: List[NormalizedTopic], start_time: float) -> NormalizationMetrics:
        """Calculate comprehensive metrics for the normalization process."""
        processing_time = time.time() - start_time
        
        # Basic counts
        total_raw = len(raw_topics)
        total_normalized = len(normalized_topics)
        reduction_ratio = (total_raw - total_normalized) / total_raw if total_raw > 0 else 0
        
        # Cross-level alignments
        cross_level_topics = [t for t in normalized_topics if len(t.educational_levels) > 1]
        cross_level_alignments = len(cross_level_topics)
        
        # Quality distribution
        quality_dist = {'high': 0, 'medium': 0, 'low': 0}
        consensus_scores = []
        
        for topic in normalized_topics:
            consensus_scores.append(topic.consensus_score)
            if topic.quality_score >= 0.8:
                quality_dist['high'] += 1
            elif topic.quality_score >= 0.6:
                quality_dist['medium'] += 1
            else:
                quality_dist['low'] += 1
        
        avg_consensus = np.mean(consensus_scores) if consensus_scores else 0
        
        # Coverage by level
        coverage_by_level = {}
        for topic in normalized_topics:
            for level in topic.educational_levels:
                coverage_by_level[level] = coverage_by_level.get(level, 0) + 1
        
        return NormalizationMetrics(
            total_raw_topics=total_raw,
            normalized_topics=total_normalized,
            reduction_ratio=reduction_ratio,
            cross_level_alignments=cross_level_alignments,
            semantic_clusters=len(set(t.semantic_cluster_id for t in normalized_topics if t.semantic_cluster_id >= 0)),
            avg_consensus_score=avg_consensus,
            coverage_by_level=coverage_by_level,
            processing_time=processing_time,
            quality_distribution=quality_dist,
            duplicate_removal_count=total_raw - total_normalized
        )

    def _generate_cache_key(self, discipline: str, language: str) -> str:
        """Generate cache key for normalization results."""
        return f"{discipline}_{language}_{self.similarity_threshold}_{self.cache_version}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load normalization results from cache."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check cache validity
            cache_time = datetime.fromisoformat(cached_data['normalization_timestamp'])
            if datetime.now() - cache_time > timedelta(hours=self.cache_ttl_hours):
                return None
            
            return cached_data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save normalization results to cache."""
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            converted_data = convert_numpy_types(data)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Normalize topics from extracted TOCs")
    parser.add_argument("--discipline", required=True, help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Similarity threshold for clustering")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached results")
    parser.add_argument("--openai-api-key", help="OpenAI API key for embeddings")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run normalization
    try:
        engine = TopicNormalizationEngine(
            similarity_threshold=args.similarity_threshold,
            openai_api_key=args.openai_api_key
        )
        
        result = engine.normalize_topics_from_tocs(
            discipline=args.discipline,
            language=args.language,
            force_refresh=args.force_refresh
        )
        
        # Print summary
        print(f"\nTopic Normalization Summary for {args.discipline} ({args.language}):")
        print(f"Raw topics: {result['preprocessing_summary']['raw_topics']}")
        print(f"Normalized topics: {result['total_normalized_topics']}")
        print(f"Reduction ratio: {result['metrics']['reduction_ratio']:.2%}")
        print(f"Cross-level alignments: {result['metrics']['cross_level_alignments']}")
        print(f"Semantic clusters: {result['metrics']['semantic_clusters']}")
        print(f"Average consensus score: {result['metrics']['avg_consensus_score']:.3f}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        
        # Quality distribution
        quality_dist = result['metrics']['quality_distribution']
        print(f"Quality distribution: High: {quality_dist['high']}, Medium: {quality_dist['medium']}, Low: {quality_dist['low']}")
        
        # Coverage by level
        coverage = result['metrics']['coverage_by_level']
        print(f"Coverage by level: {coverage}")
        
        print(f"\n✅ Results saved to: {OUTPUT_DIR / f'{args.discipline}_{args.language}_topics_normalized.json'}")
        
    except Exception as e:
        logger.error(f"Error during topic normalization: {e}")
        exit(1)

if __name__ == "__main__":
    main()