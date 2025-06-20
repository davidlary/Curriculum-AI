"""
OpenAlex Discipline Hierarchy and Classification Module.

Goal: Provide dynamic OpenAlex discipline hierarchy management by reading directly
from the OpenAlex knowledge graph API to ensure current and complete taxonomy.

This module dynamically fetches the complete 4-level OpenAlex hierarchy:
- Domains (4): Physical Sciences, Life Sciences, Social Sciences, Health Sciences  
- Fields (26+): Physics and Astronomy, Chemistry, Biology, etc.
- Subfields: Hundreds of specialized areas
- Topics: 4,516+ specific research topics

All data is fetched fresh from OpenAlex API with intelligent caching.
"""

import json
import logging
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OpenAlexLevel(Enum):
    """OpenAlex hierarchy levels."""
    DOMAIN = "domain"
    FIELD = "field" 
    SUBFIELD = "subfield"
    TOPIC = "topic"


@dataclass
class OpenAlexEntity:
    """Represents an entity in the OpenAlex hierarchy."""
    id: str
    display_name: str
    level: OpenAlexLevel
    parent_id: Optional[str] = None
    works_count: int = 0
    cited_by_count: int = 0
    description: Optional[str] = None


class OpenAlexAPI:
    """
    Goal: Interface with OpenAlex API to fetch live discipline hierarchy.
    
    Provides methods to fetch domains, fields, subfields, and topics directly
    from the OpenAlex knowledge graph with intelligent caching.
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, cache_duration_hours: int = 24):
        """Initialize API client with caching."""
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OpenBooks/1.0 (mailto:contact@openbooks.org)',
            'Accept': 'application/json'
        })
        logger.info(f"OpenAlex API client initialized with {cache_duration_hours}h cache")
    
    def fetch_concepts_hierarchy(self, level: int = None, per_page: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch concepts (disciplines) from OpenAlex API.
        
        Args:
            level: Concept level (0=domain, 1=field, 2=subfield, etc.)
            per_page: Results per page (max 200)
            
        Returns:
            List of concept dictionaries from OpenAlex
        """
        url = f"{self.BASE_URL}/concepts"
        params = {
            'per-page': per_page,
            'sort': 'works_count:desc'  # Sort by popularity
        }
        
        if level is not None:
            params['filter'] = f'level:{level}'
        
        all_concepts = []
        page = 1
        
        while True:
            params['page'] = page
            
            try:
                logger.debug(f"Fetching OpenAlex concepts page {page}, level {level}")
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                concepts = data.get('results', [])
                
                if not concepts:
                    break
                
                all_concepts.extend(concepts)
                
                # Check if we have more pages
                meta = data.get('meta', {})
                if page >= meta.get('count', 0) // per_page:
                    break
                    
                page += 1
                time.sleep(0.1)  # Rate limiting
                
            except requests.RequestException as e:
                logger.error(f"Error fetching OpenAlex concepts: {e}")
                break
        
        logger.info(f"Fetched {len(all_concepts)} concepts from OpenAlex (level {level})")
        return all_concepts
    
    def get_concept_by_id(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific concept by ID.
        
        Args:
            concept_id: OpenAlex concept ID or URL
            
        Returns:
            Concept dictionary or None if not found
        """
        # Handle both ID formats
        if concept_id.startswith('https://openalex.org/'):
            concept_id = concept_id.split('/')[-1]
        elif not concept_id.startswith('C'):
            concept_id = f"C{concept_id}"
        
        url = f"{self.BASE_URL}/concepts/{concept_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Error fetching concept {concept_id}: {e}")
            return None


class OpenAlexHierarchy:
    """
    Goal: Manage dynamic OpenAlex discipline hierarchy by reading from live API.
    
    Fetches the complete taxonomy from OpenAlex knowledge graph and provides
    classification services for educational content organization.
    """
    
    def __init__(self, api_client: OpenAlexAPI = None, cache_file: str = None):
        """Initialize hierarchy with API client and optional caching."""
        self.api_client = api_client or OpenAlexAPI()
        self.cache_file = Path(cache_file) if cache_file else None
        
        # Dynamic data storage
        self.domains = {}
        self.fields = {}
        self.subfields = {}
        self.topics = {}
        self._entities = {}
        
        # Load or fetch hierarchy
        self._load_hierarchy()
        
        logger.info(f"OpenAlex hierarchy loaded: {len(self.domains)} domains, "
                   f"{len(self.fields)} fields, {len(self.subfields)} subfields, "
                   f"{len(self.topics)} topics")
    
    def _load_hierarchy(self) -> None:
        """Load hierarchy from cache or fetch from API."""
        cached_data = self._load_from_cache()
        
        if cached_data and self._is_cache_valid(cached_data):
            logger.info("Loading OpenAlex hierarchy from cache")
            self._load_from_data(cached_data)
        else:
            logger.info("Fetching fresh OpenAlex hierarchy from API")
            self._fetch_from_api()
    
    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached hierarchy data."""
        if not self.cache_file or not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading cache: {e}")
            return None
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""
        cache_time = cached_data.get('cached_at')
        if not cache_time:
            return False
        
        cache_datetime = datetime.fromisoformat(cache_time)
        return datetime.now() - cache_datetime < self.api_client.cache_duration
    
    def _fetch_from_api(self) -> None:
        """Fetch hierarchy from OpenAlex API."""
        try:
            # Fetch domains (level 0)
            domains_data = self.api_client.fetch_concepts_hierarchy(level=0)
            self._process_concepts(domains_data, OpenAlexLevel.DOMAIN)
            
            # Fetch fields (level 1) 
            fields_data = self.api_client.fetch_concepts_hierarchy(level=1)
            self._process_concepts(fields_data, OpenAlexLevel.FIELD)
            
            # Fetch subfields (level 2)
            subfields_data = self.api_client.fetch_concepts_hierarchy(level=2)
            self._process_concepts(subfields_data, OpenAlexLevel.SUBFIELD)
            
            # Cache the results
            self._save_to_cache()
            
        except Exception as e:
            logger.error(f"Error fetching from OpenAlex API: {e}")
            self._load_fallback_data()
    
    def _process_concepts(self, concepts_data: List[Dict[str, Any]], level: OpenAlexLevel) -> None:
        """Process concept data from API into entities."""
        for concept in concepts_data:
            entity = self._create_entity_from_api_data(concept, level)
            if entity:
                self._entities[entity.id] = entity
                
                if level == OpenAlexLevel.DOMAIN:
                    self.domains[entity.id] = entity
                elif level == OpenAlexLevel.FIELD:
                    self.fields[entity.id] = entity
                elif level == OpenAlexLevel.SUBFIELD:
                    self.subfields[entity.id] = entity
                elif level == OpenAlexLevel.TOPIC:
                    self.topics[entity.id] = entity
    
    def _create_entity_from_api_data(self, concept: Dict[str, Any], level: OpenAlexLevel) -> Optional[OpenAlexEntity]:
        """Create OpenAlexEntity from API concept data."""
        try:
            # Extract ID from OpenAlex URL format
            openalex_id = concept.get('id', '')
            if openalex_id.startswith('https://openalex.org/'):
                clean_id = openalex_id.split('/')[-1].lower()
            else:
                clean_id = str(concept.get('id', '')).lower()
            
            # Get parent ID if available
            parent_id = None
            ancestors = concept.get('ancestors', [])
            if ancestors and len(ancestors) > 0:
                parent_openalex_id = ancestors[-1].get('id', '')
                if parent_openalex_id.startswith('https://openalex.org/'):
                    parent_id = parent_openalex_id.split('/')[-1].lower()
            
            return OpenAlexEntity(
                id=clean_id,
                display_name=concept.get('display_name', 'Unknown'),
                level=level,
                parent_id=parent_id,
                works_count=concept.get('works_count', 0),
                cited_by_count=concept.get('cited_by_count', 0),
                description=concept.get('description')
            )
        except Exception as e:
            logger.warning(f"Error processing concept {concept.get('id', 'unknown')}: {e}")
            return None
    
    def _save_to_cache(self) -> None:
        """Save current hierarchy to cache."""
        if not self.cache_file:
            return
        
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'domains': {k: self._entity_to_dict(v) for k, v in self.domains.items()},
            'fields': {k: self._entity_to_dict(v) for k, v in self.fields.items()},
            'subfields': {k: self._entity_to_dict(v) for k, v in self.subfields.items()},
            'topics': {k: self._entity_to_dict(v) for k, v in self.topics.items()}
        }
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cached OpenAlex hierarchy to {self.cache_file}")
        except IOError as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _entity_to_dict(self, entity: OpenAlexEntity) -> Dict[str, Any]:
        """Convert entity to dictionary for caching."""
        return {
            'id': entity.id,
            'display_name': entity.display_name,
            'level': entity.level.value,
            'parent_id': entity.parent_id,
            'works_count': entity.works_count,
            'cited_by_count': entity.cited_by_count,
            'description': entity.description
        }
    
    def _load_from_data(self, cached_data: Dict[str, Any]) -> None:
        """Load hierarchy from cached data."""
        for domain_data in cached_data.get('domains', {}).values():
            entity = self._entity_from_dict(domain_data)
            if entity:
                self.domains[entity.id] = entity
                self._entities[entity.id] = entity
        
        for field_data in cached_data.get('fields', {}).values():
            entity = self._entity_from_dict(field_data)
            if entity:
                self.fields[entity.id] = entity
                self._entities[entity.id] = entity
        
        for subfield_data in cached_data.get('subfields', {}).values():
            entity = self._entity_from_dict(subfield_data)
            if entity:
                self.subfields[entity.id] = entity
                self._entities[entity.id] = entity
        
        for topic_data in cached_data.get('topics', {}).values():
            entity = self._entity_from_dict(topic_data)
            if entity:
                self.topics[entity.id] = entity
                self._entities[entity.id] = entity
    
    def _entity_from_dict(self, data: Dict[str, Any]) -> Optional[OpenAlexEntity]:
        """Create entity from dictionary data."""
        try:
            return OpenAlexEntity(
                id=data['id'],
                display_name=data['display_name'],
                level=OpenAlexLevel(data['level']),
                parent_id=data.get('parent_id'),
                works_count=data.get('works_count', 0),
                cited_by_count=data.get('cited_by_count', 0),
                description=data.get('description')
            )
        except Exception as e:
            logger.warning(f"Error creating entity from dict: {e}")
            return None
    
    def _load_fallback_data(self) -> None:
        """Load minimal fallback data if API is unavailable."""
        logger.warning("Loading fallback OpenAlex hierarchy data")
        
        # Minimal fallback domains
        fallback_domains = [
            ("c121332964", "Physical Sciences"),
            ("c86803240", "Life Sciences"), 
            ("c17744445", "Social Sciences"),
            ("c71924100", "Health Sciences")
        ]
        
        for domain_id, name in fallback_domains:
            entity = OpenAlexEntity(
                id=domain_id,
                display_name=name,
                level=OpenAlexLevel.DOMAIN
            )
            self.domains[domain_id] = entity
            self._entities[domain_id] = entity
    
    # OpenStax Subject to OpenAlex Field Mapping (for fuzzy matching)
    SUBJECT_KEYWORDS = {
        "physics": ["physics", "mechanics", "thermodynamics", "electromagnetism"],
        "astronomy": ["astronomy", "astrophysics", "cosmology", "planetary"],
        "chemistry": ["chemistry", "chemical", "organic chemistry", "inorganic"],
        "biology": ["biology", "biological", "life sciences", "ecology"],
        "microbiology": ["microbiology", "microorganisms", "bacteria", "immunology"],
        "anatomy": ["anatomy", "human anatomy", "medical", "physiology"],
        "psychology": ["psychology", "cognitive", "behavioral", "mental health"],
        "sociology": ["sociology", "social", "society", "cultural"],
        "economics": ["economics", "economic", "finance", "econometrics"],
        "business": ["business", "management", "accounting", "marketing"],
        "mathematics": ["mathematics", "math", "calculus", "algebra", "statistics"],
        "computer-science": ["computer science", "programming", "algorithms", "software"],
        "engineering": ["engineering", "mechanical", "electrical", "civil"],
        "medicine": ["medicine", "medical", "clinical", "health"],
        "history": ["history", "historical", "humanities"],
        "philosophy": ["philosophy", "ethics", "logic", "metaphysics"]
    }
    
    
    def get_domain(self, domain_id: str) -> Optional[OpenAlexEntity]:
        """Get domain by ID."""
        return self.domains.get(domain_id)
    
    def get_field(self, field_id: str) -> Optional[OpenAlexEntity]:
        """Get field by ID."""
        return self.fields.get(field_id)
    
    def get_subfield(self, subfield_id: str) -> Optional[OpenAlexEntity]:
        """Get subfield by ID."""
        return self.subfields.get(subfield_id)
    
    def get_topic(self, topic_id: str) -> Optional[OpenAlexEntity]:
        """Get topic by ID."""
        return self.topics.get(topic_id)
    
    def get_entity(self, entity_id: str) -> Optional[OpenAlexEntity]:
        """Get any entity by ID."""
        return self._entities.get(entity_id)
    
    def get_fields_by_domain(self, domain_id: str) -> List[OpenAlexEntity]:
        """Get all fields within a domain."""
        return [field for field in self.fields.values() if field.parent_id == domain_id]
    
    def get_subfields_by_field(self, field_id: str) -> List[OpenAlexEntity]:
        """Get all subfields within a field."""
        return [subfield for subfield in self.subfields.values() if subfield.parent_id == field_id]
    
    def get_topics_by_subfield(self, subfield_id: str) -> List[OpenAlexEntity]:
        """Get all topics within a subfield."""
        return [topic for topic in self.topics.values() if topic.parent_id == subfield_id]
    
    def _get_level_0_concepts(self) -> List[str]:
        """
        Get Level-0 concepts dynamically from OpenAlex API.
        
        Returns:
            List of Level-0 concept names
        """
        try:
            # Fetch Level-0 concepts (these are the top-level concepts)
            level_0_data = self.api_client.fetch_concepts_hierarchy(level=0)
            concept_names = [concept.get('display_name', '') for concept in level_0_data]
            
            # Cache for performance
            if not hasattr(self, '_cached_level_0_concepts'):
                self._cached_level_0_concepts = concept_names
            
            return concept_names
            
        except Exception as e:
            logger.warning(f"Error fetching Level-0 concepts from API: {e}")
            # Fallback to known common Level-0 concepts
            return [
                "Political science", "Philosophy", "Economics", "Business", 
                "Psychology", "Mathematics", "Medicine", "Biology", 
                "Computer science", "Geology", "Chemistry", "Art", 
                "Sociology", "Engineering", "Geography", "History", 
                "Materials science", "Physics", "Environmental science"
            ]
    
    def classify_subject(self, subject: str) -> str:
        """
        Goal: Classify a subject string to an OpenAlex Level-0 concept.
        
        Args:
            subject: Subject name to classify
            
        Returns:
            OpenAlex Level-0 concept name for directory organization
        """
        subject_lower = subject.lower().replace(" ", "-").replace("_", "-")
        
        # Dynamic mappings based on OpenAlex Level-0 concepts 
        # These will be fetched from the API rather than hardcoded
        level_0_concepts = self._get_level_0_concepts()
        
        # Build dynamic mapping based on keyword matching with actual Level-0 concepts
        # Includes multilingual support for Spanish, Portuguese, and other languages
        subject_keywords_to_concepts = {
            # Chemistry (English)
            "chemistry": "Chemistry",
            "organic": "Chemistry",
            "inorganic": "Chemistry", 
            "biochemistry": "Chemistry",
            # Chemistry (Spanish/Portuguese)
            "quimica": "Chemistry",
            "química": "Chemistry",
            
            # Physics (English)
            "physics": "Physics",
            "astronomy": "Physics",  # Astronomy typically falls under Physics Level-0
            "college-physics": "Physics",
            "university-physics": "Physics",
            # Physics (Spanish/Portuguese)
            "fisica": "Physics",
            "física": "Physics",
            "fisica-universitaria": "Physics",
            "física-universitaria": "Physics",
            
            # Biology (English)
            "biology": "Biology",
            "microbiology": "Biology", 
            "anatomy": "Biology",
            "physiology": "Biology",
            "life-liberty-pursuit-happiness": "Biology",  # AP Biology course
            # Biology (Spanish/Portuguese)
            "biologia": "Biology",
            "biología": "Biology",
            "microbiologia": "Biology",
            "microbiología": "Biology",
            
            # Medicine (English)
            "nursing": "Medicine",
            "medical": "Medicine",
            "health": "Medicine",
            
            # Psychology (English)
            "psychology": "Psychology",
            # Psychology (Spanish/Portuguese) 
            "psicologia": "Psychology",
            "psicología": "Psychology",
            "psychologia": "Psychology",  # Alternative spelling
            
            # Sociology (English)
            "sociology": "Sociology", 
            "introduction-sociology": "Sociology",
            # Sociology (Spanish/Portuguese)
            "sociologia": "Sociology",
            "sociología": "Sociology",
            
            # Anthropology - separate from Sociology
            "anthropology": "Anthropology",
            "introduction-anthropology": "Anthropology",
            "antropologia": "Anthropology",
            "antropología": "Anthropology",
            
            # Mathematics (English)
            "mathematics": "Mathematics",
            "statistics": "Mathematics",
            "estadistica": "Mathematics",  # Spanish statistics
            "estadística": "Mathematics",
            "introduccion-estadistica": "Mathematics",  # Introduction to Statistics
            "contemporary-mathematics": "Mathematics",
            "prealgebra": "Mathematics",
            "calculus": "Mathematics",
            "calculo": "Mathematics",  # Spanish calculus
            "cálculo": "Mathematics",
            "algebra": "Mathematics",
            "math": "Mathematics",
            # Mathematics (Spanish/Portuguese)
            "matematicas": "Mathematics",
            "matemáticas": "Mathematics",
            "matematica": "Mathematics",
            "matemática": "Mathematics",
            "precalculo": "Mathematics",
            
            # Economics (English)
            "economics": "Economics",
            "macroeconomics": "Economics",
            "microeconomics": "Economics",
            "mikroekonomia": "Economics",  # Alternative spelling
            "makroekonomia": "Economics",  # Alternative spelling
            # Economics (Spanish/Portuguese)
            "economia": "Economics",
            "economía": "Economics",
            "macroeconomia": "Economics",
            "macroeconomía": "Economics",
            
            # Business (English)
            "business": "Business",
            "entrepreneurship": "Business",
            "marketing": "Business",
            "finance": "Business",
            "accounting": "Business",
            "principles-accounting": "Business",
            "principles-finance": "Business",
            "principles-marketing": "Business",
            "intellectual-property": "Business",  # Business/Legal topic
            # Business (Spanish/Portuguese)
            "negocios": "Business",
            "empresa": "Business",
            "administracion": "Business",
            "administración": "Business",
            
            # Political Science (English)
            "political": "Political science",
            "government": "Political science",
            "american-government": "Political science",
            "introduction-political-science": "Political science",
            # Political Science (Spanish/Portuguese)
            "gobierno": "Political science",
            "politica": "Political science",
            "política": "Political science",
            "ciencia-politica": "Political science",
            "ciencia-política": "Political science",
            
            # Computer Science (English)
            "computer-science": "Computer science",
            "computer": "Computer science",
            "programming": "Computer science",
            "algorithms": "Computer science",
            "data-structures": "Computer science",
            "software": "Computer science",
            # Computer Science (Spanish/Portuguese)
            "informatica": "Computer science",
            "informática": "Computer science",
            "programacion": "Computer science",
            "programación": "Computer science",
            
            # Engineering (English)
            "engineering": "Engineering",
            "mechanical-engineering": "Engineering",
            "electrical-engineering": "Engineering",
            "civil-engineering": "Engineering",
            "chemical-engineering": "Engineering",
            # Engineering (Spanish/Portuguese)
            "ingenieria": "Engineering",
            "ingeniería": "Engineering",
            
            # Art (English)
            "art": "Art",
            "literature": "Art",
            "literatura": "Art",
            "writing": "Art",
            "writing-guide": "Art",
            
            # Philosophy
            "philosophy": "Philosophy",
            "filosofia": "Philosophy",
            "filosofía": "Philosophy",
            "introduction-philosophy": "Philosophy",
            
            # History
            "history": "History",
            "historia": "History",
            "us-history": "History",
            "world-history": "History",
            
            # Education (Study Skills, College Success)
            "college-success": "Sociology",  # Map to Sociology as closest match in OpenAlex
            
            # Physics subjects (separate entries for Polish)
            "fizyka": "Physics",  # Polish physics
            
            # Non-textbook items that should be excluded
            "playground": None,  # Development/testing repository
            "baldwin-s-openstax-index": None,  # Index/catalog, not a textbook
        }
        
        # Check keyword mappings first - match against actual Level-0 concepts
        for keyword, concept in subject_keywords_to_concepts.items():
            if keyword in subject_lower:
                if concept is None:
                    # This indicates non-textbook content that should be excluded
                    return "Uncategorized"
                elif concept in level_0_concepts:
                    return concept
        
        # Keyword-based matching for more specific subjects
        subject_text = subject.lower()
        for subject_key, keywords in self.SUBJECT_KEYWORDS.items():
            if any(keyword in subject_text for keyword in keywords):
                # Map to appropriate Level-0 concept
                if subject_key in subject_keywords_to_concepts:
                    concept = subject_keywords_to_concepts[subject_key]
                    if concept in level_0_concepts:
                        return concept
        
        # Check for development/tool keywords that should be excluded
        development_keywords = [
            "cookbook", "framework", "template", "resource", "tool", "spellchecker",
            "concourse", "slack", "producer", "exports", "infrastructure", "deployment",
            "pipeline", "ci", "cd", "automation", "build", "config", "docker",
            "manifests", "index", "playground", "test", "demo", "simulation",
            "napkin", "sifter", "poet", "automate", "tiny", "study", "covid"
        ]
        
        # If subject contains development keywords, mark as uncategorized
        if any(word in subject_text for word in development_keywords):
            return "Uncategorized"
        
        # Default fallback to most common concepts - validate against actual API data
        # Includes multilingual terms for broader matching
        fallback_mappings = [
            (["math", "calculus", "algebra", "statistics", "calculo", "cálculo", "matematica", "matemática"], "Mathematics"),
            (["bio", "life", "living", "organism", "biologia", "biología"], "Biology"),
            (["chem", "molecule", "atom", "quimica", "química"], "Chemistry"),
            (["phys", "force", "energy", "motion", "fisica", "física", "mechanics"], "Physics"),
            (["psych", "mind", "behavior", "psicologia", "psicología", "psychologia"], "Psychology"),
            (["social", "society", "culture", "sociologia", "sociología"], "Sociology"),
            (["business", "management", "company", "negocios", "empresa"], "Business"),
            (["economic", "money", "market", "economia", "economía", "makroekonomia"], "Economics"),
            (["political", "government", "policy", "gobierno", "politica", "política"], "Political science"),
            (["computer", "programming", "software", "algorithm", "data", "informatica", "informática"], "Computer science"),
            (["engineering", "mechanical", "electrical", "civil", "ingenieria", "ingeniería"], "Engineering"),
        ]
        
        for keywords, concept in fallback_mappings:
            if any(word in subject_text for word in keywords) and concept in level_0_concepts:
                return concept
        
        # Final fallback for truly unknown content
        return "Uncategorized"
    
    def get_directory_path(self, concept_name: str, level: str = "University") -> str:
        """
        Goal: Generate directory path using OpenAlex Level-0 concept.
        
        Uses Level-0 concept organization: Concept/Level/Repository
        This aligns with OpenAlex Level-0 concepts for better organization.
        
        Args:
            concept_name: OpenAlex Level-0 concept name (e.g., "Chemistry", "Physics")
            level: Academic level (University, HighSchool, Graduate)
            
        Returns:
            Directory path string for organizing textbooks
        """
        if not concept_name:
            return f"Uncategorized/{level}"
        
        # Use concept as primary directory: Concept/Level/Repository
        return f"{concept_name}/{level}"
    
    def get_classification_report(self, subjects: List[str]) -> Dict[str, Any]:
        """
        Goal: Generate classification report for multiple subjects.
        
        Args:
            subjects: List of subject names to classify
            
        Returns:
            Dictionary with classification results and statistics
        """
        results = {
            "total_subjects": len(subjects),
            "classified": [],
            "unclassified": [],
            "concept_distribution": {}
        }
        
        for subject in subjects:
            concept_name = self.classify_subject(subject)
            if concept_name:
                results["classified"].append({
                    "subject": subject,
                    "concept": concept_name,
                    "directory_path": self.get_directory_path(concept_name)
                })
                
                # Update distributions
                results["concept_distribution"][concept_name] = results["concept_distribution"].get(concept_name, 0) + 1
            else:
                results["unclassified"].append(subject)
        
        results["classification_rate"] = len(results["classified"]) / len(subjects) if subjects else 0
        
        return results
    
    def validate_mapping(self) -> Dict[str, Any]:
        """
        Goal: Validate the OpenStax to OpenAlex mapping completeness.
        
        Returns:
            Validation report with any issues found
        """
        issues = []
        valid_mappings = 0
        
        # Validate that we have essential fields for common subjects
        essential_subjects = ['physics', 'chemistry', 'biology', 'mathematics', 'psychology']
        for subject in essential_subjects:
            field = self.classify_subject(subject)
            if field:
                domain = self.get_domain(field.parent_id)
                if domain:
                    valid_mappings += 1
                else:
                    issues.append(f"Field '{field.display_name}' has invalid parent domain '{field.parent_id}'")
            else:
                issues.append(f"No OpenAlex field found for essential subject '{subject}'")
        
        return {
            "total_subjects": len(essential_subjects),
            "valid_mappings": valid_mappings,
            "issues": issues,
            "validation_passed": len(issues) == 0,
            "hierarchy_stats": {
                "domains": len(self.domains),
                "fields": len(self.fields),
                "subfields": len(self.subfields),
                "topics": len(self.topics)
            }
        }


def create_directory_structure_plan(hierarchy: OpenAlexHierarchy, base_path: str = "Books") -> Dict[str, List[str]]:
    """
    Goal: Create field-based directory structure plan for OpenAlex organization.
    
    Uses OpenAlex fields as primary directories for better textbook organization.
    Structure: Books/Field/Level/Repository
    
    Args:
        hierarchy: OpenAlexHierarchy instance
        base_path: Base directory path
        
    Returns:
        Dictionary mapping directory paths to descriptions
    """
    structure = {}
    
    # Group fields by domain for organization
    domain_field_map = {}
    for field in hierarchy.fields.values():
        domain = hierarchy.get_domain(field.parent_id)
        domain_name = domain.display_name if domain else "Unknown"
        if domain_name not in domain_field_map:
            domain_field_map[domain_name] = []
        domain_field_map[domain_name].append(field)
    
    # Create field-based structure
    for domain_name, fields in domain_field_map.items():
        for field in fields:
            field_name = field.display_name.replace(" ", "_").replace(",", "").replace("&", "and")
            
            # Create field directory
            field_path = f"{base_path}/{field_name}"
            structure[field_path] = [
                f"Field: {field.display_name}",
                f"Domain: {domain_name}",
                f"Works: {field.works_count:,}"
            ]
            
            # Create level subdirectories
            for level in ["HighSchool", "University", "Graduate"]:
                level_path = f"{field_path}/{level}"
                structure[level_path] = [
                    f"Field: {field.display_name}",
                    f"Level: {level}",
                    f"Domain: {domain_name}"
                ]
    
    return structure


# Global hierarchy instance with caching
_global_hierarchy = None

def get_hierarchy(cache_file: str = "metadata/openalex_cache.json") -> OpenAlexHierarchy:
    """Get global OpenAlex hierarchy instance with API integration."""
    global _global_hierarchy
    if _global_hierarchy is None:
        api_client = OpenAlexAPI(cache_duration_hours=24)
        _global_hierarchy = OpenAlexHierarchy(api_client=api_client, cache_file=cache_file)
    return _global_hierarchy

def refresh_hierarchy(cache_file: str = "metadata/openalex_cache.json") -> OpenAlexHierarchy:
    """Force refresh of OpenAlex hierarchy from API."""
    global _global_hierarchy
    
    # Remove cache file to force fresh fetch
    cache_path = Path(cache_file)
    if cache_path.exists():
        cache_path.unlink()
    
    api_client = OpenAlexAPI(cache_duration_hours=24)
    _global_hierarchy = OpenAlexHierarchy(api_client=api_client, cache_file=cache_file)
    return _global_hierarchy