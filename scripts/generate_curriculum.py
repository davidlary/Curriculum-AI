import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, TypedDict
import logging

# Import the comprehensive curriculum systems
try:
    from pedagogical_ordering_system import PedagogicalOrderingSystem
    from comprehensive_curriculum_system import ComprehensiveCurriculumSystem
    from adaptive_curriculum_system import AdaptiveJSONParser, DataDrivenCurriculumBuilder
    from toc_aware_curriculum_system import TOCAwareCurriculumSystem
except ImportError:
    # Fallback if modules not found
    PedagogicalOrderingSystem = None
    ComprehensiveCurriculumSystem = None
    AdaptiveJSONParser = None
    DataDrivenCurriculumBuilder = None
    TOCAwareCurriculumSystem = None

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('curriculum_generation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import OpenAI
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser, ResponseSchema
    from langchain.schema import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.info("Install with: pip install langchain-openai langchain-community langgraph networkx matplotlib")
    exit(1)

CHUNKS_DIR = Path("Chunks")
CURRICULUM_DIR = Path("Curriculum")
CURRICULUM_DIR.mkdir(exist_ok=True)

class CurriculumState(TypedDict):
    """State for the curriculum generation workflow."""
    discipline: str
    chunks: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    subtopics: List[Dict[str, Any]]
    prerequisites: Dict[str, List[str]]
    curriculum_graph: Dict[str, Any]
    error: str

class CurriculumGenerator:
    def __init__(self, openai_api_key: str = None, provider: str = "openai"):
        """Initialize the curriculum generator with AI models."""
        # Check environment variables for API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")
        
        self.provider = provider.lower()
        
        # Initialize adaptive components
        self.adaptive_parser = AdaptiveJSONParser() if AdaptiveJSONParser else None
        self.adaptive_builder = DataDrivenCurriculumBuilder() if DataDrivenCurriculumBuilder else None
        
        # Initialize models based on provider
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
                api_key=self.openai_api_key
            )
            self.fast_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=self.openai_api_key
            )
        elif self.provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.3,
                    anthropic_api_key=self.anthropic_api_key
                )
                self.fast_llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.1,
                    anthropic_api_key=self.anthropic_api_key
                )
            except ImportError:
                logger.error("langchain_anthropic not installed. Install with: pip install langchain-anthropic")
                raise
        elif self.provider == "xai":
            if not self.xai_api_key:
                raise ValueError("XAI API key required. Set XAI_API_KEY environment variable.")
            # XAI uses OpenAI-compatible API
            self.llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.3,
                api_key=self.xai_api_key,
                base_url="https://api.x.ai/v1"
            )
            self.fast_llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.1,
                api_key=self.xai_api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, xai")
        
        # Create the workflow graph
        self.workflow = self.create_workflow()
        
    def load_chunks(self, discipline: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL files for a specific discipline with TOC enhancement."""
        chunks = []
        chunk_files = list(CHUNKS_DIR.glob(f"{discipline}_*.jsonl"))
        
        if not chunk_files:
            logger.warning(f"No chunk files found for discipline: {discipline}")
            return chunks
        
        # First, extract actual TOC data from XML collection files
        toc_data = self._extract_all_toc_data(discipline)
        
        for chunk_file in chunk_files:
            logger.info(f"📚 Loading chunks from: {chunk_file}")
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            chunk = json.loads(line.strip())
                            
                            # Enhance chunk with actual TOC information
                            chunk = self._enhance_chunk_with_toc(chunk, toc_data)
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num} in {chunk_file}: {e}")
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
                
        logger.info(f"Loaded {len(chunks)} chunks for discipline: {discipline}")
        return chunks
    
    def _extract_all_toc_data(self, discipline: str) -> Dict[str, Dict]:
        """Extract Table of Contents data from all XML collection files."""
        import xml.etree.ElementTree as ET
        
        toc_data = {}
        books_path = Path("Books/english") / discipline
        
        if not books_path.exists():
            logger.warning(f"Books path not found: {books_path}")
            return toc_data
        
        logger.info(f"🔍 Extracting TOC data from {discipline} books...")
        
        # Find all collection XML files
        for level_dir in books_path.iterdir():
            if not level_dir.is_dir():
                continue
                
            for book_dir in level_dir.iterdir():
                if not book_dir.is_dir():
                    continue
                    
                collections_dir = book_dir / "collections"
                if collections_dir.exists():
                    for xml_file in collections_dir.glob("*.xml"):
                        try:
                            tree = ET.parse(xml_file)
                            root = tree.getroot()
                            
                            # Define namespace
                            ns = {'col': 'http://cnx.rice.edu/collxml', 'md': 'http://cnx.rice.edu/mdml'}
                            
                            # Get book title
                            title_elem = root.find('.//md:title', ns)
                            book_title = title_elem.text if title_elem is not None else xml_file.stem
                            
                            # Extract chapter structure
                            chapters = []
                            for subcol in root.findall('.//col:subcollection', ns):
                                title_elem = subcol.find('md:title', ns)
                                if title_elem is not None:
                                    chapter_title = title_elem.text.strip()
                                    
                                    # Extract modules (sections) within this chapter
                                    sections = []
                                    for module in subcol.findall('.//col:module', ns):
                                        doc_attr = module.get('document')
                                        if doc_attr:
                                            sections.append(doc_attr)
                                    
                                    chapters.append({
                                        'title': chapter_title,
                                        'sections': sections
                                    })
                            
                            # Determine educational level
                            level = "undergraduate"  # default
                            if "HighSchool" in str(book_dir) or "high" in str(book_dir).lower():
                                level = "high_school"
                            elif "Graduate" in str(book_dir) or "graduate" in str(book_dir).lower():
                                level = "graduate"
                            
                            toc_data[book_dir.name] = {
                                'title': book_title,
                                'level': level,
                                'chapters': chapters,
                                'path': str(xml_file)
                            }
                            
                            logger.info(f"📖 Extracted TOC from {book_title}: {len(chapters)} chapters")
                            
                        except Exception as e:
                            logger.warning(f"Error extracting TOC from {xml_file}: {e}")
        
        return toc_data
    
    def _enhance_chunk_with_toc(self, chunk: Dict, toc_data: Dict) -> Dict:
        """Enhance chunk with actual TOC information."""
        book_title = chunk.get('book_title', '')
        
        # Find matching TOC data
        matching_toc = None
        for book_key, toc_info in toc_data.items():
            if book_key in book_title or book_title in book_key:
                matching_toc = toc_info
                break
        
        if matching_toc:
            # Try to find the actual chapter title for this chunk
            for chapter in matching_toc['chapters']:
                chapter_title = chapter['title']
                
                # If chunk content contains references to chapter concepts, use the chapter title
                chunk_text = chunk.get('text', '').lower()
                chapter_lower = chapter_title.lower()
                
                # Look for chapter title keywords in the chunk
                chapter_words = [word for word in chapter_lower.split() if len(word) > 3]
                if any(word in chunk_text for word in chapter_words[:3]):  # Match first 3 significant words
                    chunk['actual_chapter_title'] = chapter_title
                    chunk['educational_level'] = matching_toc['level']
                    break
            
            # If no specific chapter match, use book-level information
            if 'actual_chapter_title' not in chunk:
                chunk['actual_chapter_title'] = chunk.get('chapter_title', 'General Topics')
                chunk['educational_level'] = matching_toc['level']
        
        return chunk

    def load_book_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load book metadata to understand educational levels and core/elective status."""
        book_metadata = {}
        
        # Load from BookList.json
        book_list_file = Path("Books/BookList.json")
        if book_list_file.exists():
            try:
                with open(book_list_file, 'r', encoding='utf-8') as f:
                    books = json.load(f)
                
                for book in books:
                    book_key = book.get('book_id', '')
                    book_metadata[book_key] = {
                        'name': book.get('book_name', ''),
                        'language': book.get('language', 'english'),
                        'discipline': book.get('discipline', ''),
                        'level': book.get('level', 'University'),
                        'format': book.get('format', ''),
                        'is_core': self._determine_core_status(book.get('book_name', ''), book.get('discipline', '')),
                        'educational_level': self._map_level_to_standard(book.get('level', 'University'))
                    }
                    
                logger.info(f"Loaded metadata for {len(book_metadata)} books")
                    
            except Exception as e:
                logger.error(f"Error loading book metadata: {e}")
        
        return book_metadata
    
    def _determine_core_status(self, book_name: str, discipline: str) -> bool:
        """Determine if a book covers core or elective material."""
        book_name_lower = book_name.lower()
        discipline_lower = discipline.lower()
        
        # Core physics topics
        if discipline_lower == 'physics':
            core_indicators = [
                'mechanics', 'thermodynamics', 'electromagnetism', 'waves', 'optics',
                'university physics', 'college physics', 'physics', 'introductory',
                'general physics', 'principles', 'fundamentals'
            ]
            elective_indicators = [
                'astronomy', 'astrophysics', 'biophysics', 'geophysics', 'medical physics',
                'plasma', 'nuclear', 'particle', 'condensed matter', 'quantum field theory',
                'advanced', 'specialized', 'research', 'graduate'
            ]
        else:
            # Default patterns for other disciplines
            core_indicators = [
                'introduction', 'principles', 'fundamentals', 'general', 'basic',
                'college', 'university', 'essential'
            ]
            elective_indicators = [
                'advanced', 'specialized', 'applied', 'research', 'graduate',
                'topics', 'special', 'current'
            ]
        
        # Check for core indicators
        if any(indicator in book_name_lower for indicator in core_indicators):
            return True
        
        # Check for elective indicators
        if any(indicator in book_name_lower for indicator in elective_indicators):
            return False
        
        # Default to core for introductory level materials
        return True
    
    def _map_level_to_standard(self, level: str) -> str:
        """Map book level to standard educational levels."""
        level_lower = level.lower()
        
        if level_lower in ['highschool', 'high_school', 'secondary']:
            return 'high_school'
        elif level_lower in ['university', 'college', 'undergraduate']:
            return 'undergraduate'
        elif level_lower in ['graduate', 'postgraduate', 'masters', 'phd']:
            return 'graduate'
        else:
            return 'undergraduate'  # Default
    
    def extract_topics(self, state: CurriculumState) -> CurriculumState:
        """Extract main topics from chunks using LLM."""
        logger.info("Extracting main topics from chunks...")
        
        # Sample chunks for topic extraction - use more comprehensive sampling
        total_chunks = len(state["chunks"])
        if total_chunks > 100:
            # Sample from throughout the entire collection for comprehensive coverage
            sample_indices = list(range(0, total_chunks, total_chunks // 100))[:100]
            sample_chunks = [state["chunks"][i] for i in sample_indices]
        else:
            sample_chunks = state["chunks"]
        
        # Extract more text per chunk for better topic identification
        chunk_texts = [chunk["text"][:800] for chunk in sample_chunks if chunk.get("text")]
        
        topic_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert physics curriculum designer. Extract the main physics domains from comprehensive textbook content spanning high school through university levels.
            
            For each major physics domain, provide:
            1. Domain name (clear and concise, e.g., "Classical Mechanics", "Electromagnetism")
            2. Description (2-3 sentences explaining the domain scope)
            3. Estimated difficulty level (1-5, where 1=high school, 3=undergraduate, 5=advanced)
            
            Extract 15-20 major physics domains that comprehensively cover the entire discipline from foundational concepts to graduate specializations.
            Include core domains: Classical Mechanics, Thermodynamics, Electromagnetism, Waves & Optics, Modern Physics, Quantum Mechanics.
            Include specialized tracks: Engineering Physics, Biophysics, Geophysics, Medical Physics, Computational Physics.
            Include advanced areas: Theoretical Physics, Plasma Physics, and emerging interdisciplinary fields.
            
            IMPORTANT: Ensure your response is valid JSON. Do not include trailing commas before closing brackets or braces.
            Format as JSON array with objects having 'name', 'description', 'difficulty' fields."""),
            ("human", "Discipline: {discipline}\n\nComprehensive physics textbook content samples (High School + College + University):\n{content}")
        ])
        
        content = "\n\n".join(chunk_texts[:10])  # Use first 10 chunks
        
        # Check if we have actual content or just metadata
        if not any(len(text.strip()) > 200 for text in chunk_texts[:5]):
            logger.warning("Chunks appear to contain only metadata. Using discipline-based topic generation.")
            return self._generate_discipline_topics(state)
        
        try:
            response = self.llm.invoke(
                topic_extraction_prompt.format_messages(
                    discipline=state["discipline"],
                    content=content
                )
            )
            
            # Parse JSON response
            topics_text = response.content.strip()
            logger.info(f"Raw response: {topics_text[:200]}...")  # Log first 200 chars for debugging
            
            # Clean up markdown formatting
            if topics_text.startswith("```json"):
                topics_text = topics_text.split("```json")[1].split("```")[0]
            elif topics_text.startswith("```"):
                topics_text = topics_text.split("```")[1].split("```")[0]
            
            topics_text = topics_text.strip()
            
            # Try to parse JSON with better error handling
            try:
                topics = json.loads(topics_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed for topics: {e}")
                # Try to clean up common JSON issues
                try:
                    # Remove any trailing commas before closing brackets/braces
                    import re
                    cleaned_text = re.sub(r',\s*([}\]])', r'\1', topics_text)
                    topics = json.loads(cleaned_text)
                    logger.info("Successfully parsed cleaned topics JSON")
                except json.JSONDecodeError:
                    # If still failing, try to extract JSON array from the response
                    json_match = re.search(r'\[.*\]', topics_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group()
                            # Clean the extracted JSON too
                            cleaned_extracted = re.sub(r',\s*([}\]])', r'\1', extracted_json)
                            topics = json.loads(cleaned_extracted)
                            logger.info("Successfully parsed extracted topics JSON")
                        except json.JSONDecodeError:
                            raise ValueError(f"Could not extract valid JSON from response: {topics_text[:200]}")
                    else:
                        raise ValueError(f"Could not extract valid JSON from response: {topics_text[:200]}")
            state["topics"] = topics
            logger.info(f"Extracted {len(topics)} main topics")
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            logger.info("Falling back to discipline-based topic generation...")
            return self._generate_discipline_topics(state)
        
        return state
    
    def generate_subtopics(self, state: CurriculumState) -> CurriculumState:
        """Generate detailed subtopics from actual TOC and chunk data - entirely data-driven."""
        logger.info("🔍 Extracting subtopics from actual Table of Contents and textbook content...")
        
        subtopics = []
        
        # Extract subtopics directly from chunks with TOC enhancement
        subtopics_from_chunks = self._extract_subtopics_from_chunks(state["chunks"])
        
        # Apply cross-level normalization and merging
        normalized_subtopics = self._normalize_cross_level_topics(subtopics_from_chunks)
        
        # Remove duplicates while preserving topic diversity
        deduplicated_subtopics = self._remove_duplicates_preserve_diversity(normalized_subtopics)
        
        # Apply pedagogical ordering with prerequisite enforcement
        ordered_subtopics = self._apply_pedagogical_ordering(deduplicated_subtopics)
        
        # Ensure astrophysics and other electives are placed last
        final_subtopics = self._place_electives_last(ordered_subtopics)
        
        # Apply adaptive quality assessment and iterative improvement
        quality_improved_subtopics = self._adaptive_quality_improvement(final_subtopics)
        
        logger.info(f"✅ Generated {len(quality_improved_subtopics)} meaningful subtopics from actual textbook content")
        
        state["subtopics"] = quality_improved_subtopics
        return state
    
    def _extract_subtopics_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Extract meaningful subtopics directly from chunk data with actual TOC titles."""
        subtopics = []
        
        # Group chunks by book and chapter
        chunks_by_book = {}
        for chunk in chunks:
            book_title = chunk.get('book_title', 'Unknown')
            chapter_title = chunk.get('actual_chapter_title', chunk.get('chapter_title', 'General'))
            
            if book_title not in chunks_by_book:
                chunks_by_book[book_title] = {}
            if chapter_title not in chunks_by_book[book_title]:
                chunks_by_book[book_title][chapter_title] = []
            
            chunks_by_book[book_title][chapter_title].append(chunk)
        
        # Create subtopics from actual chapter structure
        for book_title, chapters in chunks_by_book.items():
            logger.info(f"📚 Processing {book_title}: {len(chapters)} chapters")
            
            for chapter_title, chapter_chunks in chapters.items():
                if chapter_title == "Untitled Chapter":
                    continue  # Skip generic chapters
                
                # Determine educational level from chunks
                educational_level = chapter_chunks[0].get('educational_level', 'undergraduate')
                
                # Classify domain
                domain = self._classify_physics_domain(chapter_title)
                
                # Create main chapter subtopic
                subtopic = {
                    'name': chapter_title,
                    'description': f"Comprehensive study of {chapter_title.lower()} concepts",
                    'educational_level': educational_level,
                    'domain': domain,
                    'source_book': book_title,
                    'parent_topic': domain,
                    'is_core': domain not in ['astrophysics', 'biophysics', 'geophysics'],
                    'mcat_relevant': self._is_mcat_relevant_domain(domain),
                    'difficulty': self._determine_difficulty_from_level(educational_level),
                    'duration': self._estimate_chapter_duration(len(chapter_chunks)),
                    'objectives': self._generate_learning_objectives_for_domain(domain, educational_level),
                    'assessment_methods': self._generate_assessment_methods_for_level(educational_level),
                    'prerequisites': [],
                    'level': educational_level,
                    'core_status': 'core' if domain not in ['astrophysics', 'biophysics'] else 'elective'
                }
                
                subtopics.append(subtopic)
                
                # Create section-level subtopics for detailed coverage
                for i, chunk in enumerate(chapter_chunks[:5]):  # Limit sections per chapter
                    if len(chunk.get('text', '')) > 200:  # Only substantial content
                        section_subtopic = {
                            'name': f"{chapter_title}: Section {i+1}",
                            'description': f"Detailed study of section {i+1} within {chapter_title.lower()}",
                            'educational_level': educational_level,
                            'domain': domain,
                            'source_book': book_title,
                            'parent_topic': chapter_title,
                            'is_core': subtopic['is_core'],
                            'mcat_relevant': subtopic['mcat_relevant'],
                            'difficulty': subtopic['difficulty'],
                            'duration': 2,
                            'objectives': subtopic['objectives'],
                            'assessment_methods': subtopic['assessment_methods'],
                            'prerequisites': [subtopic['name']],
                            'level': educational_level,
                            'core_status': subtopic['core_status']
                        }
                        subtopics.append(section_subtopic)
        
        logger.info(f"📋 Extracted {len(subtopics)} subtopics from actual textbook structure")
        return subtopics
    
    def _classify_physics_domain(self, title: str) -> str:
        """Classify a chapter title into a physics domain."""
        title_lower = title.lower()
        
        # Domain classification based on keywords
        domain_keywords = {
            'units_measurement': ['units', 'measurement', 'dimensional', 'significant', 'what is physics'],
            'vectors': ['vectors', 'vector'],
            'kinematics': ['motion', 'velocity', 'acceleration', 'kinematics', 'displacement'],
            'dynamics': ['forces', 'newton', 'friction', 'dynamics', 'tension', 'mass'],
            'energy': ['work', 'energy', 'conservation', 'kinetic', 'potential', 'power'],
            'momentum': ['momentum', 'impulse', 'collision', 'conservation of momentum'],
            'rotation': ['rotation', 'angular', 'torque', 'rotational', 'spinning'],
            'gravitation': ['gravity', 'gravitation', 'gravitational', 'planetary', 'orbital'],
            'oscillations': ['oscillation', 'harmonic', 'vibration', 'pendulum', 'spring'],
            'waves': ['waves', 'wave', 'sound', 'acoustic', 'interference', 'diffraction'],
            'thermodynamics': ['heat', 'temperature', 'thermal', 'entropy', 'gas', 'thermodynamics'],
            'electricity': ['electric', 'charge', 'coulomb', 'electric field', 'potential', 'capacitor'],
            'magnetism': ['magnetic', 'magnetism', 'electromagnetic', 'induction', 'flux'],
            'circuits': ['circuit', 'current', 'resistance', 'ohm', 'capacitor', 'inductor'],
            'optics': ['light', 'optics', 'reflection', 'refraction', 'lens', 'mirror', 'interference'],
            'modern_physics': ['quantum', 'atomic', 'nuclear', 'particle', 'relativity', 'radioactive'],
            'astrophysics': ['astronomy', 'stellar', 'galaxy', 'universe', 'cosmology', 'planet', 'star']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return domain
        
        return 'general'  # fallback
    
    def _normalize_cross_level_topics(self, subtopics: List[Dict]) -> List[Dict]:
        """Normalize topics across educational levels to create progression."""
        logger.info("🔄 Normalizing topics across educational levels...")
        
        # Group by normalized topic name
        topic_groups = {}
        
        for subtopic in subtopics:
            # Create normalized key
            normalized_name = self._normalize_topic_name(subtopic['name'])
            key = f"{subtopic['domain']}_{normalized_name}"
            
            if key not in topic_groups:
                topic_groups[key] = []
            topic_groups[key].append(subtopic)
        
        normalized_subtopics = []
        
        for group_key, group_subtopics in topic_groups.items():
            if len(group_subtopics) == 1:
                # Unique topic
                normalized_subtopics.append(group_subtopics[0])
            else:
                # Multiple levels - create progression
                level_order = {'high_school': 1, 'undergraduate': 2, 'graduate': 3}
                group_subtopics.sort(key=lambda x: (level_order.get(x['educational_level'], 2), x['name']))
                
                for i, subtopic in enumerate(group_subtopics):
                    # Add level suffix for progression
                    level_suffix = {
                        'high_school': ' (Introductory)',
                        'undergraduate': ' (Intermediate)',
                        'graduate': ' (Advanced)'
                    }.get(subtopic['educational_level'], '')
                    
                    if len(group_subtopics) > 1:
                        subtopic['name'] = f"{subtopic['name']}{level_suffix}"
                        
                        # Add prerequisites from lower levels
                        if i > 0:
                            prev_subtopic = group_subtopics[i-1]
                            subtopic['prerequisites'].append(prev_subtopic['name'])
                    
                    normalized_subtopics.append(subtopic)
        
        logger.info(f"✅ Normalized {len(subtopics)} → {len(normalized_subtopics)} subtopics")
        return normalized_subtopics
    
    def _normalize_topic_name(self, name: str) -> str:
        """Create normalized version of topic name for grouping."""
        # Remove level indicators and common variations
        import re
        normalized = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parentheticals
        normalized = re.sub(r'\s*-\s*Section\s*\d+', '', normalized)  # Remove section numbers
        normalized = re.sub(r'\s*:\s*.*', '', normalized)  # Remove everything after colon
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        return normalized
    
    def _remove_duplicates_preserve_diversity(self, subtopics: List[Dict]) -> List[Dict]:
        """Remove duplicates while preserving topic diversity."""
        logger.info("🗂️ Removing duplicates while preserving diversity...")
        
        seen_names = set()
        unique_subtopics = []
        
        # Sort by pedagogical importance (core first, then by domain priority)
        domain_priority = {
            'units_measurement': 1, 'vectors': 2, 'kinematics': 3, 'dynamics': 4,
            'energy': 5, 'momentum': 6, 'rotation': 7, 'gravitation': 8,
            'oscillations': 9, 'waves': 10, 'thermodynamics': 11,
            'electricity': 12, 'magnetism': 13, 'circuits': 14, 'optics': 15,
            'modern_physics': 16, 'astrophysics': 17
        }
        
        subtopics.sort(key=lambda x: (
            0 if x.get('is_core', True) else 1,  # Core topics first
            domain_priority.get(x.get('domain', 'general'), 99),
            x.get('educational_level', 'undergraduate'),
            x.get('name', '')
        ))
        
        for subtopic in subtopics:
            name = subtopic.get('name', '')
            if name not in seen_names and name:
                seen_names.add(name)
                unique_subtopics.append(subtopic)
        
        logger.info(f"✅ Preserved {len(unique_subtopics)} unique subtopics")
        return unique_subtopics
    
    def _apply_pedagogical_ordering(self, subtopics: List[Dict]) -> List[Dict]:
        """Apply pedagogical ordering ensuring prerequisites come first."""
        logger.info("📋 Applying pedagogical ordering with prerequisite enforcement...")
        
        # Domain prerequisite mapping
        domain_prerequisites = {
            'vectors': ['units_measurement'],
            'kinematics': ['vectors', 'units_measurement'],
            'dynamics': ['kinematics', 'vectors'],
            'energy': ['dynamics', 'kinematics'],
            'momentum': ['dynamics', 'kinematics'],
            'rotation': ['dynamics', 'vectors'],
            'gravitation': ['dynamics', 'energy'],
            'oscillations': ['dynamics', 'energy'],
            'waves': ['oscillations', 'kinematics'],
            'thermodynamics': ['energy', 'dynamics'],
            'electricity': ['vectors', 'energy'],
            'magnetism': ['electricity', 'vectors'],
            'circuits': ['electricity'],
            'optics': ['waves', 'electricity'],
            'modern_physics': ['electricity', 'magnetism', 'energy'],
            'astrophysics': ['gravitation', 'modern_physics', 'thermodynamics']
        }
        
        # Sort by domain prerequisites, then educational level
        domain_priority = {domain: i for i, domain in enumerate(domain_prerequisites.keys(), 1)}
        level_priority = {'high_school': 1, 'undergraduate': 2, 'graduate': 3}
        
        ordered_subtopics = sorted(subtopics, key=lambda x: (
            domain_priority.get(x.get('domain', 'general'), 99),
            level_priority.get(x.get('educational_level', 'undergraduate'), 2),
            x.get('name', '')
        ))
        
        # Assign pedagogical order numbers
        for i, subtopic in enumerate(ordered_subtopics, 1):
            subtopic['pedagogical_order'] = i
        
        logger.info(f"✅ Applied pedagogical ordering to {len(ordered_subtopics)} subtopics")
        return ordered_subtopics
    
    def _place_electives_last(self, subtopics: List[Dict]) -> List[Dict]:
        """Ensure electives like astrophysics are placed last in ordering."""
        logger.info("🌟 Placing electives (astrophysics) at the end...")
        
        core_subtopics = [s for s in subtopics if s.get('is_core', True)]
        elective_subtopics = [s for s in subtopics if not s.get('is_core', True)]
        
        # Sort electives with astrophysics absolutely last
        elective_subtopics.sort(key=lambda x: (
            1 if x.get('domain') == 'astrophysics' else 0,
            x.get('educational_level', 'undergraduate'),
            x.get('name', '')
        ))
        
        # Combine and re-number
        final_subtopics = core_subtopics + elective_subtopics
        
        for i, subtopic in enumerate(final_subtopics, 1):
            subtopic['pedagogical_order'] = i
        
        logger.info(f"✅ Placed {len(elective_subtopics)} electives after {len(core_subtopics)} core topics")
        return final_subtopics
    
    def _adaptive_quality_improvement(self, subtopics: List[Dict]) -> List[Dict]:
        """Apply adaptive quality assessment and iterative improvement."""
        logger.info("🎯 Starting adaptive quality assessment and improvement...")
        
        best_subtopics = subtopics
        best_quality = 0
        target_quality = 0.9
        max_iterations = 3
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"🔍 Quality assessment iteration {iteration}")
            
            # Assess current quality
            quality_score = self._assess_curriculum_quality(subtopics)
            logger.info(f"   Quality score: {quality_score:.3f}")
            
            if quality_score > best_quality:
                best_quality = quality_score
                best_subtopics = subtopics.copy()
                logger.info(f"   ✅ New best quality: {best_quality:.3f}")
            
            if quality_score >= target_quality:
                logger.info("   🎯 Quality target achieved!")
                break
            
            # Apply improvements for next iteration
            if iteration < max_iterations:
                subtopics = self._apply_quality_improvements(subtopics, quality_score)
        
        logger.info(f"✅ Quality improvement completed. Final quality: {best_quality:.3f}")
        return best_subtopics
    
    def _assess_curriculum_quality(self, subtopics: List[Dict]) -> float:
        """Assess the quality of the curriculum."""
        total_subtopics = len(subtopics)
        
        if total_subtopics == 0:
            return 0.0
        
        # Essential domain coverage
        essential_domains = {
            'units_measurement', 'vectors', 'kinematics', 'dynamics', 'energy',
            'momentum', 'waves', 'thermodynamics', 'electricity', 'magnetism', 'optics'
        }
        covered_domains = set(s.get('domain') for s in subtopics)
        coverage_score = len(covered_domains & essential_domains) / len(essential_domains)
        
        # Meaningful naming (avoid generic names)
        meaningful_count = sum(1 for s in subtopics 
                             if not any(generic in s.get('name', '').lower() 
                                      for generic in ['untitled', 'section', 'equations', 'general']))
        naming_score = meaningful_count / total_subtopics
        
        # Educational progression
        levels = set(s.get('educational_level') for s in subtopics)
        progression_score = min(1.0, len(levels) / 2)  # At least 2 levels is good
        
        # Target quantity (aim for ~1000)
        quantity_score = min(1.0, total_subtopics / 1000)
        
        # Overall quality
        quality_score = (coverage_score + naming_score + progression_score + quantity_score) / 4
        
        return quality_score
    
    def _apply_quality_improvements(self, subtopics: List[Dict], current_quality: float) -> List[Dict]:
        """Apply improvements to increase quality."""
        improved_subtopics = subtopics.copy()
        
        # Improve naming if needed
        for subtopic in improved_subtopics:
            name = subtopic.get('name', '')
            if any(generic in name.lower() for generic in ['untitled', 'equations', 'general']):
                # Replace with domain-based name
                domain = subtopic.get('domain', 'general')
                subtopic['name'] = f"{domain.replace('_', ' ').title()} Concepts"
        
        return improved_subtopics
    
    # Helper methods for domain classification and properties
    def _is_mcat_relevant_domain(self, domain: str) -> bool:
        """Determine if domain is relevant for MCAT."""
        mcat_domains = {
            'kinematics', 'dynamics', 'energy', 'momentum', 'waves',
            'thermodynamics', 'electricity', 'magnetism', 'circuits', 'optics'
        }
        return domain in mcat_domains
    
    def _determine_difficulty_from_level(self, level: str) -> int:
        """Determine difficulty based on educational level."""
        level_difficulty = {'high_school': 1, 'undergraduate': 2, 'graduate': 4}
        return level_difficulty.get(level, 2)
    
    def _estimate_chapter_duration(self, num_chunks: int) -> int:
        """Estimate duration based on content amount."""
        return min(6, max(2, num_chunks // 2))
    
    def _generate_learning_objectives_for_domain(self, domain: str, level: str) -> List[str]:
        """Generate appropriate learning objectives for domain and level."""
        base_objectives = {
            'units_measurement': [
                'Apply appropriate units and unit conversions',
                'Perform dimensional analysis',
                'Use significant figures correctly'
            ],
            'vectors': [
                'Add and subtract vectors graphically and analytically',
                'Resolve vectors into components',
                'Apply vector analysis to physical problems'
            ],
            'kinematics': [
                'Analyze motion using position, velocity, and acceleration',
                'Apply kinematic equations to solve motion problems',
                'Interpret motion graphs and diagrams'
            ]
        }
        
        objectives = base_objectives.get(domain, [
            'Understand fundamental concepts',
            'Apply principles to solve problems',
            'Connect theory to real-world applications'
        ])
        
        if level == 'graduate':
            objectives.append('Analyze advanced theoretical implications')
        
        return objectives
    
    def _generate_assessment_methods_for_level(self, level: str) -> List[str]:
        """Generate appropriate assessment methods for educational level."""
        methods = ['Conceptual understanding', 'Problem solving']
        
        if level in ['undergraduate', 'graduate']:
            methods.extend(['Laboratory work', 'Research projects'])
        
        if level == 'graduate':
            methods.append('Advanced theoretical analysis')
        
        return methods

    def build_prerequisite_graph(self, state: CurriculumState) -> CurriculumState:
        """Build prerequisite relationships between subtopics."""
        logger.info("Building prerequisite relationships...")
        
        prerequisites = {}
        
        # Create mapping of subtopic names
        subtopic_names = [sub["name"] for sub in state["subtopics"]]
        
        for subtopic in state["subtopics"]:
            prereq_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert curriculum designer. Determine prerequisite relationships for a subtopic.
                
                Given a subtopic and a list of all available subtopics, identify which subtopics should be completed BEFORE this one.
                Consider:
                1. Conceptual dependencies
                2. Mathematical foundations
                3. Logical learning progression
                
                Return ONLY the names of prerequisite subtopics as a JSON array of strings.
                If no prerequisites, return empty array []."""),
                ("human", """Subtopic: {subtopic_name}
Description: {description}
Learning Objectives: {objectives}

Available subtopics:
{all_subtopics}""")
            ])
            
            try:
                response = self.fast_llm.invoke(
                    prereq_prompt.format_messages(
                        subtopic_name=subtopic["name"],
                        description=subtopic["description"],
                        objectives="\n".join(subtopic.get("objectives", [])),
                        all_subtopics="\n".join(f"- {name}" for name in subtopic_names)
                    )
                )
                
                # Parse JSON response
                prereqs_text = response.content.strip()
                if prereqs_text.startswith("```json"):
                    prereqs_text = prereqs_text.split("```json")[1].split("```")[0]
                elif prereqs_text.startswith("```"):
                    prereqs_text = prereqs_text.split("```")[1].split("```")[0]
                
                # Try to parse JSON with better error handling
                try:
                    prereqs = json.loads(prereqs_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed for prerequisites of {subtopic['name']}: {e}")
                    try:
                        # Remove any trailing commas and clean up
                        import re
                        cleaned_text = re.sub(r',\s*([}\]])', r'\1', prereqs_text)
                        prereqs = json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        logger.error(f"Could not parse prerequisites for {subtopic['name']}, setting to empty list")
                        prereqs = []
                
                # Validate prerequisites exist in subtopic list
                valid_prereqs = [p for p in prereqs if p in subtopic_names and p != subtopic["name"]]
                prerequisites[subtopic["name"]] = valid_prereqs
                
            except Exception as e:
                logger.error(f"Error determining prerequisites for '{subtopic['name']}': {e}")
                prerequisites[subtopic["name"]] = []
        
        state["prerequisites"] = prerequisites
        logger.info(f"Built prerequisite relationships for {len(prerequisites)} subtopics")
        return state
    
    def create_curriculum_graph(self, state: CurriculumState) -> CurriculumState:
        """Create a directed graph representation of the curriculum."""
        logger.info("Creating curriculum graph...")
        
        # Create NetworkX directed graph
        G = nx.DiGraph()
        
        # Add nodes (subtopics)
        for subtopic in state["subtopics"]:
            G.add_node(subtopic["name"], 
                      description=subtopic["description"],
                      duration=subtopic.get("duration", 2),
                      difficulty=subtopic.get("topic_difficulty", 3),
                      parent_topic=subtopic.get("parent_topic", ""))
        
        # Add edges (prerequisites)
        for subtopic_name, prereqs in state["prerequisites"].items():
            for prereq in prereqs:
                if prereq in G.nodes and subtopic_name in G.nodes:
                    G.add_edge(prereq, subtopic_name)
        
        # Calculate graph metrics
        curriculum_info = {
            "total_subtopics": len(state["subtopics"]),
            "total_topics": len(state["topics"]),
            "prerequisite_edges": G.number_of_edges(),
            "graph_density": nx.density(G),
            "is_dag": nx.is_directed_acyclic_graph(G),
            "connected_components": nx.number_weakly_connected_components(G)
        }
        
        # Generate topological order if DAG
        if curriculum_info["is_dag"]:
            try:
                topological_order = list(nx.topological_sort(G))
                curriculum_info["learning_path"] = topological_order
            except:
                curriculum_info["learning_path"] = []
        else:
            logger.warning("Curriculum graph is not a DAG - there may be circular dependencies")
            curriculum_info["learning_path"] = []
        
        state["curriculum_graph"] = curriculum_info
        
        # Save graph visualization
        self.save_graph_visualization(G, state["discipline"])
        
        logger.info(f"Created curriculum graph with {curriculum_info['total_subtopics']} subtopics and {curriculum_info['prerequisite_edges']} prerequisite relationships")
        return state
    
    def save_graph_visualization(self, G: nx.DiGraph, discipline: str):
        """Save a visualization of the curriculum graph."""
        try:
            plt.figure(figsize=(20, 16))
            
            # Use hierarchical layout
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue', alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels (truncated for readability)
            labels = {node: node[:15] + "..." if len(node) > 15 else node for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title(f"{discipline} Curriculum Graph\n({G.number_of_nodes()} subtopics, {G.number_of_edges()} prerequisites)", 
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot
            graph_file = CURRICULUM_DIR / f"{discipline}_curriculum_graph.png"
            plt.savefig(graph_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved curriculum graph visualization: {graph_file}")
            
        except Exception as e:
            logger.error(f"Error creating graph visualization: {e}")
    
    def _generate_discipline_topics(self, state: CurriculumState) -> CurriculumState:
        """Generate topics based on discipline when content is insufficient."""
        discipline = state["discipline"]
        
        topic_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert curriculum designer. Generate a comprehensive list of main topics for a given academic discipline.
            
            For each topic, provide:
            1. Topic name (clear and concise)
            2. Description (2-3 sentences explaining the topic)
            3. Estimated difficulty level (1-5, where 1=beginner, 5=advanced)
            
            Generate exactly 20 main topics that would be covered in a comprehensive university-level course.
            Format as JSON array with objects having 'name', 'description', 'difficulty' fields.
            
            Example format:
            [
                {
                    "name": "Topic Name",
                    "description": "Brief description of the topic and its importance.",
                    "difficulty": 3
                }
            ]"""),
            ("human", "Generate comprehensive main topics for: {discipline}")
        ])
        
        try:
            response = self.llm.invoke(
                topic_generation_prompt.format_messages(discipline=discipline)
            )
            
            # Parse JSON response
            topics_text = response.content.strip()
            
            # Clean up markdown formatting
            if topics_text.startswith("```json"):
                topics_text = topics_text.split("```json")[1].split("```")[0]
            elif topics_text.startswith("```"):
                topics_text = topics_text.split("```")[1].split("```")[0]
            
            topics_text = topics_text.strip()
            
            # Try to parse JSON with better error handling
            try:
                topics = json.loads(topics_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parsing failed for discipline topics: {e}")
                # Try to clean up common JSON issues
                try:
                    # Remove any trailing commas before closing brackets/braces
                    import re
                    cleaned_text = re.sub(r',\s*([}\]])', r'\1', topics_text)
                    topics = json.loads(cleaned_text)
                    logger.info("Successfully parsed cleaned discipline topics JSON")
                except json.JSONDecodeError:
                    # If still failing, try to extract JSON array from the response
                    json_match = re.search(r'\[.*\]', topics_text, re.DOTALL)
                    if json_match:
                        try:
                            extracted_json = json_match.group()
                            # Clean the extracted JSON too
                            cleaned_extracted = re.sub(r',\s*([}\]])', r'\1', extracted_json)
                            topics = json.loads(cleaned_extracted)
                            logger.info("Successfully parsed extracted discipline topics JSON")
                        except json.JSONDecodeError:
                            # As a last resort, create default topics
                            topics = self._get_default_topics(discipline)
                    else:
                        # As a last resort, create default topics
                        topics = self._get_default_topics(discipline)
            
            state["topics"] = topics
            logger.info(f"Generated {len(topics)} topics for {discipline}")
            
        except Exception as e:
            logger.error(f"Error generating topics: {e}")
            state["topics"] = self._get_default_topics(discipline)
            state["error"] = str(e)
        
        return state
    
    def _determine_subtopic_core_status(self, subtopic_name: str, parent_topic: str) -> str:
        """Determine if a subtopic is core or elective material."""
        subtopic_lower = subtopic_name.lower()
        parent_lower = parent_topic.lower()
        
        # Core topics that are foundational
        core_indicators = [
            'units', 'dimensional analysis', 'measurement', 'basic', 'fundamental',
            'introduction', 'principles', 'laws', 'conservation', 'newton',
            'kinematics', 'dynamics', 'energy', 'momentum', 'force',
            'electric field', 'magnetic field', 'maxwell', 'wave equation',
            'thermodynamic', 'temperature', 'heat', 'entropy'
        ]
        
        # Elective/advanced topics
        elective_indicators = [
            'advanced', 'specialized', 'research', 'current', 'modern',
            'quantum field', 'cosmology', 'astrophysics', 'plasma',
            'biophysics', 'geophysics', 'medical physics', 'computational',
            'numerical', 'simulation', 'experimental techniques'
        ]
        
        # Check core patterns
        if any(indicator in subtopic_lower for indicator in core_indicators):
            return 'core'
        
        # Check elective patterns
        if any(indicator in subtopic_lower for indicator in elective_indicators):
            return 'elective'
        
        # Check parent topic for context
        if any(term in parent_lower for term in ['advanced', 'specialized', 'research']):
            return 'elective'
        
        # Default to core for basic physics domains
        basic_domains = ['classical mechanics', 'thermodynamics', 'electromagnetism', 'waves', 'optics']
        if any(domain in parent_lower for domain in basic_domains):
            return 'core'
        
        return 'core'  # Default to core
    
    def _determine_subtopic_level(self, subtopic_name: str, topic_difficulty: int) -> str:
        """Determine educational level for a subtopic."""
        subtopic_lower = subtopic_name.lower()
        
        # High school level indicators
        if (topic_difficulty <= 2 or 
            any(indicator in subtopic_lower for indicator in [
                'basic', 'simple', 'introduction', 'elementary', 'fundamental',
                'newton\'s laws', 'simple harmonic', 'basic electricity',
                'units', 'measurement', 'graphing', 'problem solving'
            ])):
            return 'high_school'
        
        # Graduate level indicators
        elif (topic_difficulty >= 5 or 
              any(indicator in subtopic_lower for indicator in [
                  'quantum field', 'advanced quantum', 'graduate', 'research',
                  'string theory', 'general relativity', 'many-body',
                  'statistical mechanics', 'field theory', 'group theory'
              ])):
            return 'graduate'
        
        # Default to undergraduate
        else:
            return 'undergraduate'
    
    def _determine_mcat_relevance(self, subtopic_name: str, parent_topic: str) -> bool:
        """Determine if a subtopic is relevant to the MCAT exam."""
        subtopic_lower = subtopic_name.lower()
        parent_lower = parent_topic.lower()
        
        # MCAT-relevant physics topics (based on official MCAT content outline)
        mcat_topics = {
            'units', 'dimensional analysis', 'kinematics', 'newton', 'force', 'friction',
            'work', 'energy', 'power', 'momentum', 'impulse', 'conservation',
            'electric charge', 'electric field', 'circuits', 'magnetic field',
            'pressure', 'density', 'buoyancy', 'waves', 'sound', 'optics',
            'thermodynamics', 'heat', 'temperature', 'gas laws', 'fluids'
        }
        
        # Check if subtopic or parent topic contains MCAT-relevant keywords
        if any(topic in subtopic_lower for topic in mcat_topics):
            return True
        if any(topic in parent_lower for topic in mcat_topics):
            return True
            
        # Specific MCAT physics domains
        mcat_domains = {
            'classical mechanics', 'thermodynamics', 'electromagnetism', 
            'waves & optics', 'modern physics'
        }
        
        if any(domain in parent_lower for domain in mcat_domains):
            # Exclude very advanced topics
            advanced_exclusions = {
                'quantum field', 'general relativity', 'graduate', 'research',
                'theoretical', 'advanced quantum', 'particle physics'
            }
            if not any(exclusion in subtopic_lower for exclusion in advanced_exclusions):
                return True
        
        return False
    
    def _determine_pedagogical_order(self, subtopic_name: str, parent_topic: str, topic_difficulty: int) -> int:
        """Determine the pedagogical order for optimal learning sequence."""
        subtopic_lower = subtopic_name.lower()
        parent_lower = parent_topic.lower()
        
        # Fundamental concepts (order 1-10)
        if any(term in subtopic_lower for term in ['units', 'dimensional analysis']):
            return 1
        if any(term in subtopic_lower for term in ['measurement', 'problem solving', 'introduction']):
            return 5
        
        # Classical Mechanics progression (order 10-40)
        mechanics_order = {
            'kinematics': 10, 'newton': 15, 'force': 16, 'work': 20,
            'energy': 21, 'momentum': 25, 'circular motion': 30,
            'oscillation': 32, 'gravitation': 35, 'fluid': 38
        }
        
        # Electromagnetism progression (order 40-70)
        em_order = {
            'electric charge': 40, 'electric field': 42, 'electric potential': 45,
            'capacitance': 48, 'current': 50, 'resistance': 52, 'magnetic field': 55,
            'electromagnetic induction': 60, 'maxwell': 65, 'electromagnetic waves': 68
        }
        
        # Thermodynamics progression (order 70-90)
        thermo_order = {
            'temperature': 70, 'heat': 72, 'first law': 75, 'second law': 78,
            'entropy': 80, 'heat engines': 82, 'statistical mechanics': 85
        }
        
        # Waves and Optics progression (order 90-110)
        waves_order = {
            'wave properties': 90, 'sound': 92, 'electromagnetic spectrum': 95,
            'reflection': 98, 'refraction': 100, 'interference': 102,
            'diffraction': 105, 'polarization': 108
        }
        
        # Modern Physics progression (order 110-140)
        modern_order = {
            'special relativity': 110, 'photoelectric': 115, 'atomic structure': 120,
            'quantum mechanics': 125, 'nuclear': 130, 'particle physics': 135
        }
        
        # Advanced topics (order 140+)
        advanced_base = 140
        
        # Check each category
        for term, order in mechanics_order.items():
            if term in subtopic_lower:
                return order
                
        for term, order in em_order.items():
            if term in subtopic_lower:
                return order
                
        for term, order in thermo_order.items():
            if term in subtopic_lower:
                return order
                
        for term, order in waves_order.items():
            if term in subtopic_lower:
                return order
                
        for term, order in modern_order.items():
            if term in subtopic_lower:
                return order
        
        # Advanced/graduate topics
        if any(term in subtopic_lower for term in ['quantum field', 'general relativity', 'graduate']):
            return advanced_base + 50
            
        # Default based on topic difficulty and parent topic
        if 'classical mechanics' in parent_lower:
            return 20 + topic_difficulty * 5
        elif 'thermodynamics' in parent_lower:
            return 75 + topic_difficulty * 3
        elif 'electromagnetism' in parent_lower:
            return 50 + topic_difficulty * 4
        elif 'waves' in parent_lower or 'optics' in parent_lower:
            return 95 + topic_difficulty * 3
        elif 'modern physics' in parent_lower:
            return 115 + topic_difficulty * 5
        elif 'quantum' in parent_lower:
            return 125 + topic_difficulty * 5
        else:
            return 50 + topic_difficulty * 10  # Default ordering
    
    def _get_default_topics(self, discipline: str) -> List[Dict[str, Any]]:
        """Get default comprehensive physics domains for fallback."""
        if discipline.lower() == "physics":
            return [
                {"name": "Units and Dimensional Analysis", "description": "Fundamental concepts of measurement, unit systems, and dimensional analysis. Essential foundation for all physics calculations and problem-solving.", "difficulty": 1},
                {"name": "Classical Mechanics", "description": "Study of motion, forces, energy, and momentum from Newton's laws to Lagrangian mechanics. Covers kinematics, dynamics, oscillations, and rotational motion.", "difficulty": 2},
                {"name": "Thermodynamics and Statistical Mechanics", "description": "Study of heat, temperature, entropy, and energy transfer. Includes laws of thermodynamics, heat engines, and statistical interpretation of thermal phenomena.", "difficulty": 3},
                {"name": "Fluid Dynamics", "description": "Study of fluid motion and forces in liquids and gases. Covers fluid statics, flow dynamics, Bernoulli's principle, and viscosity.", "difficulty": 3},
                {"name": "Electricity and Magnetism", "description": "Study of electric and magnetic phenomena, circuits, and electromagnetic waves. Covers electrostatics, magnetostatics, induction, and Maxwell's equations.", "difficulty": 3},
                {"name": "Waves and Oscillations", "description": "Study of periodic motion, wave propagation, and resonance phenomena. Includes simple harmonic motion, wave equations, sound, and mechanical vibrations.", "difficulty": 2},
                {"name": "Optics", "description": "Study of light behavior, image formation, and optical instruments. Covers geometric optics, wave optics, interference, diffraction, and polarization.", "difficulty": 3},
                {"name": "Modern Physics and Relativity", "description": "Study of early 20th century physics developments. Includes special and general relativity, time dilation, length contraction, and spacetime.", "difficulty": 4},
                {"name": "Quantum Mechanics", "description": "Study of matter and energy at atomic and subatomic scales. Covers wave-particle duality, uncertainty principle, Schrödinger equation, and quantum states.", "difficulty": 5},
                {"name": "Atomic and Molecular Physics", "description": "Study of atomic structure, electron configurations, and molecular bonding. Includes spectroscopy, atomic models, and chemical physics.", "difficulty": 4},
                {"name": "Nuclear and Particle Physics", "description": "Study of atomic nuclei, radioactivity, and elementary particles. Covers nuclear reactions, decay processes, and fundamental forces.", "difficulty": 4},
                {"name": "Condensed Matter Physics", "description": "Study of solid and liquid matter properties. Includes crystal structures, electronic properties, superconductivity, and phase transitions.", "difficulty": 5},
                {"name": "Astrophysics and Cosmology", "description": "Study of celestial objects and the universe. Covers stellar evolution, galaxies, black holes, Big Bang theory, and cosmic phenomena.", "difficulty": 4},
                {"name": "Mathematical Methods in Physics", "description": "Mathematical tools and techniques used in physics. Covers differential equations, vector calculus, complex analysis, and computational methods.", "difficulty": 3},
                {"name": "Experimental Physics and Instrumentation", "description": "Experimental techniques, measurement methods, and scientific instrumentation. Covers data analysis, error propagation, and laboratory methods.", "difficulty": 2},
                {"name": "Engineering Physics", "description": "Application of physics principles to engineering problems. Covers materials science, device physics, and technological applications of physical principles.", "difficulty": 4},
                {"name": "Biophysics", "description": "Application of physics methods to biological systems. Covers biomechanics, molecular motors, membrane physics, and biological imaging techniques.", "difficulty": 5},
                {"name": "Geophysics", "description": "Physics of Earth and planetary systems. Covers seismology, magnetism, gravitational fields, and atmospheric physics.", "difficulty": 4},
                {"name": "Computational Physics", "description": "Numerical methods and computer simulations in physics. Covers Monte Carlo methods, finite element analysis, and parallel computing.", "difficulty": 4},
                {"name": "Theoretical Physics", "description": "Advanced theoretical frameworks and mathematical physics. Covers field theory, group theory, and advanced quantum mechanics.", "difficulty": 6},
                {"name": "Medical Physics", "description": "Physics applications in medicine and healthcare. Covers radiation therapy, medical imaging, and biological effects of radiation.", "difficulty": 5},
                {"name": "Plasma Physics", "description": "Physics of ionized gases and plasma states. Covers fusion physics, magnetohydrodynamics, and space plasma phenomena.", "difficulty": 5}
            ]
        else:
            return [
                {"name": f"{discipline} Fundamentals", "description": f"Basic concepts and principles of {discipline}.", "difficulty": 2},
                {"name": f"Advanced {discipline}", "description": f"Advanced topics and applications in {discipline}.", "difficulty": 4}
            ]
    
    def apply_pedagogical_ordering(self, state: CurriculumState) -> CurriculumState:
        """Apply comprehensive pedagogical ordering based on expert textbook TOC analysis."""
        logger.info("Applying comprehensive pedagogical ordering...")
        
        # Phase 1: Use TOC-based pedagogical system if available
        if PedagogicalOrderingSystem is not None:
            try:
                logger.info("Using comprehensive TOC-based pedagogical ordering system")
                physics_books_path = Path("/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics")
                
                ordering_system = PedagogicalOrderingSystem()
                pedagogical_result = ordering_system.generate_pedagogical_curriculum(physics_books_path)
                
                # Create concept-to-order mapping from expert analysis
                concept_order_map = {}
                for seq in pedagogical_result['pedagogical_sequences']:
                    concept_order_map[seq['concept_name']] = seq['order_position']
                
                # Apply expert ordering to subtopics
                for subtopic in state["subtopics"]:
                    expert_order = self._map_subtopic_to_expert_order(subtopic, concept_order_map)
                    subtopic["pedagogical_order"] = expert_order
                    subtopic["pedagogical_justification"] = f"Based on expert textbook analysis from {pedagogical_result['statistics']['source_books']} sources"
                
                # Save pedagogical analysis for reference
                state["pedagogical_analysis"] = pedagogical_result
                
                logger.info(f"Applied expert TOC-based ordering with {pedagogical_result['statistics']['average_confidence']:.2f} average confidence")
                
            except Exception as e:
                logger.warning(f"TOC-based ordering failed, falling back to heuristic: {e}")
                self._apply_heuristic_ordering(state)
        else:
            logger.warning("TOC-based ordering system not available, using heuristic ordering")
            self._apply_heuristic_ordering(state)
        
        # Phase 2: Sort subtopics by pedagogical order
        def sort_key(x):
            return (x.get("pedagogical_order", 999), x.get("parent_topic", ""), x.get("name", ""))
        
        sorted_subtopics = sorted(state["subtopics"], key=sort_key)
        state["subtopics"] = sorted_subtopics
        
        logger.info(f"Applied pedagogical ordering to {len(sorted_subtopics)} subtopics")
        return state
    
    def _map_subtopic_to_expert_order(self, subtopic: Dict[str, Any], concept_order_map: Dict[str, int]) -> int:
        """Map a subtopic to expert pedagogical order based on concept analysis."""
        subtopic_name = subtopic.get("name", "").lower()
        parent_topic = subtopic.get("parent_topic", "").lower()
        
        # Physics concept mapping based on expert analysis
        physics_concept_keywords = {
            'units_measurement': ['units', 'dimensional analysis', 'measurement', 'si units'],
            'kinematics': ['kinematics', 'motion', 'velocity', 'acceleration', 'displacement'],
            'dynamics': ['dynamics', 'force', 'newton', 'friction', 'tension'],
            'energy_work': ['work', 'energy', 'power', 'conservation of energy', 'kinetic energy', 'potential energy'],
            'momentum': ['momentum', 'impulse', 'conservation of momentum', 'collision'],
            'rotational_motion': ['rotation', 'torque', 'angular', 'moment of inertia'],
            'oscillations': ['oscillation', 'harmonic motion', 'pendulum', 'spring'],
            'gravitation': ['gravity', 'gravitational', 'orbital', 'kepler'],
            'temperature_heat': ['temperature', 'heat', 'thermal', 'calorimetry'],
            'thermodynamic_laws': ['thermodynamic', 'entropy', 'enthalpy', 'first law', 'second law'],
            'kinetic_theory': ['kinetic theory', 'ideal gas', 'molecular'],
            'heat_transfer': ['conduction', 'convection', 'radiation', 'heat transfer'],
            'electrostatics': ['electric charge', 'electric field', 'coulomb', 'gauss'],
            'electric_potential': ['electric potential', 'voltage', 'capacitor'],
            'electric_current': ['current', 'resistance', 'ohm', 'circuit'],
            'magnetism': ['magnetic', 'magnet', 'ampere', 'magnetic field'],
            'electromagnetic_induction': ['induction', 'faraday', 'lenz', 'inductance'],
            'electromagnetic_waves': ['electromagnetic wave', 'maxwell', 'light', 'radiation'],
            'wave_properties': ['wave', 'wavelength', 'frequency', 'amplitude'],
            'wave_behavior': ['interference', 'diffraction', 'reflection', 'refraction'],
            'sound': ['sound', 'acoustic', 'doppler'],
            'geometric_optics': ['mirror', 'lens', 'ray optics', 'image'],
            'wave_optics': ['wave optics', 'polarization', 'coherence'],
            'special_relativity': ['relativity', 'lorentz', 'time dilation'],
            'quantum_mechanics': ['quantum', 'photon', 'uncertainty', 'wave particle'],
            'atomic_physics': ['atomic', 'electron', 'spectroscopy'],
            'nuclear_physics': ['nuclear', 'radioactive', 'fission', 'fusion']
        }
        
        # Find best matching concept
        best_match = None
        max_matches = 0
        
        for concept, keywords in physics_concept_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in subtopic_name or keyword in parent_topic)
            if matches > max_matches:
                max_matches = matches
                best_match = concept
        
        # Get expert order for the concept
        if best_match and best_match in concept_order_map:
            base_order = concept_order_map[best_match] * 10  # Scale to leave room for sub-ordering
            
            # Fine-tune within concept based on complexity
            complexity_bonus = 0
            if any(word in subtopic_name for word in ['advanced', 'graduate', 'quantum field', 'general relativity']):
                complexity_bonus = 8
            elif any(word in subtopic_name for word in ['introduction', 'basic', 'fundamental']):
                complexity_bonus = -2
            
            return base_order + complexity_bonus
        
        # Fallback to heuristic ordering
        return self._determine_pedagogical_order_heuristic(subtopic_name, parent_topic, subtopic.get("topic_difficulty", 3))
    
    def _apply_heuristic_ordering(self, state: CurriculumState) -> None:
        """Apply heuristic pedagogical ordering when TOC-based system is unavailable."""
        for subtopic in state["subtopics"]:
            subtopic_name = subtopic.get("name", "")
            parent_topic = subtopic.get("parent_topic", "")
            topic_difficulty = subtopic.get("topic_difficulty", 3)
            
            order = self._determine_pedagogical_order_heuristic(subtopic_name, parent_topic, topic_difficulty)
            subtopic["pedagogical_order"] = order
            subtopic["pedagogical_justification"] = "Heuristic ordering based on physics education principles"
    
    def _determine_pedagogical_order_heuristic(self, subtopic_name: str, parent_topic: str, topic_difficulty: int) -> int:
        """Fallback heuristic ordering method."""
        # This is the original heuristic method for fallback
        subtopic_lower = subtopic_name.lower()
        parent_lower = parent_topic.lower()
        
        # Fundamental concepts (order 1-10)
        if any(term in subtopic_lower for term in ['units', 'dimensional analysis']):
            return 1
        if any(term in subtopic_lower for term in ['measurement', 'problem solving', 'introduction']):
            return 5
        
        # Classical Mechanics progression (order 10-40)
        mechanics_order = {
            'kinematics': 10, 'newton': 15, 'force': 16, 'work': 20,
            'energy': 21, 'momentum': 25, 'circular motion': 30,
            'oscillation': 32, 'gravitation': 35, 'fluid': 38
        }
        
        # Check each category and return appropriate order
        for term, order in mechanics_order.items():
            if term in subtopic_lower:
                return order
        
        # Default based on topic difficulty and parent topic
        if 'classical mechanics' in parent_lower:
            return 20 + topic_difficulty * 5
        elif 'thermodynamics' in parent_lower:
            return 75 + topic_difficulty * 3
        elif 'electromagnetism' in parent_lower:
            return 50 + topic_difficulty * 4
        elif 'waves' in parent_lower or 'optics' in parent_lower:
            return 95 + topic_difficulty * 3
        elif 'modern physics' in parent_lower:
            return 115 + topic_difficulty * 5
        elif 'quantum' in parent_lower:
            return 125 + topic_difficulty * 5
        else:
            return 50 + topic_difficulty * 10  # Default ordering
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for curriculum generation."""
        workflow = StateGraph(CurriculumState)
        
        # Add nodes
        workflow.add_node("extract_topics", self.extract_topics)
        workflow.add_node("generate_subtopics", self.generate_subtopics)
        workflow.add_node("apply_ordering", self.apply_pedagogical_ordering)
        workflow.add_node("build_prerequisites", self.build_prerequisite_graph)
        workflow.add_node("create_graph", self.create_curriculum_graph)
        
        # Add edges
        workflow.set_entry_point("extract_topics")
        workflow.add_edge("extract_topics", "generate_subtopics")
        workflow.add_edge("generate_subtopics", "apply_ordering")
        workflow.add_edge("apply_ordering", "build_prerequisites")
        workflow.add_edge("build_prerequisites", "create_graph")
        workflow.add_edge("create_graph", END)
        
        # Compile workflow
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def save_curriculum(self, state: CurriculumState):
        """Save the generated curriculum to files."""
        discipline = state["discipline"]
        
        # Save complete curriculum
        curriculum_data = {
            "discipline": discipline,
            "topics": state["topics"],
            "subtopics": state["subtopics"],
            "prerequisites": state["prerequisites"],
            "graph_info": state["curriculum_graph"],
            "generation_stats": {
                "total_chunks_processed": len(state["chunks"]),
                "topics_extracted": len(state["topics"]),
                "subtopics_generated": len(state["subtopics"]),
                "prerequisite_relationships": sum(len(prereqs) for prereqs in state["prerequisites"].values())
            }
        }
        
        curriculum_file = CURRICULUM_DIR / f"{discipline}_curriculum.json"
        with open(curriculum_file, 'w', encoding='utf-8') as f:
            json.dump(curriculum_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved curriculum to: {curriculum_file}")
        
        # Save curriculum as TSV for spreadsheet viewing with level information
        tsv_file = CURRICULUM_DIR / f"{discipline}_curriculum.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            # Write header with all columns including MCAT and pedagogical ordering
            f.write("Topic\tSubtopic\tDescription\tObjectives\tDuration (hours)\tDifficulty\tLevel\tCore/Elective\tMCAT Relevant\tPedagogical Order\tPrerequisites\n")
            
            # Group subtopics by their parent topics
            topic_map = {topic["name"]: topic for topic in state["topics"]}
            
            for subtopic in state["subtopics"]:
                # Find parent topic
                parent_topic = subtopic.get("parent_topic", "Unknown")
                
                # Format data for TSV
                topic_name = parent_topic
                subtopic_name = subtopic.get("name", "")
                description = subtopic.get("description", "").replace('\n', ' ').replace('\t', ' ')
                objectives = "; ".join(subtopic.get("objectives", [])).replace('\n', ' ').replace('\t', ' ')
                duration = str(subtopic.get("duration", ""))
                difficulty = str(subtopic.get("difficulty", topic_map.get(parent_topic, {}).get("difficulty", "")))
                level = subtopic.get("level", "undergraduate")
                core_status = subtopic.get("core_status", "core")
                mcat_relevant = "Yes" if subtopic.get("mcat_relevant", False) else "No"
                pedagogical_order = str(subtopic.get("pedagogical_order", 50))
                prerequisites = "; ".join(state["prerequisites"].get(subtopic_name, [])).replace('\n', ' ').replace('\t', ' ')
                
                # Write row
                f.write(f"{topic_name}\t{subtopic_name}\t{description}\t{objectives}\t{duration}\t{difficulty}\t{level}\t{core_status}\t{mcat_relevant}\t{pedagogical_order}\t{prerequisites}\n")
        
        logger.info(f"Saved curriculum TSV to: {tsv_file}")
        
        # Save learning path
        if state["curriculum_graph"].get("learning_path"):
            learning_path_file = CURRICULUM_DIR / f"{discipline}_learning_path.txt"
            with open(learning_path_file, 'w', encoding='utf-8') as f:
                f.write(f"Learning Path for {discipline}\n")
                f.write("=" * 50 + "\n\n")
                for i, subtopic in enumerate(state["curriculum_graph"]["learning_path"], 1):
                    f.write(f"{i:2d}. {subtopic}\n")
            
            logger.info(f"Saved learning path to: {learning_path_file}")

def generate_curriculum(discipline: str = "Physics", openai_api_key: str = None, provider: str = "openai"):
    """Main function to generate curriculum for a discipline."""
    logger.info(f"Starting curriculum generation for discipline: {discipline} using {provider}")
    
    try:
        # Initialize generator (will automatically use environment variables if available)
        generator = CurriculumGenerator(openai_api_key=openai_api_key, provider=provider)
        
        # Load book metadata for educational level context
        book_metadata = generator.load_book_metadata()
        
        # Load chunks
        chunks = generator.load_chunks(discipline)
        if not chunks:
            logger.error(f"No chunks found for discipline: {discipline}")
            return
        
        # Initialize state
        initial_state = CurriculumState(
            discipline=discipline,
            chunks=chunks,
            topics=[],
            subtopics=[],
            prerequisites={},
            curriculum_graph={},
            error=""
        )
        
        # Add book metadata to state
        initial_state["book_metadata"] = book_metadata
        
        # Run workflow
        logger.info("Running curriculum generation workflow...")
        config = {"configurable": {"thread_id": f"{discipline}_curriculum"}}
        
        final_state = generator.workflow.invoke(initial_state, config=config)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Workflow error: {final_state['error']}")
            return
        
        # Save results
        generator.save_curriculum(final_state)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("CURRICULUM GENERATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Discipline: {discipline}")
        logger.info(f"Topics: {len(final_state['topics'])}")
        logger.info(f"Subtopics: {len(final_state['subtopics'])}")
        logger.info(f"Prerequisites: {sum(len(prereqs) for prereqs in final_state['prerequisites'].values())}")
        logger.info(f"Graph is DAG: {final_state['curriculum_graph'].get('is_dag', False)}")
        logger.info("✅ Curriculum generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during curriculum generation: {e}")
        raise

def extract_physics_books():
    """Extract actual book titles and metadata from physics directories."""
    books_data = []
    
    # Define the specific physics directories
    physics_dirs = [
        "/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics/HighSchool/osbooks-physics",
        "/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics/University/osbooks-college-physics-bundle", 
        "/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics/University/osbooks-university-physics-bundle",
        "/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics/University/osbooks-astronomy"
    ]
    
    for dir_path in physics_dirs:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            continue
            
        # Look for collection XML files
        collections_dir = dir_path / "collections"
        if collections_dir.exists():
            for collection_file in collections_dir.glob("*.collection.xml"):
                try:
                    # Parse XML to extract book title
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(collection_file)
                    root = tree.getroot()
                    
                    # Find title in metadata
                    title_elem = root.find(".//{http://cnx.rice.edu/mdml}title")
                    if title_elem is not None:
                        book_title = title_elem.text
                        
                        # Determine level and domain based on directory structure
                        if "HighSchool" in str(dir_path):
                            level = "high_school"
                        elif "University" in str(dir_path):
                            level = "undergraduate"
                        else:
                            level = "undergraduate"
                        
                        # Determine domain based on book title
                        title_lower = book_title.lower()
                        if "physics" in title_lower:
                            if "college" in title_lower:
                                domain = "general_physics"
                            elif "university" in title_lower:
                                domain = "advanced_physics"
                            else:
                                domain = "physics"
                        elif "astronomy" in title_lower:
                            domain = "astronomy"
                        else:
                            domain = "physics"
                        
                        books_data.append({
                            'name': book_title,
                            'domain': domain,
                            'level': level,
                            'path': str(collection_file),
                            'source_dir': str(dir_path)
                        })
                        
                        logger.info(f"Found book: {book_title} ({level}, {domain})")
                        
                except Exception as e:
                    logger.warning(f"Error parsing {collection_file}: {e}")
    
    return books_data

def run_adaptive_curriculum(discipline: str = "Physics"):
    """Run the fully adaptive, data-driven curriculum generation system."""
    try:
        if DataDrivenCurriculumBuilder is None:
            logger.error("DataDrivenCurriculumBuilder not imported - check if adaptive_curriculum_system.py exists")
            logger.info("Falling back to basic curriculum generation")
            generate_curriculum(discipline=discipline)
            return None
    
        logger.info("🚀 Starting ADAPTIVE CURRICULUM GENERATION")
        logger.info("🔬 Fully data-driven, self-improving system with iterative quality enhancement")
        logger.info("🎯 Target: ~1,000+ subtopics with pedagogical ordering and error-free JSON parsing")
        
        # Extract actual book titles from physics directories
        logger.info("📖 Extracting book titles from physics directories...")
        books_data = extract_physics_books()
    
        if not books_data:
            logger.warning("No book data found, using simulated data for testing")
            books_data = [
                {'name': 'College Physics', 'domain': 'mechanics', 'level': 'undergraduate'},
                {'name': 'Introduction to Astrophysics', 'domain': 'astrophysics', 'level': 'undergraduate'},
                {'name': 'Biophysics Fundamentals', 'domain': 'biophysics', 'level': 'graduate'},
                {'name': 'Advanced Electromagnetism', 'domain': 'electromagnetism', 'level': 'graduate'},
                {'name': 'Thermodynamics and Statistical Mechanics', 'domain': 'thermodynamics', 'level': 'graduate'},
                {'name': 'Quantum Mechanics', 'domain': 'quantum', 'level': 'graduate'},
            ]
        
        logger.info(f"📚 Processing {len(books_data)} books with adaptive system:")
        for book in books_data:
            logger.info(f"   • {book['name']} ({book['level']}, {book['domain']})")
        
        # Initialize adaptive curriculum builder
        builder = DataDrivenCurriculumBuilder()
        
        # Build adaptive curriculum with iterative improvement
        result = builder.build_adaptive_curriculum(books_data)
        
        # Save results with comprehensive metadata
        output_file = CURRICULUM_DIR / f"{discipline}_adaptive_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save TSV format
        tsv_file = CURRICULUM_DIR / f"{discipline}_adaptive_curriculum.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            f.write("Topic\tSubtopic\tDescription\tEducational_Level\tDepth_Level\tDomain\tDifficulty\tDuration_Hours\tIs_Core\tPedagogical_Order\tMCAT_Relevant\tPrerequisites\tLearning_Objectives\tAssessment_Methods\tSource_Books\tExtraction_Strategy\n")
            
            for subtopic in result['subtopics']:
                row_data = [
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('description', '').replace('\t', ' ').replace('\n', ' '),
                    subtopic.get('level', ''),
                    str(subtopic.get('depth_level', 1)),
                    subtopic.get('domain', ''),
                    str(subtopic.get('difficulty', 3)),
                    str(subtopic.get('duration', 3)),
                    'Core' if subtopic.get('core_status') == 'core' else 'Elective',
                    str(subtopic.get('pedagogical_order', '')),
                    'Yes' if subtopic.get('mcat_relevant', False) else 'No',
                    '; '.join(subtopic.get('prerequisites', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('objectives', [])).replace('\t', ' '),
                    'Problem solving; Conceptual understanding; Application',
                    subtopic.get('source_book', '').replace('\t', ' '),
                    subtopic.get('extraction_strategy', 'unknown')
                ]
                f.write('\t'.join(row_data) + '\n')
        
        # Print comprehensive results
        print("\n" + "="*80)
        print("🎓 ADAPTIVE CURRICULUM GENERATION COMPLETED")
        print("="*80)
        
        metadata = result['adaptive_metadata']
        print(f"📊 Total Subtopics Generated: {result['total_subtopics']:,}")
        print(f"🔄 Iterations Required: {metadata['iterations_run']}")
        print(f"⭐ Final Quality Score: {metadata['final_quality']:.3f}")
        print(f"🎯 Quality Target Achieved: {'✅ YES' if metadata['target_achieved'] else '❌ NO'}")
        
        print(f"\n📈 Quality Progress: {metadata['quality_history']}")
        
        print(f"\n🔧 JSON Parsing Strategy Success Rates:")
        for strategy, rate in metadata['parsing_success_rates'].items():
            if rate > 0:
                print(f"   • {strategy}: {rate:.2f}")
        
        print(f"\n📊 Learning Log: {metadata['learning_log_size']} parsing attempts recorded")
        
        print(f"\n📁 Files Generated:")
        print(f"   📄 JSON: {output_file}")
        print(f"   📄 TSV:  {tsv_file}")
        
        logger.info("✅ Adaptive curriculum generation completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error in adaptive curriculum generation: {e}", exc_info=True)
        logger.info("Attempting to continue with basic generation")
        try:
            generate_curriculum(discipline=discipline)
        except Exception as e2:
            logger.error(f"Basic generation also failed: {e2}", exc_info=True)
        return None

def run_comprehensive_curriculum(discipline: str = "Physics"):
    """Run the comprehensive curriculum generation system."""
    try:
        if ComprehensiveCurriculumSystem is None:
            logger.error("ComprehensiveCurriculumSystem not imported - check if comprehensive_curriculum_system.py exists")
            logger.info("Falling back to basic curriculum generation")
            generate_curriculum(discipline=discipline)
            return None
    
        logger.info("🚀 Starting COMPREHENSIVE curriculum generation mode")
        logger.info("📊 Target: ~6,000+ fine-grained subtopics across all educational levels")
        
        system = ComprehensiveCurriculumSystem()
        physics_path = Path("/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books/english/Physics")
        
        if not physics_path.exists():
            logger.error(f"Physics books directory not found: {physics_path}")
            logger.info("Please check the path to physics books")
            return None
        
        logger.info(f"Processing physics books from: {physics_path}")
        result = system.generate_comprehensive_curriculum(physics_path)
        
        # Save comprehensive curriculum
        output_file = CURRICULUM_DIR / f"{discipline}_comprehensive_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved comprehensive curriculum to: {output_file}")
        
        # Save TSV with all columns
        tsv_file = CURRICULUM_DIR / f"{discipline}_comprehensive_curriculum.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            # Write comprehensive header
            f.write("Topic\tSubtopic\tDescription\tEducational_Level\tDepth_Level\tDomain\tDifficulty\tDuration_Hours\tIs_Core\tPedagogical_Order\tMCAT_Relevant\tPrerequisites\tLearning_Objectives\tAssessment_Methods\tSource_Books\n")
            
            for subtopic in result['subtopics']:
                # Format data for TSV
                row_data = [
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('description', '').replace('\t', ' ').replace('\n', ' '),
                    subtopic.get('educational_level', ''),
                    str(subtopic.get('depth_level', '')),
                    subtopic.get('domain', ''),
                    str(subtopic.get('difficulty', '')),
                    str(subtopic.get('duration_hours', '')),
                    'Core' if subtopic.get('is_core', True) else 'Elective',
                    str(subtopic.get('pedagogical_order', '')),
                    'Yes' if subtopic.get('mcat_relevant', False) else 'No',
                    '; '.join(subtopic.get('prerequisites', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('learning_objectives', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('assessment_methods', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('source_books', [])).replace('\t', ' ')
                ]
                f.write('\t'.join(row_data) + '\n')
        
        # Print summary
        print("\n" + "="*80)
        print("🎓 COMPREHENSIVE CURRICULUM GENERATION COMPLETED")
        print("="*80)
        summary = result['curriculum_summary']
        print(f"📊 Total Subtopics Generated: {summary['total_subtopics']:,}")
        print(f"🎯 Target Achievement: {'✅ EXCEEDED' if summary['target_achieved'] else '❌ Below Target'}")
        print(f"⭐ Quality Score: {summary['quality_score']:.2f}/1.00")
        print(f"🏆 High Quality Standard: {'✅ MET' if summary['high_quality'] else '❌ Needs Improvement'}")
        
        print(f"\n📁 Files Generated:")
        print(f"   📄 JSON: {output_file}")
        print(f"   📄 TSV:  {tsv_file}")
        
        print(f"\n📈 Statistics by Educational Level:")
        for level, count in result['statistics']['by_level'].items():
            print(f"   {level.replace('_', ' ').title()}: {count:,} subtopics")
        
        print(f"\n🔬 Statistics by Domain:")
        for domain, count in result['statistics']['by_domain'].items():
            print(f"   {domain.replace('_', ' ').title()}: {count:,} subtopics")
        
        if result['refinement_suggestions']:
            print(f"\n🔧 Quality Refinement Suggestions:")
            for suggestion in result['refinement_suggestions']:
                print(f"   • {suggestion}")
    
        logger.info("✅ Comprehensive curriculum generation completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error in comprehensive curriculum generation: {e}", exc_info=True)
        logger.info("Attempting to continue with adaptive generation only")
        return None

def run_toc_aware_curriculum(discipline: str = "Physics"):
    """Run the TOC-aware curriculum generation system."""
    try:
        if TOCAwareCurriculumSystem is None:
            logger.error("TOCAwareCurriculumSystem not imported - check if toc_aware_curriculum_system.py exists")
            logger.info("Falling back to comprehensive curriculum generation")
            return run_comprehensive_curriculum(discipline=discipline)
    
        logger.info("🎯 Starting TOC-AWARE curriculum generation mode")
        logger.info("📖 Extracting meaningful subtopics directly from Table of Contents")
        logger.info("🎯 Target: ~1,000 meaningful, well-ordered subtopics using actual book structure")
        
        system = TOCAwareCurriculumSystem()
        books_path = Path("/Users/davidlary/Dropbox/Environments/Code/Curriculum-AI/Books")
        
        if not books_path.exists():
            logger.error(f"Books directory not found: {books_path}")
            logger.info("Please check the path to books")
            return None
        
        logger.info(f"Processing books from: {books_path}")
        result = system.generate_toc_aware_curriculum(books_path)
    
        if 'error' in result:
            logger.error(f"TOC-aware generation failed: {result['error']}")
            return None
        
        # Save TOC-aware curriculum
        output_file = CURRICULUM_DIR / f"{discipline}_toc_aware_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved TOC-aware curriculum to: {output_file}")
        
        # Save TSV with all columns
        tsv_file = CURRICULUM_DIR / f"{discipline}_toc_aware_curriculum.tsv"
        with open(tsv_file, 'w', encoding='utf-8') as f:
            # Write comprehensive header
            f.write("Pedagogical_Order\tTopic\tSubtopic\tDescription\tChapter\tSection\tEducational_Level\tDepth_Level\tDomain\tDifficulty\tDuration_Hours\tIs_Core\tMCAT_Relevant\tPrerequisites\tLearning_Objectives\tAssessment_Methods\tSource_Books\n")
            
            for subtopic in result['subtopics']:
                # Format data for TSV
                row_data = [
                    str(subtopic.get('pedagogical_order', '')),
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('name', '').replace('\t', ' '),
                    subtopic.get('description', '').replace('\t', ' ').replace('\n', ' '),
                    subtopic.get('chapter_title', '').replace('\t', ' '),
                    subtopic.get('section_title', '').replace('\t', ' '),
                    subtopic.get('educational_level', ''),
                    str(subtopic.get('depth_level', '')),
                    subtopic.get('domain', ''),
                    str(subtopic.get('difficulty', '')),
                    str(subtopic.get('duration_hours', '')),
                    'Core' if subtopic.get('is_core', True) else 'Elective',
                    'Yes' if subtopic.get('mcat_relevant', False) else 'No',
                    '; '.join(subtopic.get('prerequisites', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('learning_objectives', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('assessment_methods', [])).replace('\t', ' '),
                    '; '.join(subtopic.get('source_books', [])).replace('\t', ' ')
                ]
                f.write('\t'.join(row_data) + '\n')
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("🎓 TOC-AWARE CURRICULUM GENERATION COMPLETED")
        print("="*80)
        
        print(f"📊 Total Subtopics Generated: {result['total_subtopics']:,}")
        print(f"🎯 Target Achievement: {'✅ ACHIEVED' if result['target_achieved'] else '❌ Below Target'}")
        print(f"⭐ Quality Score: {result['quality_metrics']['overall_quality']:.3f}/1.000")
        print(f"🏆 High Quality Standard: {'✅ MET' if result['high_quality'] else '❌ Needs Improvement'}")
        print(f"🔄 Iterations Required: {result['iterations']}")
        
        print(f"\n📁 Files Generated:")
        print(f"   📄 JSON: {output_file}")
        print(f"   📄 TSV:  {tsv_file}")
        
        print(f"\n📈 Statistics by Educational Level:")
        for level, count in result['statistics']['by_level'].items():
            print(f"   {level.replace('_', ' ').title()}: {count:,} subtopics")
        
        print(f"\n🔬 Statistics by Domain:")
        for domain, count in result['statistics']['by_domain'].items():
            print(f"   {domain.replace('_', ' ').title()}: {count:,} subtopics")
        
        print(f"\n📚 Books Processed:")
        for book in result['books_processed']:
            print(f"   {book['name']} ({book['level']}): {book['subtopics_generated']} subtopics")
        
        print(f"\n🎯 Quality Metrics:")
        for metric, value in result['quality_metrics'].items():
            if metric != 'overall_quality':
                print(f"   {metric.replace('_', ' ').title()}: {value:.3f}")
    
        logger.info("✅ TOC-aware curriculum generation completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error in TOC-aware curriculum generation: {e}", exc_info=True)
        logger.info("Attempting to continue with comprehensive generation")
        return run_comprehensive_curriculum(discipline=discipline)

def run_enhanced_comprehensive_curriculum(discipline: str = "Physics"):
    """Run the enhanced comprehensive curriculum generation system."""
    try:
        # Import the enhanced system
        import importlib.util
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "enhanced_comprehensive_curriculum", 
            base_dir / "scripts" / "enhanced_comprehensive_curriculum.py"
        )
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)
        
        logger.info("🚀 Starting ENHANCED COMPREHENSIVE curriculum generation")
        logger.info("📊 Merging topics across educational levels with expert pedagogical ordering")
        logger.info("🎯 Target: Comprehensive superset curriculum with optimal sequencing")
        
        system = enhanced_module.EnhancedComprehensiveCurriculumSystem()
        result = system.create_comprehensive_curriculum(discipline=discipline)
        
        if 'error' in result:
            logger.error(f"Enhanced comprehensive generation failed: {result['error']}")
            return None
        
        # Save enhanced comprehensive curriculum
        curriculum_dir = base_dir / "Curriculum"
        output_file = curriculum_dir / f"{discipline}_enhanced_comprehensive_curriculum.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved enhanced comprehensive curriculum to: {output_file}")
        
        logger.info("✅ Enhanced comprehensive curriculum generation completed successfully!")
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced comprehensive curriculum generation: {e}", exc_info=True)
        logger.info("Falling back to TOC-aware curriculum generation")
        return run_toc_aware_curriculum(discipline=discipline)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate TOC-aware, meaningful physics curriculum (DEFAULT: uses actual Table of Contents from textbooks)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/generate_curriculum.py                    # Default: TOC-aware (~1,000 meaningful subtopics)
  python scripts/generate_curriculum.py --toc-aware        # TOC-aware with actual chapter/section names
  python scripts/generate_curriculum.py --adaptive         # Adaptive only (error-free JSON parsing)
  python scripts/generate_curriculum.py --comprehensive    # Comprehensive only (~6,000 subtopics)
  python scripts/generate_curriculum.py --basic            # Legacy workflow
        '''
    )
    parser.add_argument('--discipline', '-d', default='Physics',
                       help='Discipline to generate curriculum for (default: Physics)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (optional - uses OPENAI_API_KEY env var by default)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'xai'],
                       help='AI provider for curriculum generation (default: openai)')
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced comprehensive curriculum with optimal sequencing and cross-level merging (RECOMMENDED)')
    parser.add_argument('--toc-aware', action='store_true',
                       help='Use TOC-aware system extracting meaningful subtopics from actual Table of Contents (~1,000 subtopics)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Use comprehensive curriculum system with ~6,000+ fine-grained subtopics')
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive, data-driven curriculum system with iterative quality improvement and error-free JSON parsing')
    parser.add_argument('--basic', action='store_true',
                       help='Use basic curriculum generation workflow (non-adaptive, legacy system)')
    
    args = parser.parse_args()
    
    # Default behavior: use enhanced comprehensive system for optimal curriculum
    try:
        if args.enhanced:
            logger.info("🚀 Running ENHANCED COMPREHENSIVE curriculum generation")
            run_enhanced_comprehensive_curriculum(discipline=args.discipline)
        elif args.basic:
            logger.info("🔧 Running BASIC curriculum generation")
            generate_curriculum(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)
        elif args.adaptive:
            logger.info("🔬 Running ADAPTIVE curriculum generation only")
            run_adaptive_curriculum(discipline=args.discipline)
        elif args.comprehensive:
            logger.info("📊 Running COMPREHENSIVE curriculum generation only")
            run_comprehensive_curriculum(discipline=args.discipline)
        elif getattr(args, 'toc-aware', False) or getattr(args, 'toc_aware', False):
            logger.info("🎯 Running TOC-AWARE curriculum generation")
            run_toc_aware_curriculum(discipline=args.discipline)
        else:
            # Default: Run TOC-aware system for meaningful subtopics
            logger.info("🎯 Running DEFAULT mode: TOC-aware curriculum generation")
            logger.info("   📖 Extracting meaningful subtopics from actual Table of Contents")
            logger.info("   🎯 Target: ~1,000 well-ordered, meaningful subtopics")
            logger.info("   Use --basic for legacy, --adaptive for adaptive only, --comprehensive for volume")
            
            # Run TOC-aware system for meaningful curriculum generation
            toc_result = run_toc_aware_curriculum(discipline=args.discipline)
            
            if not toc_result:
                logger.warning("⚠️ TOC-aware system encountered issues - check logs for details")
                logger.info("Attempting adaptive curriculum generation as fallback")
                try:
                    adaptive_result = run_adaptive_curriculum(discipline=args.discipline)
                    if not adaptive_result:
                        logger.info("Attempting basic curriculum generation as final fallback")
                        generate_curriculum(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)
                except Exception as fallback_error:
                    logger.error(f"Fallback failed: {fallback_error}")
                    logger.info("Attempting basic curriculum generation as emergency fallback")
                    generate_curriculum(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)
                
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        logger.info("Attempting basic curriculum generation as emergency fallback")
        try:
            generate_curriculum(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)
        except Exception as e2:
            logger.error(f"Emergency fallback also failed: {e2}", exc_info=True)
            print("\n❌ All curriculum generation methods failed. Please check the error log for details.")