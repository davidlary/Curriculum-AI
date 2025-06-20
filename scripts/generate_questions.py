import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import random
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.vectorstores import Qdrant
    from langchain.embeddings import OpenAIEmbeddings
    import qdrant_client
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.info("Install with: pip install langchain openai qdrant-client")
    exit(1)

CURRICULUM_DIR = Path("Curriculum")
QUESTIONS_DIR = Path("Questions")
QUESTIONS_DIR.mkdir(exist_ok=True)
CHUNKS_DIR = Path("Chunks")

class BloomLevel(Enum):
    """Bloom's Taxonomy levels for question generation."""
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"

class QuestionType(Enum):
    """Types of questions to generate."""
    MULTIPLE_CHOICE = "Multiple Choice"
    TRUE_FALSE = "True/False"
    SHORT_ANSWER = "Short Answer"
    ESSAY = "Essay"
    CALCULATION = "Calculation"

class QuestionGenerator:
    def __init__(self, openai_api_key: str = None, provider: str = "openai"):
        """Initialize the question generator with AI models."""
        # Check environment variables for API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")
        
        self.provider = provider.lower()
        
        # Initialize models based on provider
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.4,
                openai_api_key=self.openai_api_key
            )
            self.fast_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                openai_api_key=self.openai_api_key
            )
        elif self.provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.4,
                    anthropic_api_key=self.anthropic_api_key
                )
                self.fast_llm = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    temperature=0.2,
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
                temperature=0.4,
                openai_api_key=self.xai_api_key,
                openai_api_base="https://api.x.ai/v1"
            )
            self.fast_llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.2,
                openai_api_key=self.xai_api_key,
                openai_api_base="https://api.x.ai/v1"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, xai")
        
        # Initialize embeddings for context retrieval
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        
    def load_curriculum(self, discipline: str) -> Optional[Dict[str, Any]]:
        """Load curriculum data for a discipline."""
        curriculum_file = CURRICULUM_DIR / f"{discipline}_curriculum.json"
        
        if not curriculum_file.exists():
            logger.error(f"Curriculum file not found: {curriculum_file}")
            logger.info(f"Run generate_curriculum.py first for {discipline}")
            return None
        
        try:
            with open(curriculum_file, 'r', encoding='utf-8') as f:
                curriculum = json.load(f)
            logger.info(f"Loaded curriculum with {len(curriculum.get('subtopics', []))} subtopics")
            return curriculum
        except Exception as e:
            logger.error(f"Error loading curriculum: {e}")
            return None
    
    def load_context_chunks(self, discipline: str) -> List[Dict[str, Any]]:
        """Load textbook chunks for context."""
        chunks = []
        chunk_files = list(CHUNKS_DIR.glob(f"{discipline}_*.jsonl"))
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            chunk = json.loads(line.strip())
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
        
        logger.info(f"Loaded {len(chunks)} context chunks")
        return chunks
    
    def get_relevant_context(self, subtopic: Dict[str, Any], chunks: List[Dict[str, Any]], max_chunks: int = 3) -> str:
        """Get relevant textbook context for a subtopic."""
        subtopic_text = f"{subtopic['name']} {subtopic['description']}"
        
        # Simple text matching for context (in production, use vector search)
        relevant_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            subtopic_words = subtopic_text.lower().split()
            
            # Calculate relevance score
            score = sum(1 for word in subtopic_words if word in chunk_text)
            if score > 0:
                relevant_chunks.append((chunk, score))
        
        # Sort by relevance and take top chunks
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in relevant_chunks[:max_chunks]]
        
        # Combine context
        context = "\n\n".join([chunk['text'][:800] for chunk in top_chunks])
        return context[:2000]  # Limit context length
    
    def generate_questions_for_subtopic(self, subtopic: Dict[str, Any], context: str, 
                                      bloom_levels: List[BloomLevel], 
                                      question_types: List[QuestionType],
                                      questions_per_level: int = 2) -> List[Dict[str, Any]]:
        """Generate questions for a specific subtopic across different Bloom levels."""
        questions = []
        
        for bloom_level in bloom_levels:
            for _ in range(questions_per_level):
                question_type = random.choice(question_types)
                
                question_prompt = self.create_question_prompt(bloom_level, question_type)
                
                try:
                    response = self.llm.invoke(
                        question_prompt.format_messages(
                            subtopic_name=subtopic["name"],
                            subtopic_description=subtopic["description"],
                            learning_objectives="\n".join(subtopic.get("objectives", [])),
                            context=context,
                            bloom_level=bloom_level.value,
                            question_type=question_type.value
                        )
                    )
                    
                    # Parse the response
                    question_data = self.parse_question_response(response.content, bloom_level, question_type, subtopic)
                    if question_data:
                        questions.append(question_data)
                        
                except Exception as e:
                    logger.error(f"Error generating {bloom_level.value} question for '{subtopic['name']}': {e}")
                    continue
        
        return questions
    
    def create_question_prompt(self, bloom_level: BloomLevel, question_type: QuestionType) -> ChatPromptTemplate:
        """Create a prompt template for question generation based on Bloom level and type."""
        
        bloom_descriptions = {
            BloomLevel.REMEMBER: "Test recall of facts, terms, basic concepts, or answers",
            BloomLevel.UNDERSTAND: "Test comprehension and ability to explain ideas or concepts",
            BloomLevel.APPLY: "Test ability to use information in new situations",
            BloomLevel.ANALYZE: "Test ability to break down information and examine relationships",
            BloomLevel.EVALUATE: "Test ability to make judgments based on criteria and standards",
            BloomLevel.CREATE: "Test ability to produce new or original work"
        }
        
        question_instructions = {
            QuestionType.MULTIPLE_CHOICE: "Create a multiple choice question with 4 options (A, B, C, D). Mark the correct answer.",
            QuestionType.TRUE_FALSE: "Create a true/false question with explanation for the correct answer.",
            QuestionType.SHORT_ANSWER: "Create a short answer question (2-3 sentences expected).",
            QuestionType.ESSAY: "Create an essay question requiring detailed analysis and explanation.",
            QuestionType.CALCULATION: "Create a calculation or problem-solving question with numerical answer."
        }
        
        system_message = f"""You are an expert educational assessment designer. Create a high-quality {question_type.value} question at the {bloom_level.value} level of Bloom's Taxonomy.

Bloom Level Goal: {bloom_descriptions[bloom_level]}
Question Type: {question_instructions[question_type]}

Requirements:
1. Base the question on the provided subtopic and context
2. Ensure the question matches the specified Bloom level
3. Make the question clear, unambiguous, and academically rigorous
4. Provide complete answer/explanation
5. Include difficulty rating (1-5) and estimated time to complete

Format your response as JSON with these fields:
- "question": the question text
- "type": "{question_type.value}"
- "bloom_level": "{bloom_level.value}"
- "options": array of options (for multiple choice) or null
- "correct_answer": the correct answer
- "explanation": detailed explanation of the answer
- "difficulty": number from 1-5
- "estimated_time_minutes": estimated time to complete
- "learning_objective": which learning objective this tests"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", """Subtopic: {subtopic_name}
Description: {subtopic_description}
Learning Objectives:
{learning_objectives}

Relevant Context:
{context}

Generate a {bloom_level} level {question_type} question.""")
        ])
    
    def parse_question_response(self, response: str, bloom_level: BloomLevel, 
                              question_type: QuestionType, subtopic: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse the LLM response into a structured question object."""
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0]
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0]
            
            question_data = json.loads(response)
            
            # Add metadata
            question_data["subtopic"] = subtopic["name"]
            question_data["parent_topic"] = subtopic.get("parent_topic", "")
            question_data["id"] = f"{subtopic['name'].replace(' ', '_')}_{bloom_level.name}_{len(str(random.randint(1000, 9999)))}"
            
            # Validate required fields
            required_fields = ["question", "correct_answer", "explanation"]
            if not all(field in question_data for field in required_fields):
                logger.warning(f"Missing required fields in question response")
                return None
            
            return question_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse question JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing question response: {e}")
            return None
    
    def generate_assessment_bank(self, curriculum: Dict[str, Any], context_chunks: List[Dict[str, Any]], 
                               questions_per_subtopic: int = 6) -> Dict[str, Any]:
        """Generate a comprehensive assessment bank for the entire curriculum."""
        logger.info("Generating comprehensive assessment bank...")
        
        # Define question distribution
        bloom_levels = [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND, BloomLevel.APPLY, 
                       BloomLevel.ANALYZE, BloomLevel.EVALUATE, BloomLevel.CREATE]
        question_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, 
                         QuestionType.TRUE_FALSE, QuestionType.ESSAY]
        
        all_questions = []
        subtopics_processed = 0
        
        for subtopic in curriculum.get("subtopics", []):
            logger.info(f"Generating questions for: {subtopic['name']}")
            
            # Get relevant context
            context = self.get_relevant_context(subtopic, context_chunks)
            
            # Generate questions for this subtopic
            subtopic_questions = self.generate_questions_for_subtopic(
                subtopic, context, bloom_levels, question_types, 
                questions_per_level=1  # 1 question per Bloom level
            )
            
            all_questions.extend(subtopic_questions)
            subtopics_processed += 1
            
            if subtopics_processed % 5 == 0:
                logger.info(f"Processed {subtopics_processed}/{len(curriculum.get('subtopics', []))} subtopics")
        
        # Organize questions by topic and bloom level
        assessment_bank = {
            "discipline": curriculum["discipline"],
            "total_questions": len(all_questions),
            "questions": all_questions,
            "questions_by_topic": self.organize_by_topic(all_questions),
            "questions_by_bloom_level": self.organize_by_bloom_level(all_questions),
            "questions_by_type": self.organize_by_type(all_questions),
            "metadata": {
                "subtopics_covered": len(curriculum.get("subtopics", [])),
                "avg_questions_per_subtopic": len(all_questions) / max(len(curriculum.get("subtopics", [])), 1),
                "bloom_levels_covered": list(set([q["bloom_level"] for q in all_questions])),
                "question_types_used": list(set([q["type"] for q in all_questions]))
            }
        }
        
        logger.info(f"Generated {len(all_questions)} questions across {subtopics_processed} subtopics")
        return assessment_bank
    
    def organize_by_topic(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize questions by parent topic."""
        by_topic = {}
        for question in questions:
            topic = question.get("parent_topic", "Unknown")
            if topic not in by_topic:
                by_topic[topic] = []
            by_topic[topic].append(question)
        return by_topic
    
    def organize_by_bloom_level(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize questions by Bloom taxonomy level."""
        by_bloom = {}
        for question in questions:
            bloom = question.get("bloom_level", "Unknown")
            if bloom not in by_bloom:
                by_bloom[bloom] = []
            by_bloom[bloom].append(question)
        return by_bloom
    
    def organize_by_type(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize questions by question type."""
        by_type = {}
        for question in questions:
            qtype = question.get("type", "Unknown")
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(question)
        return by_type
    
    def save_assessment_bank(self, assessment_bank: Dict[str, Any]):
        """Save the assessment bank to files."""
        discipline = assessment_bank["discipline"]
        
        # Save complete assessment bank
        bank_file = QUESTIONS_DIR / f"{discipline}_assessment_bank.json"
        with open(bank_file, 'w', encoding='utf-8') as f:
            json.dump(assessment_bank, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved assessment bank to: {bank_file}")
        
        # Save organized question sets
        for topic, questions in assessment_bank["questions_by_topic"].items():
            topic_file = QUESTIONS_DIR / f"{discipline}_{topic.replace(' ', '_')}_questions.json"
            with open(topic_file, 'w', encoding='utf-8') as f:
                json.dump({"topic": topic, "questions": questions}, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        self.generate_assessment_report(assessment_bank)
    
    def generate_assessment_report(self, assessment_bank: Dict[str, Any]):
        """Generate a human-readable assessment report."""
        discipline = assessment_bank["discipline"]
        report_file = QUESTIONS_DIR / f"{discipline}_assessment_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Assessment Bank Report: {discipline}\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Questions: {assessment_bank['total_questions']}\n")
            f.write(f"Subtopics Covered: {assessment_bank['metadata']['subtopics_covered']}\n")
            f.write(f"Avg Questions per Subtopic: {assessment_bank['metadata']['avg_questions_per_subtopic']:.1f}\n\n")
            
            # Questions by Bloom Level
            f.write("QUESTIONS BY BLOOM LEVEL\n")
            f.write("-" * 25 + "\n")
            for bloom_level, questions in assessment_bank["questions_by_bloom_level"].items():
                f.write(f"{bloom_level}: {len(questions)} questions\n")
            f.write("\n")
            
            # Questions by Type
            f.write("QUESTIONS BY TYPE\n")
            f.write("-" * 17 + "\n")
            for qtype, questions in assessment_bank["questions_by_type"].items():
                f.write(f"{qtype}: {len(questions)} questions\n")
            f.write("\n")
            
            # Questions by Topic
            f.write("QUESTIONS BY TOPIC\n")
            f.write("-" * 18 + "\n")
            for topic, questions in assessment_bank["questions_by_topic"].items():
                f.write(f"{topic}: {len(questions)} questions\n")
            
        logger.info(f"Saved assessment report to: {report_file}")

def generate_questions(discipline: str = "Physics", openai_api_key: str = None, provider: str = "openai"):
    """Main function to generate questions for a discipline."""
    logger.info(f"Starting question generation for discipline: {discipline} using {provider}")
    
    try:
        # Initialize generator (will automatically use environment variables if available)
        generator = QuestionGenerator(openai_api_key=openai_api_key, provider=provider)
        
        # Load curriculum
        curriculum = generator.load_curriculum(discipline)
        if not curriculum:
            return
        
        # Load context chunks
        context_chunks = generator.load_context_chunks(discipline)
        if not context_chunks:
            logger.warning("No context chunks found - questions may be less specific")
            context_chunks = []
        
        # Generate assessment bank
        assessment_bank = generator.generate_assessment_bank(curriculum, context_chunks)
        
        # Save results
        generator.save_assessment_bank(assessment_bank)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("QUESTION GENERATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Discipline: {discipline}")
        logger.info(f"Total Questions: {assessment_bank['total_questions']}")
        logger.info(f"Subtopics Covered: {assessment_bank['metadata']['subtopics_covered']}")
        logger.info(f"Bloom Levels: {', '.join(assessment_bank['metadata']['bloom_levels_covered'])}")
        logger.info(f"Question Types: {', '.join(assessment_bank['metadata']['question_types_used'])}")
        logger.info("✅ Question generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during question generation: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate assessment questions using Bloom taxonomy')
    parser.add_argument('--discipline', '-d', default='Physics',
                       help='Discipline to generate questions for (default: Physics)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (optional - uses OPENAI_API_KEY env var by default)')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic', 'xai'],
                       help='AI provider for question generation (default: openai)')
    
    args = parser.parse_args()
    
    generate_questions(discipline=args.discipline, openai_api_key=args.openai_api_key, provider=args.provider)