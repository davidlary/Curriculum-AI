import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, TypedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
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
    logger.info("Install with: pip install langchain langgraph openai networkx matplotlib")
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
    def __init__(self, openai_api_key: str = None):
        """Initialize the curriculum generator with OpenAI models."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=self.openai_api_key
        )
        
        self.fast_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=self.openai_api_key
        )
        
        # Create the workflow graph
        self.workflow = self.create_workflow()
        
    def load_chunks(self, discipline: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL files for a specific discipline."""
        chunks = []
        chunk_files = list(CHUNKS_DIR.glob(f"{discipline}_*.jsonl"))
        
        if not chunk_files:
            logger.warning(f"No chunk files found for discipline: {discipline}")
            return chunks
            
        for chunk_file in chunk_files:
            logger.info(f"Loading chunks from: {chunk_file}")
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            chunk = json.loads(line.strip())
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num} in {chunk_file}: {e}")
            except Exception as e:
                logger.error(f"Error reading {chunk_file}: {e}")
                
        logger.info(f"Loaded {len(chunks)} chunks for discipline: {discipline}")
        return chunks
    
    def extract_topics(self, state: CurriculumState) -> CurriculumState:
        """Extract main topics from chunks using LLM."""
        logger.info("Extracting main topics from chunks...")
        
        # Sample chunks for topic extraction (first 50 to avoid token limits)
        sample_chunks = state["chunks"][:50]
        chunk_texts = [chunk["text"][:500] for chunk in sample_chunks if chunk.get("text")]
        
        topic_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert curriculum designer. Extract the main topics from textbook content.
            
            For each topic, provide:
            1. Topic name (clear and concise)
            2. Description (2-3 sentences)
            3. Estimated difficulty level (1-5, where 1=beginner, 5=advanced)
            
            Return exactly 15-25 main topics that cover the discipline comprehensively.
            Format as JSON array with objects having 'name', 'description', 'difficulty' fields."""),
            ("human", "Discipline: {discipline}\n\nTextbook content samples:\n{content}")
        ])
        
        content = "\n\n".join(chunk_texts[:10])  # Use first 10 chunks
        
        try:
            response = self.llm.invoke(
                topic_extraction_prompt.format_messages(
                    discipline=state["discipline"],
                    content=content
                )
            )
            
            # Parse JSON response
            topics_text = response.content.strip()
            if topics_text.startswith("```json"):
                topics_text = topics_text.split("```json")[1].split("```")[0]
            elif topics_text.startswith("```"):
                topics_text = topics_text.split("```")[1].split("```")[0]
            
            topics = json.loads(topics_text)
            state["topics"] = topics
            logger.info(f"Extracted {len(topics)} main topics")
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            state["error"] = str(e)
            state["topics"] = []
        
        return state
    
    def generate_subtopics(self, state: CurriculumState) -> CurriculumState:
        """Generate detailed subtopics for each main topic."""
        logger.info("Generating subtopics for each main topic...")
        
        subtopics = []
        
        for topic in state["topics"]:
            subtopic_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert curriculum designer. Generate comprehensive subtopics for a given main topic.
                
                For each subtopic, provide:
                1. Subtopic name
                2. Description (1-2 sentences)
                3. Learning objectives (2-3 bullet points)
                4. Estimated duration (in hours)
                5. Prerequisites (list of other subtopic names, if any)
                
                Generate 8-15 subtopics that thoroughly cover the main topic.
                Format as JSON array with objects having 'name', 'description', 'objectives', 'duration', 'prerequisites' fields."""),
                ("human", "Main Topic: {topic_name}\nDescription: {topic_description}\nDifficulty: {difficulty}")
            ])
            
            try:
                response = self.fast_llm.invoke(
                    subtopic_prompt.format_messages(
                        topic_name=topic["name"],
                        topic_description=topic["description"],
                        difficulty=topic["difficulty"]
                    )
                )
                
                # Parse JSON response
                subtopics_text = response.content.strip()
                if subtopics_text.startswith("```json"):
                    subtopics_text = subtopics_text.split("```json")[1].split("```")[0]
                elif subtopics_text.startswith("```"):
                    subtopics_text = subtopics_text.split("```")[1].split("```")[0]
                
                topic_subtopics = json.loads(subtopics_text)
                
                # Add parent topic information
                for subtopic in topic_subtopics:
                    subtopic["parent_topic"] = topic["name"]
                    subtopic["topic_difficulty"] = topic["difficulty"]
                
                subtopics.extend(topic_subtopics)
                logger.info(f"Generated {len(topic_subtopics)} subtopics for '{topic['name']}'")
                
            except Exception as e:
                logger.error(f"Error generating subtopics for '{topic['name']}': {e}")
                continue
        
        state["subtopics"] = subtopics
        logger.info(f"Generated {len(subtopics)} total subtopics")
        return state
    
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
                
                prereqs = json.loads(prereqs_text)
                
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
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for curriculum generation."""
        workflow = StateGraph(CurriculumState)
        
        # Add nodes
        workflow.add_node("extract_topics", self.extract_topics)
        workflow.add_node("generate_subtopics", self.generate_subtopics)
        workflow.add_node("build_prerequisites", self.build_prerequisite_graph)
        workflow.add_node("create_graph", self.create_curriculum_graph)
        
        # Add edges
        workflow.set_entry_point("extract_topics")
        workflow.add_edge("extract_topics", "generate_subtopics")
        workflow.add_edge("generate_subtopics", "build_prerequisites")
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
        
        # Save learning path
        if state["curriculum_graph"].get("learning_path"):
            learning_path_file = CURRICULUM_DIR / f"{discipline}_learning_path.txt"
            with open(learning_path_file, 'w', encoding='utf-8') as f:
                f.write(f"Learning Path for {discipline}\n")
                f.write("=" * 50 + "\n\n")
                for i, subtopic in enumerate(state["curriculum_graph"]["learning_path"], 1):
                    f.write(f"{i:2d}. {subtopic}\n")
            
            logger.info(f"Saved learning path to: {learning_path_file}")

def generate_curriculum(discipline: str = "Physics", openai_api_key: str = None):
    """Main function to generate curriculum for a discipline."""
    logger.info(f"Starting curriculum generation for discipline: {discipline}")
    
    try:
        # Initialize generator
        generator = CurriculumGenerator(openai_api_key=openai_api_key)
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate curriculum using LangGraph workflow')
    parser.add_argument('--discipline', '-d', default='Physics',
                       help='Discipline to generate curriculum for (default: Physics)')
    parser.add_argument('--openai-api-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    generate_curriculum(discipline=args.discipline, openai_api_key=args.openai_api_key)