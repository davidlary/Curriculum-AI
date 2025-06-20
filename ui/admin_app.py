import streamlit as st
import json
import os
import subprocess
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

# Configure page
st.set_page_config(
    page_title="Curriculum AI Admin Panel",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directories
CHUNKS_DIR = Path("Chunks")
CURRICULUM_DIR = Path("Curriculum")
QUESTIONS_DIR = Path("Questions")
EMBEDDINGS_DIR = Path("Embeddings")

class CurriculumAdmin:
    def __init__(self):
        self.disciplines = self.get_available_disciplines()
    
    def get_available_disciplines(self) -> List[str]:
        """Get list of disciplines with processed chunks."""
        if not CHUNKS_DIR.exists():
            return []
        
        disciplines = set()
        for chunk_file in CHUNKS_DIR.glob("*_*.jsonl"):
            discipline = chunk_file.stem.split('_')[0]
            disciplines.add(discipline)
        
        return sorted(list(disciplines))
    
    def load_curriculum(self, discipline: str) -> Optional[Dict[str, Any]]:
        """Load curriculum data for a discipline."""
        curriculum_file = CURRICULUM_DIR / f"{discipline}_curriculum.json"
        if not curriculum_file.exists():
            return None
        
        try:
            with open(curriculum_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading curriculum: {e}")
            return None
    
    def load_questions(self, discipline: str) -> Optional[Dict[str, Any]]:
        """Load assessment questions for a discipline."""
        questions_file = QUESTIONS_DIR / f"{discipline}_assessment_bank.json"
        if not questions_file.exists():
            return None
        
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading questions: {e}")
            return None
    
    def get_chunk_statistics(self, discipline: str) -> Dict[str, Any]:
        """Get statistics about processed chunks."""
        stats = {"total_chunks": 0, "levels": {}, "files": []}
        
        chunk_files = list(CHUNKS_DIR.glob(f"{discipline}_*.jsonl"))
        for chunk_file in chunk_files:
            level = chunk_file.stem.split('_', 1)[1]
            chunk_count = 0
            
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_count = sum(1 for line in f if line.strip())
                
                stats["levels"][level] = chunk_count
                stats["total_chunks"] += chunk_count
                stats["files"].append(str(chunk_file))
                
            except Exception as e:
                st.error(f"Error reading {chunk_file}: {e}")
        
        return stats
    
    def run_pipeline_step(self, step: str, discipline: str, openai_api_key: str) -> bool:
        """Run a specific pipeline step."""
        env = os.environ.copy()
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        
        commands = {
            "parse": ["python", "scripts/parse_textbooks.py", "--discipline", discipline],
            "embed": ["python", "scripts/embed_chunks.py", "--discipline", discipline],
            "curriculum": ["python", "scripts/generate_curriculum.py", "--discipline", discipline],
            "questions": ["python", "scripts/generate_questions.py", "--discipline", discipline]
        }
        
        if step not in commands:
            st.error(f"Unknown pipeline step: {step}")
            return False
        
        try:
            with st.spinner(f"Running {step} step..."):
                result = subprocess.run(
                    commands[step],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=300  # 5 minute timeout
                )
            
            if result.returncode == 0:
                st.success(f"✅ {step.title()} completed successfully!")
                if result.stdout:
                    with st.expander("View Output"):
                        st.code(result.stdout)
                return True
            else:
                st.error(f"❌ {step.title()} failed!")
                if result.stderr:
                    st.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            st.error(f"❌ {step.title()} timed out after 5 minutes")
            return False
        except Exception as e:
            st.error(f"❌ Error running {step}: {e}")
            return False

def main():
    st.title("🎓 Curriculum AI Admin Panel")
    st.markdown("---")
    
    admin = CurriculumAdmin()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key
        env_api_key = os.getenv("OPENAI_API_KEY", "")
        if env_api_key:
            st.success("✅ OpenAI API key found in environment variables")
            openai_api_key = env_api_key
        else:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for embedding and curriculum generation (or set OPENAI_API_KEY env var)"
            )
        
        # Discipline selection
        if admin.disciplines:
            selected_discipline = st.selectbox(
                "Select Discipline",
                admin.disciplines
            )
        else:
            st.warning("No processed disciplines found. Run textbook parsing first.")
            selected_discipline = st.text_input("Enter discipline name", value="Physics")
        
        st.markdown("---")
        
        # Pipeline Controls
        st.header("Pipeline Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Parse Textbooks", use_container_width=True):
                admin.run_pipeline_step("parse", selected_discipline, openai_api_key)
                st.rerun()
            
            if st.button("🧠 Generate Curriculum", use_container_width=True):
                if not openai_api_key:
                    st.error("OpenAI API key required!")
                else:
                    admin.run_pipeline_step("curriculum", selected_discipline, openai_api_key)
                    st.rerun()
        
        with col2:
            if st.button("📊 Embed Chunks", use_container_width=True):
                if not openai_api_key:
                    st.error("OpenAI API key required!")
                else:
                    admin.run_pipeline_step("embed", selected_discipline, openai_api_key)
                    st.rerun()
            
            if st.button("❓ Generate Questions", use_container_width=True):
                if not openai_api_key:
                    st.error("OpenAI API key required!")
                else:
                    admin.run_pipeline_step("questions", selected_discipline, openai_api_key)
                    st.rerun()
        
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
            if not openai_api_key:
                st.error("OpenAI API key required for full pipeline!")
            else:
                steps = ["parse", "embed", "curriculum", "questions"]
                for step in steps:
                    success = admin.run_pipeline_step(step, selected_discipline, openai_api_key)
                    if not success:
                        st.error(f"Pipeline stopped at {step} step")
                        break
                else:
                    st.success("🎉 Full pipeline completed successfully!")
                st.rerun()
    
    # Main content area
    if selected_discipline:
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📚 Curriculum", "❓ Questions", "📈 Analytics"])
        
        with tab1:
            st.header(f"Overview: {selected_discipline}")
            
            # Chunk statistics
            chunk_stats = admin.get_chunk_statistics(selected_discipline)
            
            if chunk_stats["total_chunks"] > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Chunks", chunk_stats["total_chunks"])
                
                with col2:
                    st.metric("Education Levels", len(chunk_stats["levels"]))
                
                with col3:
                    curriculum = admin.load_curriculum(selected_discipline)
                    if curriculum:
                        st.metric("Subtopics Generated", len(curriculum.get("subtopics", [])))
                    else:
                        st.metric("Subtopics Generated", "Not Generated", delta="Run curriculum generation")
                
                # Chunks by level chart
                if chunk_stats["levels"]:
                    st.subheader("Chunks by Education Level")
                    
                    levels_df = pd.DataFrame(
                        list(chunk_stats["levels"].items()),
                        columns=["Level", "Chunks"]
                    )
                    
                    fig = px.bar(levels_df, x="Level", y="Chunks", 
                               title="Distribution of Chunks by Education Level")
                    st.plotly_chart(fig, use_container_width=True)
                
                # File details
                with st.expander("Chunk Files Details"):
                    for file_path in chunk_stats["files"]:
                        st.text(file_path)
            
            else:
                st.warning(f"No chunks found for {selected_discipline}. Run textbook parsing first.")
        
        with tab2:
            st.header(f"Curriculum: {selected_discipline}")
            
            curriculum = admin.load_curriculum(selected_discipline)
            
            if curriculum:
                # Curriculum overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Topics", len(curriculum.get("topics", [])))
                
                with col2:
                    st.metric("Subtopics", len(curriculum.get("subtopics", [])))
                
                with col3:
                    prereq_count = sum(len(prereqs) for prereqs in curriculum.get("prerequisites", {}).values())
                    st.metric("Prerequisites", prereq_count)
                
                with col4:
                    graph_info = curriculum.get("graph_info", {})
                    st.metric("Is DAG", "✅" if graph_info.get("is_dag", False) else "❌")
                
                # Topics
                st.subheader("Main Topics")
                topics_df = pd.DataFrame(curriculum.get("topics", []))
                if not topics_df.empty:
                    st.dataframe(topics_df, use_container_width=True)
                
                # Subtopics with search
                st.subheader("Subtopics")
                
                search_term = st.text_input("Search subtopics", "")
                
                subtopics = curriculum.get("subtopics", [])
                if search_term:
                    subtopics = [s for s in subtopics if search_term.lower() in s.get("name", "").lower()]
                
                for subtopic in subtopics[:20]:  # Show first 20
                    with st.expander(f"📖 {subtopic.get('name', 'Unknown')}"):
                        st.write(f"**Description:** {subtopic.get('description', 'No description')}")
                        st.write(f"**Parent Topic:** {subtopic.get('parent_topic', 'Unknown')}")
                        st.write(f"**Duration:** {subtopic.get('duration', 'Unknown')} hours")
                        
                        objectives = subtopic.get('objectives', [])
                        if objectives:
                            st.write("**Learning Objectives:**")
                            for obj in objectives:
                                st.write(f"• {obj}")
                        
                        # Prerequisites
                        prereqs = curriculum.get("prerequisites", {}).get(subtopic.get('name', ''), [])
                        if prereqs:
                            st.write("**Prerequisites:**")
                            for prereq in prereqs:
                                st.write(f"• {prereq}")
                
                if len(subtopics) > 20:
                    st.info(f"Showing first 20 of {len(subtopics)} subtopics. Use search to filter.")
                
                # Learning path
                graph_info = curriculum.get("graph_info", {})
                learning_path = graph_info.get("learning_path", [])
                if learning_path:
                    st.subheader("Recommended Learning Path")
                    
                    with st.expander("View Learning Path"):
                        for i, subtopic in enumerate(learning_path, 1):
                            st.write(f"{i:2d}. {subtopic}")
                
            else:
                st.warning(f"No curriculum found for {selected_discipline}. Run curriculum generation first.")
        
        with tab3:
            st.header(f"Assessment Questions: {selected_discipline}")
            
            questions_data = admin.load_questions(selected_discipline)
            
            if questions_data:
                # Questions overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Questions", questions_data.get("total_questions", 0))
                
                with col2:
                    bloom_levels = len(questions_data.get("questions_by_bloom_level", {}))
                    st.metric("Bloom Levels", bloom_levels)
                
                with col3:
                    question_types = len(questions_data.get("questions_by_type", {}))
                    st.metric("Question Types", question_types)
                
                with col4:
                    subtopics_covered = questions_data.get("metadata", {}).get("subtopics_covered", 0)
                    st.metric("Subtopics Covered", subtopics_covered)
                
                # Question distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bloom level distribution
                    bloom_data = questions_data.get("questions_by_bloom_level", {})
                    if bloom_data:
                        bloom_df = pd.DataFrame([
                            {"Bloom Level": level, "Count": len(questions)}
                            for level, questions in bloom_data.items()
                        ])
                        
                        fig = px.pie(bloom_df, values="Count", names="Bloom Level",
                                   title="Questions by Bloom Level")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Question type distribution
                    type_data = questions_data.get("questions_by_type", {})
                    if type_data:
                        type_df = pd.DataFrame([
                            {"Question Type": qtype, "Count": len(questions)}
                            for qtype, questions in type_data.items()
                        ])
                        
                        fig = px.pie(type_df, values="Count", names="Question Type",
                                   title="Questions by Type")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sample questions
                st.subheader("Sample Questions")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bloom_filter = st.selectbox(
                        "Filter by Bloom Level",
                        ["All"] + list(bloom_data.keys())
                    )
                
                with col2:
                    type_filter = st.selectbox(
                        "Filter by Question Type",
                        ["All"] + list(type_data.keys())
                    )
                
                with col3:
                    topic_filter = st.selectbox(
                        "Filter by Topic",
                        ["All"] + list(questions_data.get("questions_by_topic", {}).keys())
                    )
                
                # Apply filters
                questions = questions_data.get("questions", [])
                
                if bloom_filter != "All":
                    questions = [q for q in questions if q.get("bloom_level") == bloom_filter]
                
                if type_filter != "All":
                    questions = [q for q in questions if q.get("type") == type_filter]
                
                if topic_filter != "All":
                    questions = [q for q in questions if q.get("parent_topic") == topic_filter]
                
                # Display questions
                for i, question in enumerate(questions[:10], 1):
                    with st.expander(f"Question {i}: {question.get('type', 'Unknown')} - {question.get('bloom_level', 'Unknown')}"):
                        st.write(f"**Question:** {question.get('question', 'No question text')}")
                        
                        # Multiple choice options
                        if question.get('options'):
                            st.write("**Options:**")
                            for option in question['options']:
                                st.write(f"• {option}")
                        
                        st.write(f"**Correct Answer:** {question.get('correct_answer', 'No answer provided')}")
                        st.write(f"**Explanation:** {question.get('explanation', 'No explanation provided')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Difficulty:** {question.get('difficulty', 'Unknown')}/5")
                        with col2:
                            st.write(f"**Time:** {question.get('estimated_time_minutes', 'Unknown')} min")
                        with col3:
                            st.write(f"**Subtopic:** {question.get('subtopic', 'Unknown')}")
                
                if len(questions) > 10:
                    st.info(f"Showing first 10 of {len(questions)} questions matching filters.")
            
            else:
                st.warning(f"No questions found for {selected_discipline}. Run question generation first.")
        
        with tab4:
            st.header(f"Analytics: {selected_discipline}")
            
            curriculum = admin.load_curriculum(selected_discipline)
            questions_data = admin.load_questions(selected_discipline)
            chunk_stats = admin.get_chunk_statistics(selected_discipline)
            
            if curriculum and questions_data:
                # Progress tracking
                st.subheader("Pipeline Progress")
                
                progress_data = {
                    "Step": ["Parse Textbooks", "Embed Chunks", "Generate Curriculum", "Generate Questions"],
                    "Status": ["✅", "✅", "✅", "✅"],
                    "Output": [
                        f"{chunk_stats['total_chunks']} chunks",
                        "Embeddings created",
                        f"{len(curriculum.get('subtopics', []))} subtopics",
                        f"{questions_data.get('total_questions', 0)} questions"
                    ]
                }
                
                progress_df = pd.DataFrame(progress_data)
                st.dataframe(progress_df, use_container_width=True)
                
                # Curriculum complexity analysis
                st.subheader("Curriculum Complexity")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Difficulty distribution
                    difficulties = [topic.get("difficulty", 3) for topic in curriculum.get("topics", [])]
                    if difficulties:
                        diff_df = pd.DataFrame({"Difficulty": difficulties})
                        fig = px.histogram(diff_df, x="Difficulty", bins=5,
                                         title="Topic Difficulty Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Prerequisites network
                    prereqs = curriculum.get("prerequisites", {})
                    prereq_counts = [len(p) for p in prereqs.values()]
                    if prereq_counts:
                        prereq_df = pd.DataFrame({"Prerequisite Count": prereq_counts})
                        fig = px.histogram(prereq_df, x="Prerequisite Count",
                                         title="Prerequisite Dependencies")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Question quality metrics
                st.subheader("Question Quality Metrics")
                
                questions = questions_data.get("questions", [])
                if questions:
                    difficulties = [q.get("difficulty", 3) for q in questions]
                    times = [q.get("estimated_time_minutes", 10) for q in questions]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        avg_difficulty = sum(difficulties) / len(difficulties)
                        st.metric("Average Difficulty", f"{avg_difficulty:.1f}/5")
                        
                        fig = px.histogram(pd.DataFrame({"Difficulty": difficulties}),
                                         x="Difficulty", title="Question Difficulty Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        avg_time = sum(times) / len(times)
                        st.metric("Average Time", f"{avg_time:.1f} min")
                        
                        fig = px.histogram(pd.DataFrame({"Time": times}),
                                         x="Time", title="Estimated Time Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("Complete curriculum and question generation to view analytics.")
    
    else:
        st.info("Select a discipline from the sidebar to get started.")

if __name__ == '__main__':
    main()