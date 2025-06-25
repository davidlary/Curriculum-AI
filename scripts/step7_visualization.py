#!/usr/bin/env python3
"""
Step 7: Visualization Module
Creates comprehensive visualizations of the curriculum structure and learning pathways.

This module generates:
1. Curriculum dependency graphs
2. Learning pathway flowcharts
3. Domain relationship maps
4. Educational progression visualizations
5. Adaptive branching diagrams
6. Quality metrics dashboards
7. Interactive curriculum explorer
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization libraries not available: {e}")
    print("Install with: pip install matplotlib seaborn plotly")
    VISUALIZATION_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory structure
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Curriculum"
VISUALIZATION_DIR = BASE_DIR / "Visualizations"

# Create visualization directory
VISUALIZATION_DIR.mkdir(exist_ok=True)

@dataclass
class VisualizationReport:
    """Report of generated visualizations."""
    discipline: str
    language: str
    generated_files: List[str]
    visualization_types: List[str]
    quality_metrics: Dict[str, float]
    generation_timestamp: str

class CurriculumVisualizer:
    """
    Creates comprehensive visualizations of curriculum structure,
    learning pathways, and adaptive features.
    """
    
    def __init__(self):
        self.color_schemes = {
            'educational_levels': {
                'high_school': '#FF6B6B',
                'undergraduate': '#4ECDC4', 
                'graduate': '#45B7D1'
            },
            'domains': {
                'kinematics': '#FF6B6B',
                'dynamics': '#4ECDC4',
                'energy': '#45B7D1',
                'electricity': '#96CEB4',
                'magnetism': '#FECA57',
                'waves': '#FF9FF3',
                'optics': '#54A0FF',
                'thermodynamics': '#5F27CD',
                'modern_physics': '#00D2D3',
                'general': '#C4C4C4'
            },
            'difficulty': {
                1: '#E8F5E8',
                2: '#C3E9C0',
                3: '#9CDD97',
                4: '#74D16F',
                5: '#4CC546'
            }
        }
        
        logger.info("CurriculumVisualizer initialized")

    def create_comprehensive_visualizations(self, discipline: str, language: str = "English") -> Dict[str, Any]:
        """
        Create comprehensive visualizations of the curriculum system.
        """
        if not VISUALIZATION_AVAILABLE:
            logger.error("Visualization libraries not available. Install matplotlib, seaborn, and plotly.")
            return {"error": "Visualization libraries not available"}
        
        start_time = time.time()
        logger.info(f"Creating comprehensive visualizations for {discipline} in {language}")
        
        # Load all curriculum data
        curriculum_data = self._load_all_curriculum_data(discipline, language)
        
        generated_files = []
        visualization_types = []
        
        # 1. Curriculum Dependency Graph
        if 'sequenced' in curriculum_data:
            dep_graph_file = self._create_dependency_graph(
                curriculum_data['sequenced'], discipline, language
            )
            if dep_graph_file:
                generated_files.append(dep_graph_file)
                visualization_types.append("dependency_graph")
        
        # 2. Learning Pathways Flowchart
        if 'adaptive' in curriculum_data:
            pathways_file = self._create_pathways_flowchart(
                curriculum_data['adaptive'], discipline, language
            )
            if pathways_file:
                generated_files.append(pathways_file)
                visualization_types.append("learning_pathways")
        
        # 3. Domain Relationship Map
        if 'comprehensive' in curriculum_data:
            domain_map_file = self._create_domain_relationship_map(
                curriculum_data['comprehensive'], discipline, language
            )
            if domain_map_file:
                generated_files.append(domain_map_file)
                visualization_types.append("domain_relationships")
        
        # 4. Educational Progression Visualization
        progression_file = self._create_educational_progression_viz(
            curriculum_data, discipline, language
        )
        if progression_file:
            generated_files.append(progression_file)
            visualization_types.append("educational_progression")
        
        # 5. Quality Metrics Dashboard
        metrics_file = self._create_quality_metrics_dashboard(
            curriculum_data, discipline, language
        )
        if metrics_file:
            generated_files.append(metrics_file)
            visualization_types.append("quality_metrics")
        
        # 6. Interactive Curriculum Explorer
        explorer_file = self._create_interactive_explorer(
            curriculum_data, discipline, language
        )
        if explorer_file:
            generated_files.append(explorer_file)
            visualization_types.append("interactive_explorer")
        
        # 7. Adaptive Branching Diagram
        if 'adaptive' in curriculum_data:
            branching_file = self._create_adaptive_branching_diagram(
                curriculum_data['adaptive'], discipline, language
            )
            if branching_file:
                generated_files.append(branching_file)
                visualization_types.append("adaptive_branching")
        
        processing_time = time.time() - start_time
        
        # Calculate visualization quality metrics
        quality_metrics = self._calculate_visualization_metrics(
            curriculum_data, generated_files
        )
        
        # Create visualization report
        report = VisualizationReport(
            discipline=discipline,
            language=language,
            generated_files=generated_files,
            visualization_types=visualization_types,
            quality_metrics=quality_metrics,
            generation_timestamp=datetime.now().isoformat()
        )
        
        result = {
            'discipline': discipline,
            'language': language,
            'visualization_timestamp': datetime.now().isoformat(),
            'generated_files': generated_files,
            'visualization_types': visualization_types,
            'quality_metrics': quality_metrics,
            'metrics': {
                'total_visualizations': len(generated_files),
                'processing_time': processing_time,
                'data_completeness': self._assess_data_completeness(curriculum_data),
                'visualization_coverage': len(visualization_types) / 7  # 7 planned visualization types
            }
        }
        
        # Save visualization report
        report_file = VISUALIZATION_DIR / f"{discipline}_{language}_visualization_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Visualization generation completed: {len(generated_files)} files created")
        return result

    def _load_all_curriculum_data(self, discipline: str, language: str) -> Dict[str, Any]:
        """Load all available curriculum data files."""
        data = {}
        
        # File mappings
        file_mappings = {
            'sequenced': f"{discipline}_{language}_curriculum_sequenced.json",
            'adaptive': f"{discipline}_{language}_adaptive_curriculum.json",
            'comprehensive': f"{discipline}_{language}_comprehensive_curriculum.json",
            'enhanced_comprehensive': f"{discipline}_enhanced_comprehensive_curriculum.json",
            'toc_aware': f"{discipline}_toc_aware_curriculum.json",
            'prerequisites': f"{discipline}_{language}_prerequisites_mapped.json"
        }
        
        for data_type, filename in file_mappings.items():
            file_path = OUTPUT_DIR / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data[data_type] = json.load(f)
                    logger.info(f"Loaded {data_type} data from {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        return data

    def _create_dependency_graph(self, sequenced_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create curriculum dependency graph visualization."""
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            curriculum_units = sequenced_data.get('curriculum_units', [])
            
            # Add nodes
            for unit in curriculum_units:
                unit_id = unit['unit_id']
                G.add_node(unit_id, 
                          title=unit.get('title', ''),
                          level=unit.get('educational_level', 'undergraduate'),
                          domain=unit.get('domain', 'general'))
            
            # Add edges for prerequisites
            for unit in curriculum_units:
                unit_id = unit['unit_id']
                for prereq in unit.get('prerequisites', []):
                    if prereq in G.nodes:
                        G.add_edge(prereq, unit_id)
            
            # Create visualization
            plt.figure(figsize=(20, 16))
            
            # Use hierarchical layout
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # Fallback to spring layout
                pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Color nodes by educational level
            node_colors = []
            for node in G.nodes():
                level = G.nodes[node].get('level', 'undergraduate')
                node_colors.append(self.color_schemes['educational_levels'].get(level, '#C4C4C4'))
            
            # Draw graph
            nx.draw(G, pos, 
                   node_color=node_colors,
                   node_size=300,
                   with_labels=False,
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   alpha=0.7)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color=color, label=level.replace('_', ' ').title())
                for level, color in self.color_schemes['educational_levels'].items()
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title(f"{discipline} Curriculum Dependency Graph", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_dependency_graph.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created dependency graph: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create dependency graph: {e}")
            return None

    def _create_pathways_flowchart(self, adaptive_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create learning pathways flowchart."""
        try:
            pathways = adaptive_data.get('learning_pathways', [])
            
            if not pathways:
                logger.warning("No learning pathways found for flowchart")
                return None
            
            # Create interactive plotly figure
            fig = make_subplots(
                rows=len(pathways), cols=1,
                subplot_titles=[pathway['name'] for pathway in pathways],
                vertical_spacing=0.1
            )
            
            for i, pathway in enumerate(pathways, 1):
                topics = pathway['topic_sequence'][:20]  # Limit to first 20 for visibility
                
                # Create pathway flow
                x_coords = list(range(len(topics)))
                y_coords = [i] * len(topics)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers+lines',
                        name=pathway['name'],
                        text=[f"Topic {j+1}" for j in range(len(topics))],
                        hovertemplate="<b>%{text}</b><br>Position: %{x}<extra></extra>",
                        marker=dict(size=10, color=f"rgba({50 + i*30}, {100 + i*20}, {150 + i*10}, 0.8)")
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                title=f"{discipline} Learning Pathways Overview",
                height=300 * len(pathways),
                showlegend=True
            )
            
            # Save as HTML
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_pathways_flowchart.html"
            fig.write_html(output_file)
            
            logger.info(f"Created pathways flowchart: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create pathways flowchart: {e}")
            return None

    def _create_domain_relationship_map(self, comprehensive_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create domain relationship map."""
        try:
            topics = comprehensive_data.get('comprehensive_topics', [])
            
            if not topics:
                logger.warning("No comprehensive topics found for domain map")
                return None
            
            # Count topics by domain
            domain_counts = defaultdict(int)
            domain_connections = defaultdict(set)
            
            for topic in topics:
                domain = topic.get('domain', 'general')
                domain_counts[domain] += 1
                
                # Track connections through prerequisites
                for prereq in topic.get('prerequisites', []):
                    # Find prerequisite topic's domain
                    prereq_topic = next((t for t in topics if t.get('topic_id') == prereq), None)
                    if prereq_topic:
                        prereq_domain = prereq_topic.get('domain', 'general')
                        if prereq_domain != domain:
                            domain_connections[domain].add(prereq_domain)
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (domains)
            for domain, count in domain_counts.items():
                G.add_node(domain, weight=count)
            
            # Add edges (domain connections)
            for domain, connected_domains in domain_connections.items():
                for connected_domain in connected_domains:
                    if G.has_node(connected_domain):
                        G.add_edge(domain, connected_domain)
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            
            # Position nodes
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Node sizes based on topic count
            node_sizes = [domain_counts[node] * 50 for node in G.nodes()]
            
            # Node colors from color scheme
            node_colors = [
                self.color_schemes['domains'].get(node, '#C4C4C4') 
                for node in G.nodes()
            ]
            
            # Draw network
            nx.draw(G, pos,
                   node_color=node_colors,
                   node_size=node_sizes,
                   with_labels=True,
                   font_size=10,
                   font_weight='bold',
                   edge_color='gray',
                   alpha=0.8)
            
            # Add node labels with counts
            labels = {node: f"{node}\n({domain_counts[node]} topics)" for node in G.nodes()}
            pos_labels = {k: (v[0], v[1] - 0.1) for k, v in pos.items()}
            nx.draw_networkx_labels(G, pos_labels, labels, font_size=8)
            
            plt.title(f"{discipline} Domain Relationship Map", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_domain_map.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created domain relationship map: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create domain relationship map: {e}")
            return None

    def _create_educational_progression_viz(self, curriculum_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create educational progression visualization."""
        try:
            # Use sequenced data if available, otherwise comprehensive
            data_source = curriculum_data.get('sequenced') or curriculum_data.get('comprehensive')
            
            if not data_source:
                logger.warning("No suitable data found for educational progression visualization")
                return None
            
            units = data_source.get('curriculum_units', []) or data_source.get('comprehensive_topics', [])
            
            # Group by educational level
            level_data = defaultdict(list)
            for unit in units:
                level = unit.get('educational_level') or unit.get('primary_level', 'undergraduate')
                level_data[level].append(unit)
            
            # Create stacked bar chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Chart 1: Topic count by level
            levels = list(level_data.keys())
            counts = [len(level_data[level]) for level in levels]
            colors = [self.color_schemes['educational_levels'].get(level, '#C4C4C4') for level in levels]
            
            ax1.bar(levels, counts, color=colors)
            ax1.set_title('Topics by Educational Level')
            ax1.set_ylabel('Number of Topics')
            
            # Chart 2: Domain distribution within levels
            domains_by_level = defaultdict(lambda: defaultdict(int))
            for level, level_units in level_data.items():
                for unit in level_units:
                    domain = unit.get('domain', 'general')
                    domains_by_level[level][domain] += 1
            
            # Create stacked bars for domains
            all_domains = set()
            for level_domains in domains_by_level.values():
                all_domains.update(level_domains.keys())
            all_domains = sorted(all_domains)
            
            bottom = [0] * len(levels)
            for domain in all_domains:
                values = [domains_by_level[level][domain] for level in levels]
                ax2.bar(levels, values, bottom=bottom, 
                       label=domain.replace('_', ' ').title(),
                       color=self.color_schemes['domains'].get(domain, '#C4C4C4'))
                bottom = [b + v for b, v in zip(bottom, values)]
            
            ax2.set_title('Domain Distribution by Educational Level')
            ax2.set_ylabel('Number of Topics')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save figure
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_educational_progression.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created educational progression visualization: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create educational progression visualization: {e}")
            return None

    def _create_quality_metrics_dashboard(self, curriculum_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create quality metrics dashboard."""
        try:
            # Collect quality metrics from all data sources
            all_metrics = {}
            
            for data_type, data in curriculum_data.items():
                if 'quality_metrics' in data:
                    for metric, value in data['quality_metrics'].items():
                        all_metrics[f"{data_type}_{metric}"] = value
            
            if not all_metrics:
                logger.warning("No quality metrics found for dashboard")
                return None
            
            # Create dashboard with plotly
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Quality Scores', 'Coverage Metrics', 'Progression Metrics', 'Overall Summary'),
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Quality scores (subset)
            quality_metrics = {k: v for k, v in all_metrics.items() 
                             if 'quality' in k.lower() or 'score' in k.lower()}
            
            if quality_metrics:
                fig.add_trace(
                    go.Bar(x=list(quality_metrics.keys()), 
                          y=list(quality_metrics.values()),
                          name="Quality Scores",
                          marker_color='lightblue'),
                    row=1, col=1
                )
            
            # Coverage metrics
            coverage_metrics = {k: v for k, v in all_metrics.items() 
                              if 'coverage' in k.lower()}
            
            if coverage_metrics:
                fig.add_trace(
                    go.Scatter(x=list(coverage_metrics.keys()),
                              y=list(coverage_metrics.values()),
                              mode='markers+lines',
                              name="Coverage Metrics",
                              marker_color='green'),
                    row=1, col=2
                )
            
            # Progression metrics
            progression_metrics = {k: v for k, v in all_metrics.items() 
                                 if 'progression' in k.lower() or 'order' in k.lower()}
            
            if progression_metrics:
                fig.add_trace(
                    go.Bar(x=list(progression_metrics.keys()),
                          y=list(progression_metrics.values()),
                          name="Progression Metrics",
                          marker_color='orange'),
                    row=2, col=1
                )
            
            # Overall summary (gauge)
            overall_quality = sum(all_metrics.values()) / len(all_metrics) if all_metrics else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_quality,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Quality"},
                    gauge={'axis': {'range': [None, 1]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.5, 0.8], 'color': "yellow"},
                                   {'range': [0.8, 1], 'color': "green"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75,
                                       'value': 0.9}}),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f"{discipline} Curriculum Quality Metrics Dashboard",
                height=800,
                showlegend=False
            )
            
            # Save as HTML
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_quality_dashboard.html"
            fig.write_html(output_file)
            
            logger.info(f"Created quality metrics dashboard: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create quality metrics dashboard: {e}")
            return None

    def _create_interactive_explorer(self, curriculum_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create interactive curriculum explorer."""
        try:
            # Use the most comprehensive data available
            data_source = (curriculum_data.get('enhanced_comprehensive') or 
                          curriculum_data.get('comprehensive') or 
                          curriculum_data.get('sequenced'))
            
            if not data_source:
                logger.warning("No suitable data found for interactive explorer")
                return None
            
            topics = (data_source.get('comprehensive_topics', []) or 
                     data_source.get('curriculum_units', []))
            
            if not topics:
                logger.warning("No topics found for interactive explorer")
                return None
            
            # Create interactive scatter plot
            x_coords = []
            y_coords = []
            texts = []
            colors = []
            sizes = []
            
            for i, topic in enumerate(topics):
                x_coords.append(i % 20)  # Arrange in grid
                y_coords.append(i // 20)
                
                title = topic.get('title', topic.get('normalized_title', f"Topic {i+1}"))
                texts.append(title)
                
                domain = topic.get('domain', 'general')
                colors.append(self.color_schemes['domains'].get(domain, '#C4C4C4'))
                
                # Size based on estimated duration or importance
                duration = topic.get('estimated_duration_hours', 1)
                sizes.append(max(10, min(30, duration * 5)))
            
            fig = go.Figure(data=go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.8,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=texts,
                hovertemplate="<b>%{text}</b><br>Position: (%{x}, %{y})<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"{discipline} Interactive Curriculum Explorer",
                xaxis_title="Topic Position (X)",
                yaxis_title="Topic Position (Y)",
                height=600,
                hovermode='closest'
            )
            
            # Save as HTML
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_interactive_explorer.html"
            fig.write_html(output_file)
            
            logger.info(f"Created interactive curriculum explorer: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create interactive explorer: {e}")
            return None

    def _create_adaptive_branching_diagram(self, adaptive_data: Dict, discipline: str, language: str) -> Optional[str]:
        """Create adaptive branching diagram."""
        try:
            pathways = adaptive_data.get('learning_pathways', [])
            
            if not pathways:
                logger.warning("No pathways found for adaptive branching diagram")
                return None
            
            # Create a simplified branching diagram for the first pathway
            main_pathway = pathways[0]
            topics = main_pathway['topic_sequence'][:15]  # Limit for clarity
            branching = main_pathway.get('adaptive_branching', {})
            
            # Create network graph
            G = nx.DiGraph()
            
            # Add main pathway nodes
            for i, topic in enumerate(topics):
                G.add_node(topic, type='main', position=i)
            
            # Add main pathway edges
            for i in range(len(topics) - 1):
                G.add_edge(topics[i], topics[i + 1], type='main')
            
            # Add branching options
            for topic, branches in branching.items():
                if topic in topics:
                    for branch in branches[:2]:  # Limit branches for clarity
                        if branch not in G.nodes:
                            G.add_node(branch, type='branch')
                        G.add_edge(topic, branch, type='branch')
            
            # Create visualization
            plt.figure(figsize=(16, 10))
            
            # Position nodes
            pos = {}
            
            # Main pathway - horizontal line
            for i, topic in enumerate(topics):
                pos[topic] = (i * 2, 0)
            
            # Branch nodes - above and below main line
            branch_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'branch']
            branch_y_offset = 1
            
            for i, branch in enumerate(branch_nodes):
                # Find parent node in main pathway
                parents = [n for n in G.predecessors(branch) if n in topics]
                if parents:
                    parent_x = pos[parents[0]][0]
                    y_offset = branch_y_offset if i % 2 == 0 else -branch_y_offset
                    pos[branch] = (parent_x + 0.5, y_offset)
            
            # Draw main pathway
            main_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'main']
            nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, 
                                  node_color='lightblue', node_size=500)
            
            # Draw branch nodes
            nx.draw_networkx_nodes(G, pos, nodelist=branch_nodes,
                                  node_color='lightcoral', node_size=300)
            
            # Draw main pathway edges
            main_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'main']
            nx.draw_networkx_edges(G, pos, edgelist=main_edges,
                                  edge_color='blue', width=2, arrows=True)
            
            # Draw branch edges
            branch_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'branch']
            nx.draw_networkx_edges(G, pos, edgelist=branch_edges,
                                  edge_color='red', width=1, style='dashed', arrows=True)
            
            # Add labels
            labels = {node: f"T{i+1}" for i, node in enumerate(topics)}
            for i, branch in enumerate(branch_nodes):
                labels[branch] = f"B{i+1}"
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title(f"{discipline} Adaptive Branching Diagram\nMain Pathway: {main_pathway['name']}", 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            output_file = VISUALIZATION_DIR / f"{discipline}_{language}_adaptive_branching.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created adaptive branching diagram: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create adaptive branching diagram: {e}")
            return None

    def _calculate_visualization_metrics(self, curriculum_data: Dict, generated_files: List[str]) -> Dict[str, float]:
        """Calculate quality metrics for visualizations."""
        
        # Data completeness
        data_sources = len(curriculum_data)
        max_data_sources = 6  # Expected number of data sources
        data_completeness = min(data_sources / max_data_sources, 1.0)
        
        # Visualization completeness
        visualization_completeness = len(generated_files) / 7  # 7 planned visualizations
        
        # File generation success rate
        file_success_rate = 1.0 if generated_files else 0.0
        
        return {
            'data_completeness': data_completeness,
            'visualization_completeness': visualization_completeness,
            'file_generation_success': file_success_rate,
            'overall_visualization_quality': (
                data_completeness * 0.4 +
                visualization_completeness * 0.4 +
                file_success_rate * 0.2
            )
        }

    def _assess_data_completeness(self, curriculum_data: Dict) -> float:
        """Assess completeness of loaded curriculum data."""
        expected_sources = [
            'sequenced', 'adaptive', 'comprehensive', 
            'enhanced_comprehensive', 'toc_aware', 'prerequisites'
        ]
        
        available_sources = len(curriculum_data)
        return min(available_sources / len(expected_sources), 1.0)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create comprehensive curriculum visualizations")
    parser.add_argument("--discipline", required=True, help="Target discipline")
    parser.add_argument("--language", default="English", help="Target language")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization libraries not available. Install with: pip install matplotlib seaborn plotly")
        sys.exit(1)
    
    try:
        visualizer = CurriculumVisualizer()
        result = visualizer.create_comprehensive_visualizations(
            discipline=args.discipline,
            language=args.language
        )
        
        if 'error' in result:
            logger.error(f"Visualization failed: {result['error']}")
            sys.exit(1)
        
        # Print summary
        print(f"\nVisualization Summary for {args.discipline} ({args.language}):")
        print(f"Generated visualizations: {result['metrics']['total_visualizations']}")
        print(f"Visualization coverage: {result['metrics']['visualization_coverage']:.1%}")
        print(f"Data completeness: {result['metrics']['data_completeness']:.1%}")
        print(f"Processing time: {result['metrics']['processing_time']:.2f}s")
        
        print(f"\nGenerated Files:")
        for viz_file in result['generated_files']:
            print(f"  • {Path(viz_file).name}")
        
        print(f"\nVisualization Types:")
        for viz_type in result['visualization_types']:
            print(f"  • {viz_type.replace('_', ' ').title()}")
        
        print(f"\nQuality Metrics:")
        for metric, value in result['quality_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\n✅ Visualizations saved to: {VISUALIZATION_DIR}")
        
    except Exception as e:
        logger.error(f"Error during visualization creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()