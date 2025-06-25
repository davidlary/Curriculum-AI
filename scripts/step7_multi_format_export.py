#!/usr/bin/env python3
"""
Step 7: Multi-Format Export Engine

This module exports the complete curriculum in multiple formats for different use cases:
- TSV: Tab-separated format with full metadata for spreadsheet analysis
- JSON: Hierarchical JSON for APIs, LLM training, and tree viewers
- DOT: Graph structure for Graphviz prerequisite visualization
- DuckDB: Structured database for fast querying and dashboard integration

Features:
- Complete data export with all metadata
- Optimized formats for different use cases
- Data validation and integrity checks
- Performance monitoring
- Export statistics and summaries
"""

import sys
import json
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict
import argparse
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.curriculum_utils import (
    CurriculumConfig, CurriculumLogger, FileManager, 
    DataValidator, load_config
)


class MultiFormatExporter:
    """Main engine for exporting curriculum data in multiple formats."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger):
        self.config = config
        self.logger = logger
        self.file_manager = FileManager(logger)
        
        # Export statistics
        self.export_stats = {
            "tsv": {"records": 0, "size_mb": 0},
            "json": {"size_mb": 0, "hierarchy_depth": 0},
            "dot": {"nodes": 0, "edges": 0, "size_mb": 0},
            "duckdb": {"tables": 0, "records": 0, "size_mb": 0}
        }
    
    def export_to_tsv(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export curriculum to tab-separated values format."""
        self.logger.start_timer("tsv_export")
        
        try:
            tsv_path = Path(output_path)
            
            # Flatten hierarchy to tabular format
            records = self._flatten_hierarchy_to_records(data)
            
            if not records:
                self.logger.warning("No records to export to TSV")
                return False
            
            # Write TSV file
            with open(tsv_path, 'w', newline='', encoding='utf-8') as tsvfile:
                fieldnames = records[0].keys()
                writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
                
                writer.writeheader()
                writer.writerows(records)
            
            # Calculate statistics
            file_size = tsv_path.stat().st_size / (1024 * 1024)  # MB
            self.export_stats["tsv"] = {
                "records": len(records),
                "size_mb": round(file_size, 2)
            }
            
            self.logger.info(f"TSV export: {len(records)} records, {file_size:.2f} MB")
            self.logger.end_timer("tsv_export")
            return True
            
        except Exception as e:
            self.logger.error(f"TSV export failed: {e}")
            self.logger.end_timer("tsv_export")
            return False
    
    def _flatten_hierarchy_to_records(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten hierarchical curriculum data into tabular records."""
        records = []
        
        # Extract hierarchy data
        hierarchy = data.get("hierarchy", {})
        topics = data.get("topics", {})
        prerequisites = data.get("prerequisites", [])
        standards_mappings = data.get("standards_mappings", {})
        
        # Build lookup tables
        prereq_lookup = defaultdict(list)
        for prereq in prerequisites:
            prereq_lookup[prereq["target"]].append(prereq)
        
        standards_lookup = defaultdict(list)
        for standard_type, mappings in standards_mappings.items():
            for mapping in mappings:
                topic_id = mapping.get("topic_id", "")
                standards_lookup[topic_id].append(mapping)
        
        # Process each topic in the hierarchy
        def process_hierarchy_level(obj, path=[], level=1, classification=""):
            if level > 6:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = path + [key]
                    topic_id = " → ".join(current_path)
                    
                    # Create record
                    record = {
                        "topic_id": topic_id,
                        "title": key,
                        "classification": classification,
                        "level": level,
                        "level_name": self._get_level_name(level),
                        "domain": current_path[1] if len(current_path) > 1 else current_path[0] if current_path else "",
                        "category": current_path[2] if len(current_path) > 2 else "",
                        "concept": current_path[3] if len(current_path) > 3 else "",
                        "topic": current_path[4] if len(current_path) > 4 else "",
                        "subtopic": current_path[5] if len(current_path) > 5 else "",
                        "learning_element": current_path[6] if len(current_path) > 6 else "",
                        "full_path": " → ".join(current_path),
                        "parent_path": " → ".join(current_path[:-1]) if len(current_path) > 1 else "",
                    }
                    
                    # Add topic metadata if available
                    topic_info = topics.get(topic_id, {})
                    record.update({
                        "topic_level": topic_info.get("level", level),
                        "topic_domain": topic_info.get("domain", ""),
                    })
                    
                    # Add prerequisite information
                    topic_prereqs = prereq_lookup.get(topic_id, [])
                    record.update({
                        "prerequisites_count": len(topic_prereqs),
                        "prerequisites": "; ".join([p["source"] for p in topic_prereqs[:3]]),  # First 3
                        "prerequisite_types": "; ".join(list(set([p["type"] for p in topic_prereqs]))),
                        "avg_prerequisite_confidence": sum([p["confidence"] for p in topic_prereqs]) / len(topic_prereqs) if topic_prereqs else 0
                    })
                    
                    # Add standards mappings
                    topic_standards = standards_lookup.get(topic_id, [])
                    record.update({
                        "standards_count": len(topic_standards),
                        "mapped_standards": "; ".join(list(set([s["standard_type"] for s in topic_standards]))),
                        "mcat_mapped": any(s["standard_type"] == "MCAT" for s in topic_standards),
                        "ib_mapped": any(s["standard_type"].startswith("IB_") for s in topic_standards),
                        "a_level_mapped": any(s["standard_type"] == "A_Level" for s in topic_standards),
                        "igcse_mapped": any(s["standard_type"] == "IGCSE" for s in topic_standards),
                        "avg_standards_confidence": sum([s["confidence"] for s in topic_standards]) / len(topic_standards) if topic_standards else 0,
                        "bloom_levels": "; ".join(list(set([s["bloom_level"] for s in topic_standards]))),
                        "difficulty_levels": "; ".join(list(set([s["difficulty_level"] for s in topic_standards]))),
                        "application_domains": "; ".join(list(set([d for s in topic_standards for d in s.get("application_domains", [])])))
                    })
                    
                    records.append(record)
                    
                    # Recurse
                    process_hierarchy_level(value, current_path, level + 1, classification)
            
            elif isinstance(obj, list):
                # Level 6 learning elements
                for i, element in enumerate(obj):
                    if isinstance(element, str):
                        element_path = path + [element]
                        topic_id = " → ".join(element_path)
                        
                        record = {
                            "topic_id": topic_id,
                            "title": element,
                            "classification": classification,
                            "level": level,
                            "level_name": self._get_level_name(level),
                            "domain": element_path[1] if len(element_path) > 1 else "",
                            "category": element_path[2] if len(element_path) > 2 else "",
                            "concept": element_path[3] if len(element_path) > 3 else "",
                            "topic": element_path[4] if len(element_path) > 4 else "",
                            "subtopic": element_path[5] if len(element_path) > 5 else "",
                            "learning_element": element,
                            "full_path": " → ".join(element_path),
                            "parent_path": " → ".join(element_path[:-1]),
                            "prerequisites_count": 0,
                            "standards_count": 0
                        }
                        
                        records.append(record)
        
        # Process core and electives
        for classification, content in hierarchy.items():
            if isinstance(content, dict):
                process_hierarchy_level(content, [classification], 1, classification)
        
        return records
    
    def _get_level_name(self, level: int) -> str:
        """Get the name for a hierarchy level."""
        level_names = [
            "Classification", "Domain", "Category", "Concept", "Topic", "Subtopic", "Learning Element"
        ]
        return level_names[level] if level < len(level_names) else f"Level {level}"
    
    def export_to_json(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export curriculum to hierarchical JSON format."""
        self.logger.start_timer("json_export")
        
        try:
            json_path = Path(output_path)
            
            # Create enhanced JSON with additional metadata
            enhanced_data = self._enhance_json_structure(data)
            
            # Write JSON file with proper formatting
            with open(json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(enhanced_data, jsonfile, indent=2, ensure_ascii=False)
            
            # Calculate statistics
            file_size = json_path.stat().st_size / (1024 * 1024)  # MB
            hierarchy_depth = self._calculate_hierarchy_depth(enhanced_data.get("hierarchy", {}))
            
            self.export_stats["json"] = {
                "size_mb": round(file_size, 2),
                "hierarchy_depth": hierarchy_depth
            }
            
            self.logger.info(f"JSON export: {file_size:.2f} MB, depth {hierarchy_depth}")
            self.logger.end_timer("json_export")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            self.logger.end_timer("json_export")
            return False
    
    def _enhance_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance JSON structure with additional metadata and navigation aids."""
        enhanced = data.copy()
        
        # Add export metadata
        enhanced["export_metadata"] = {
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "format": "hierarchical_json",
            "generated_by": "Curriculum AI Pipeline Step 7"
        }
        
        # Add navigation metadata to hierarchy
        if "hierarchy" in enhanced:
            enhanced["hierarchy"] = self._add_navigation_metadata(enhanced["hierarchy"])
        
        return enhanced
    
    def _add_navigation_metadata(self, obj: Any, path: List[str] = [], level: int = 1) -> Any:
        """Add navigation metadata to hierarchy structure."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = path + [key]
                
                if level < 6:  # Not at learning elements level
                    result[key] = {
                        "_metadata": {
                            "level": level,
                            "path": current_path,
                            "id": " → ".join(current_path),
                            "parent_id": " → ".join(current_path[:-1]) if len(current_path) > 1 else None,
                            "children_count": len(value) if isinstance(value, dict) else 0
                        }
                    }
                    
                    if isinstance(value, dict):
                        result[key].update(self._add_navigation_metadata(value, current_path, level + 1))
                    else:
                        result[key]["content"] = value
                else:
                    result[key] = value
            
            return result
        
        elif isinstance(obj, list):
            # Learning elements - add metadata
            return {
                "_metadata": {
                    "level": level,
                    "type": "learning_elements",
                    "count": len(obj)
                },
                "elements": obj
            }
        
        else:
            return obj
    
    def _calculate_hierarchy_depth(self, hierarchy: Dict) -> int:
        """Calculate the maximum depth of the hierarchy."""
        def max_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(max_depth(value, current_depth + 1) for value in obj.values())
            elif isinstance(obj, list):
                return current_depth + 1
            else:
                return current_depth
        
        return max_depth(hierarchy)
    
    def export_to_dot(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export prerequisite graph to DOT format for Graphviz."""
        self.logger.start_timer("dot_export")
        
        try:
            dot_path = Path(output_path)
            
            # Extract graph data
            graph_data = data.get("graph_data", {})
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # If no graph data available, create from hierarchy
            if not nodes and not edges:
                self.logger.info("No graph data found, creating DOT from hierarchy structure")
                nodes, edges = self._create_graph_from_hierarchy(data)
                
            if not nodes and not edges:
                self.logger.warning("No graph data or hierarchy available for DOT export")
                # Create minimal placeholder DOT
                nodes = [{"id": "curriculum", "label": "Physics Curriculum", "type": "domain"}]
                edges = []
            
            # Generate DOT content
            dot_content = self._generate_dot_content(nodes, edges, data.get("metadata", {}))
            
            # Write DOT file
            with open(dot_path, 'w', encoding='utf-8') as dotfile:
                dotfile.write(dot_content)
            
            # Calculate statistics
            file_size = dot_path.stat().st_size / (1024 * 1024)  # MB
            self.export_stats["dot"] = {
                "nodes": len(nodes),
                "edges": len(edges),
                "size_mb": round(file_size, 2)
            }
            
            self.logger.info(f"DOT export: {len(nodes)} nodes, {len(edges)} edges, {file_size:.2f} MB")
            self.logger.end_timer("dot_export")
            return True
            
        except Exception as e:
            self.logger.error(f"DOT export failed: {e}")
            self.logger.end_timer("dot_export")
            return False
    
    def _generate_dot_content(self, nodes: List[Dict], edges: List[Dict], metadata: Dict) -> str:
        """Generate DOT graph content."""
        lines = []
        
        # Graph header
        lines.append("digraph curriculum_prerequisites {")
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")
        lines.append("    edge [arrowhead=normal];")
        lines.append("")
        
        # Graph metadata
        title = f"Curriculum Prerequisites Graph"
        if metadata:
            if "standards_mapping" in metadata:
                title += f" ({metadata['standards_mapping']['processed_topics']} topics)"
        
        lines.append(f'    label="{title}";')
        lines.append("    labelloc=t;")
        lines.append("")
        
        # Domain-based clustering and coloring
        domain_colors = {
            "Mechanics": "#FF6B6B",
            "Thermodynamics": "#4ECDC4", 
            "Electricity": "#45B7D1",
            "Waves": "#96CEB4",
            "Modern Physics": "#FFEAA7",
            "Mathematical Prerequisites": "#DDA0DD",
            "General Physics": "#98D8C8"
        }
        
        # Group nodes by domain
        nodes_by_domain = defaultdict(list)
        for node in nodes:
            domain = node.get("domain", "General Physics")
            # Simplify domain name
            domain_key = self._simplify_domain_name(domain)
            nodes_by_domain[domain_key].append(node)
        
        # Add subgraphs for domains
        for domain, domain_nodes in nodes_by_domain.items():
            if len(domain_nodes) > 1:  # Only cluster if multiple nodes
                color = domain_colors.get(domain, "#F0F0F0")
                lines.append(f'    subgraph cluster_{domain.replace(" ", "_")} {{')
                lines.append(f'        label="{domain}";')
                lines.append(f'        color="{color}";')
                lines.append(f'        fillcolor="{color}20";')
                lines.append("        style=filled;")
                
                for node in domain_nodes:
                    node_id = self._sanitize_node_id(node["id"])
                    node_title = self._truncate_title(node.get("title", ""))
                    lines.append(f'        "{node_id}" [label="{node_title}", fillcolor="{color}"];')
                
                lines.append("    }")
                lines.append("")
            else:
                # Single node
                node = domain_nodes[0]
                node_id = self._sanitize_node_id(node["id"])
                node_title = self._truncate_title(node.get("title", ""))
                color = domain_colors.get(domain, "#F0F0F0")
                lines.append(f'    "{node_id}" [label="{node_title}", fillcolor="{color}"];')
        
        lines.append("")
        
        # Add edges
        for edge in edges:
            source_id = self._sanitize_node_id(edge["source"])
            target_id = self._sanitize_node_id(edge["target"])
            weight = edge.get("weight", 0.5)
            edge_type = edge.get("type", "")
            
            # Style edge based on type and confidence
            edge_attrs = []
            
            if weight > 0.8:
                edge_attrs.append("style=bold")
            elif weight < 0.5:
                edge_attrs.append("style=dashed")
            
            if edge_type == "hierarchical":
                edge_attrs.append("color=blue")
            elif edge_type == "domain":
                edge_attrs.append("color=green")
            elif edge_type == "llm":
                edge_attrs.append("color=purple")
            
            edge_attr_str = ", ".join(edge_attrs)
            if edge_attr_str:
                lines.append(f'    "{source_id}" -> "{target_id}" [{edge_attr_str}];')
            else:
                lines.append(f'    "{source_id}" -> "{target_id}";')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _simplify_domain_name(self, domain: str) -> str:
        """Simplify domain name for grouping."""
        domain_mappings = {
            "Mathematical Prerequisites": "Mathematics",
            "Units and Measurements": "General Physics",
            "Problem-Solving Strategies": "General Physics",
            "Electricity and Magnetism": "Electricity",
            "Waves and Optics": "Waves"
        }
        
        for full_name, simplified in domain_mappings.items():
            if full_name in domain:
                return simplified
        
        # Extract first word for simplification
        first_word = domain.split()[0] if domain else "General"
        return first_word if first_word else "General Physics"
    
    def _create_graph_from_hierarchy(self, data: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """Create graph nodes and edges from hierarchy structure."""
        nodes = []
        edges = []
        
        hierarchy = data.get("hierarchy", {})
        if not hierarchy:
            return nodes, edges
        
        # Create nodes from hierarchy structure
        node_id = 0
        
        def process_level(obj, parent_id=None, level=1, path=""):
            nonlocal node_id
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_id = f"node_{node_id}"
                    node_id += 1
                    
                    # Determine node type and color
                    node_type = {
                        1: "domain",
                        2: "category", 
                        3: "concept",
                        4: "topic",
                        5: "subtopic",
                        6: "learning_element"
                    }.get(level, "content")
                    
                    color = {
                        "domain": "lightblue",
                        "category": "lightgreen",
                        "concept": "lightyellow",
                        "topic": "lightcoral",
                        "subtopic": "lightpink",
                        "learning_element": "lavender"
                    }.get(node_type, "white")
                    
                    nodes.append({
                        "id": current_id,
                        "label": self._truncate_title(key),
                        "title": key,
                        "type": node_type,
                        "color": color,
                        "level": level
                    })
                    
                    # Create edge from parent
                    if parent_id:
                        edges.append({
                            "source": parent_id,
                            "target": current_id,
                            "type": "hierarchy"
                        })
                    
                    # Recurse for children (but limit depth to avoid huge graphs)
                    if level < 4:  # Only go 4 levels deep
                        process_level(value, current_id, level + 1, f"{path}/{key}")
            
            elif isinstance(obj, list):
                # For lists (learning elements), create nodes for each item
                for i, item in enumerate(obj[:5]):  # Limit to first 5 items
                    if isinstance(item, str):
                        current_id = f"node_{node_id}"
                        node_id += 1
                        
                        nodes.append({
                            "id": current_id,
                            "label": self._truncate_title(item),
                            "title": item,
                            "type": "learning_element",
                            "color": "lavender",
                            "level": level
                        })
                        
                        if parent_id:
                            edges.append({
                                "source": parent_id,
                                "target": current_id,
                                "type": "hierarchy"
                            })
        
        # Process each classification (core, electives)
        for classification, content in hierarchy.items():
            class_id = f"node_{node_id}"
            node_id += 1
            
            nodes.append({
                "id": class_id,
                "label": classification.title(),
                "title": classification,
                "type": "classification",
                "color": "lightsteelblue",
                "level": 0
            })
            
            process_level(content, class_id, 1)
        
        self.logger.info(f"Created {len(nodes)} nodes and {len(edges)} edges from hierarchy")
        return nodes, edges
    
    def _sanitize_node_id(self, node_id: str) -> str:
        """Sanitize node ID for DOT format."""
        # Replace problematic characters
        sanitized = node_id.replace("→", "_").replace(" ", "_").replace("'", "").replace('"', "")
        return sanitized[:50]  # Limit length
    
    def _truncate_title(self, title: str) -> str:
        """Truncate title for display in graph."""
        if len(title) > 30:
            return title[:27] + "..."
        return title
    
    def export_to_duckdb(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export curriculum to DuckDB database format."""
        self.logger.start_timer("duckdb_export")
        
        try:
            # Use SQLite for compatibility (DuckDB might not be available)
            db_path = Path(output_path)
            
            # Create connection
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create tables and insert data
            tables_created = self._create_database_tables(cursor, data)
            
            # Commit and close
            conn.commit()
            conn.close()
            
            # Calculate statistics
            file_size = db_path.stat().st_size / (1024 * 1024)  # MB
            
            # Count total records
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            total_records = 0
            for table_name in ["topics", "prerequisites", "standards_mappings", "hierarchy"]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    total_records += count
                except:
                    pass
            conn.close()
            
            self.export_stats["duckdb"] = {
                "tables": tables_created,
                "records": total_records,
                "size_mb": round(file_size, 2)
            }
            
            self.logger.info(f"DuckDB export: {tables_created} tables, {total_records} records, {file_size:.2f} MB")
            self.logger.end_timer("duckdb_export")
            return True
            
        except Exception as e:
            self.logger.error(f"DuckDB export failed: {e}")
            self.logger.end_timer("duckdb_export")
            return False
    
    def _create_database_tables(self, cursor, data: Dict[str, Any]) -> int:
        """Create database tables and insert data."""
        tables_created = 0
        
        # Topics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                topic_id TEXT PRIMARY KEY,
                title TEXT,
                level INTEGER,
                domain TEXT,
                full_path TEXT,
                classification TEXT
            )
        """)
        
        topics = data.get("topics", {})
        for topic_id, topic_data in topics.items():
            cursor.execute("""
                INSERT OR REPLACE INTO topics VALUES (?, ?, ?, ?, ?, ?)
            """, (
                topic_id,
                topic_data.get("title", ""),
                topic_data.get("level", 0),
                topic_data.get("domain", ""),
                topic_data.get("full_path", ""),
                topic_data.get("classification", "")
            ))
        
        tables_created += 1
        
        # Prerequisites table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prerequisites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_topic TEXT,
                target_topic TEXT,
                relationship_type TEXT,
                confidence REAL,
                reasoning TEXT,
                domain TEXT
            )
        """)
        
        prerequisites = data.get("prerequisites", [])
        for prereq in prerequisites:
            cursor.execute("""
                INSERT INTO prerequisites VALUES (NULL, ?, ?, ?, ?, ?, ?)
            """, (
                prereq.get("source", ""),
                prereq.get("target", ""),
                prereq.get("type", ""),
                prereq.get("confidence", 0.0),
                prereq.get("reasoning", ""),
                prereq.get("domain", "")
            ))
        
        tables_created += 1
        
        # Standards mappings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS standards_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT,
                standard_type TEXT,
                standard_code TEXT,
                standard_description TEXT,
                confidence REAL,
                bloom_level TEXT,
                difficulty_level TEXT,
                application_domains TEXT
            )
        """)
        
        standards_mappings = data.get("standards_mappings", {})
        for standard_type, mappings in standards_mappings.items():
            for mapping in mappings:
                cursor.execute("""
                    INSERT INTO standards_mappings VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mapping.get("topic_id", ""),
                    mapping.get("standard_type", ""),
                    mapping.get("standard_code", ""),
                    mapping.get("standard_description", ""),
                    mapping.get("confidence", 0.0),
                    mapping.get("bloom_level", ""),
                    mapping.get("difficulty_level", ""),
                    "; ".join(mapping.get("application_domains", []))
                ))
        
        tables_created += 1
        
        # Hierarchy summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hierarchy (
                level INTEGER,
                level_name TEXT,
                item_count INTEGER,
                classification TEXT
            )
        """)
        
        # Calculate hierarchy statistics
        hierarchy_stats = self._calculate_hierarchy_statistics(data.get("hierarchy", {}))
        for stat in hierarchy_stats:
            cursor.execute("""
                INSERT INTO hierarchy VALUES (?, ?, ?, ?)
            """, stat)
        
        tables_created += 1
        
        return tables_created
    
    def _calculate_hierarchy_statistics(self, hierarchy: Dict) -> List[Tuple]:
        """Calculate statistics for hierarchy table."""
        stats = []
        
        def count_by_level(obj, classification, level=1):
            counts = defaultdict(int)
            
            if isinstance(obj, dict):
                counts[level] += len(obj)
                for value in obj.values():
                    sub_counts = count_by_level(value, classification, level + 1)
                    for sub_level, count in sub_counts.items():
                        counts[sub_level] += count
            elif isinstance(obj, list):
                counts[level] += len(obj)
            
            return counts
        
        level_names = [
            "Classification", "Domain", "Category", "Concept", "Topic", "Subtopic", "Learning Element"
        ]
        
        for classification, content in hierarchy.items():
            level_counts = count_by_level(content, classification)
            for level, count in level_counts.items():
                level_name = level_names[level] if level < len(level_names) else f"Level {level}"
                stats.append((level, level_name, count, classification))
        
        return stats
    
    def export_all_formats(self, data: Dict[str, Any], base_path: str) -> Dict[str, bool]:
        """Export curriculum data to all supported formats."""
        self.logger.start_timer("all_formats_export")
        
        base_path = Path(base_path)
        results = {}
        
        # Define output files
        output_files = {
            "tsv": base_path.with_suffix(".tsv"),
            "json": base_path.with_suffix(".json"),
            "dot": base_path.with_suffix(".dot"),
            "duckdb": base_path.with_suffix(".db")
        }
        
        # Export each format
        export_methods = {
            "tsv": self.export_to_tsv,
            "json": self.export_to_json,
            "dot": self.export_to_dot,
            "duckdb": self.export_to_duckdb
        }
        
        # Always export all required formats, regardless of config
        required_formats = ["tsv", "json", "dot", "duckdb"]
        formats_to_export = required_formats if not self.config.export_formats else self.config.export_formats
        
        for format_name in formats_to_export:
            if format_name in export_methods:
                output_path = output_files[format_name]
                success = export_methods[format_name](data, str(output_path))
                results[format_name] = success
                
                if success:
                    self.logger.info(f"Successfully exported {format_name} to {output_path}")
                else:
                    self.logger.error(f"Failed to export {format_name}")
            else:
                self.logger.warning(f"Unknown export format: {format_name}")
                results[format_name] = False
        
        self.logger.end_timer("all_formats_export")
        return results
    
    def create_export_summary(self, export_results: Dict[str, bool], 
                            input_metadata: Dict) -> Dict[str, Any]:
        """Create summary of export results."""
        successful_exports = [fmt for fmt, success in export_results.items() if success]
        failed_exports = [fmt for fmt, success in export_results.items() if not success]
        
        summary = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_formats": len(export_results),
                "successful_exports": len(successful_exports),
                "failed_exports": len(failed_exports),
                "success_rate": len(successful_exports) / len(export_results) if export_results else 0
            },
            "format_results": export_results,
            "export_statistics": self.export_stats,
            "input_metadata": input_metadata
        }
        
        return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Multi-Format Export Engine")
    parser.add_argument("--input", "-i", default="standards_mapped_curriculum.json",
                       help="Input standards-mapped curriculum file")
    parser.add_argument("--output", "-o", default="curriculum_export",
                       help="Output base path (extensions will be added)")
    parser.add_argument("--config", "-c", default="config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--formats", nargs="+", 
                       choices=["tsv", "json", "dot", "duckdb"],
                       help="Specific formats to export")
    parser.add_argument("--summary", "-s", 
                       help="Path to save export summary JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    if args.formats:
        config.export_formats = args.formats
    
    logger = CurriculumLogger("step7_export", "DEBUG" if args.verbose else "INFO")
    file_manager = FileManager(logger)
    
    logger.info("Starting Multi-Format Export Engine")
    logger.info(f"Export formats: {config.export_formats}")
    
    # Load input data
    logger.start_timer("data_loading")
    curriculum_data = file_manager.load_json(args.input)
    if not curriculum_data:
        logger.error(f"Failed to load input file: {args.input}")
        return 1
    
    logger.end_timer("data_loading")
    
    # Export to all formats
    exporter = MultiFormatExporter(config, logger)
    
    try:
        export_results = exporter.export_all_formats(curriculum_data, args.output)
        
        # Create summary
        input_metadata = curriculum_data.get("metadata", {})
        summary = exporter.create_export_summary(export_results, input_metadata)
        
        # Save summary if requested
        if args.summary:
            if file_manager.save_json(summary, args.summary):
                logger.info(f"Export summary saved to: {args.summary}")
        
        # Performance summary
        logger.log_performance_summary()
        
        # Results summary
        successful = [fmt for fmt, success in export_results.items() if success]
        failed = [fmt for fmt, success in export_results.items() if not success]
        
        logger.info("Export Results:")
        logger.info(f"  Successful formats: {', '.join(successful) if successful else 'None'}")
        logger.info(f"  Failed formats: {', '.join(failed) if failed else 'None'}")
        logger.info(f"  Success rate: {len(successful)}/{len(export_results)} ({len(successful)/len(export_results)*100:.1f}%)")
        
        # Format-specific statistics
        for fmt, stats in exporter.export_stats.items():
            if fmt in successful:
                if fmt == "tsv":
                    logger.info(f"  TSV: {stats['records']} records, {stats['size_mb']} MB")
                elif fmt == "json":
                    logger.info(f"  JSON: {stats['size_mb']} MB, depth {stats['hierarchy_depth']}")
                elif fmt == "dot":
                    logger.info(f"  DOT: {stats['nodes']} nodes, {stats['edges']} edges, {stats['size_mb']} MB")
                elif fmt == "duckdb":
                    logger.info(f"  DuckDB: {stats['tables']} tables, {stats['records']} records, {stats['size_mb']} MB")
        
        return 0 if successful else 1
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())