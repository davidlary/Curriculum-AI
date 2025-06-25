#!/usr/bin/env python3
"""
Master Curriculum Generation Orchestrator

This script orchestrates the complete curriculum generation pipeline from 
extracted TOCs to fully-featured curriculum with exports. It runs Steps 3-7
in sequence with proper error handling and progress tracking.

Pipeline Overview:
1. Input: Extracted TOCs (from Step 2)
2. Step 3: Core/Elective Classification
3. Step 4: Six-Level Hierarchy Building  
4. Step 5: Prerequisites & Dependencies
5. Step 6: Standards Mapping
6. Step 7: Multi-Format Export
7. Output: Complete curriculum in multiple formats

Features:
- Sequential execution with dependency checking
- Comprehensive error handling and recovery
- Progress tracking and performance monitoring
- Configurable step selection
- Automatic intermediate file management
- Detailed logging and reporting
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.curriculum_utils import (
    CurriculumConfig, CurriculumLogger, FileManager, 
    DataValidator, load_config
)


@dataclass
class PipelineStep:
    """Represents a single step in the curriculum generation pipeline."""
    step_number: int
    name: str
    script_path: str
    input_file: str
    output_file: str
    description: str
    required: bool = True
    dependencies: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ExecutionResult:
    """Results from executing a pipeline step."""
    step: PipelineStep
    success: bool
    execution_time: float
    output_file_created: bool
    error_message: str = ""
    return_code: int = 0


class MasterCurriculumOrchestrator:
    """Main orchestrator for the curriculum generation pipeline."""
    
    def __init__(self, config: CurriculumConfig, logger: CurriculumLogger, 
                 base_output_dir: str = "Curriculum"):
        self.config = config
        self.logger = logger
        self.file_manager = FileManager(logger)
        self.base_output_dir = Path(base_output_dir)
        
        # Ensure output directory exists
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Pipeline configuration
        self.steps = self._define_pipeline_steps()
        self.execution_results = []
        
        # State tracking
        self.pipeline_start_time = None
        self.current_step = None
    
    def _define_pipeline_steps(self) -> List[PipelineStep]:
        """Define all pipeline steps with their configurations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = getattr(self.config, 'subject', 'Physics')
        language = getattr(self.config, 'language', 'English')
        
        base_name = f"{subject}_{language}"
        
        return [
            PipelineStep(
                step_number=3,
                name="Core/Elective Classification",
                script_path="scripts/step3_core_elective_classification.py",
                input_file="../TOCs/tocs_extracted.json",  # From Step 2
                output_file=f"{base_name}_classified_curriculum.json",
                description="Classify topics as core or elective using multi-method analysis",
                dependencies=[]
            ),
            PipelineStep(
                step_number=4,
                name="Six-Level Hierarchy Building",
                script_path="scripts/step4_six_level_hierarchy.py",
                input_file=f"{base_name}_classified_curriculum.json",
                output_file=f"{base_name}_six_level_hierarchy.json",
                description="Build comprehensive six-level hierarchical curriculum structure",
                dependencies=[3]
            ),
            PipelineStep(
                step_number=5,
                name="Prerequisites & Dependencies",
                script_path="scripts/step5_prerequisites_dependencies.py",
                input_file=f"{base_name}_six_level_hierarchy.json",
                output_file=f"{base_name}_prerequisites_mapped.json",
                description="Generate prerequisite relationships and dependency graphs",
                dependencies=[4]
            ),
            PipelineStep(
                step_number=6,
                name="Standards Mapping",
                script_path="scripts/step6_standards_mapping.py",
                input_file=f"{base_name}_prerequisites_mapped.json",
                output_file=f"{base_name}_standards_mapped.json",
                description="Map curriculum to educational standards (MCAT, IB, A-Level, etc.)",
                dependencies=[5]
            ),
            PipelineStep(
                step_number=7,
                name="Multi-Format Export",
                script_path="scripts/step7_multi_format_export.py",
                input_file=f"{base_name}_standards_mapped.json",
                output_file=f"{base_name}_complete_curriculum",  # Multiple extensions
                description="Export curriculum in multiple formats (TSV, JSON, DOT, DuckDB)",
                dependencies=[6]
            )
        ]
    
    def validate_input_file(self, step: PipelineStep) -> bool:
        """Validate that the input file for a step exists."""
        # Handle both absolute and relative paths
        if step.input_file.startswith('../'):
            # Path relative to project root
            project_root = Path(__file__).parent.parent
            input_path = project_root / step.input_file.lstrip('../')
        else:
            # Path relative to output directory
            input_path = self.base_output_dir / step.input_file
        
        if not input_path.exists():
            self.logger.error(f"Input file missing for Step {step.step_number}: {input_path}")
            return False
        
        # Validate file is not empty and is valid JSON
        try:
            data = self.file_manager.load_json(str(input_path))
            if not data:
                self.logger.error(f"Input file empty or invalid JSON: {input_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to validate input file {input_path}: {e}")
            return False
        
        return True
    
    def check_dependencies(self, step: PipelineStep) -> bool:
        """Check if all dependencies for a step have been completed successfully."""
        for dep_step_num in step.dependencies:
            # Find the dependency step result
            dep_result = next(
                (r for r in self.execution_results if r.step.step_number == dep_step_num),
                None
            )
            
            if not dep_result:
                self.logger.error(f"Dependency Step {dep_step_num} not yet executed")
                return False
            
            if not dep_result.success:
                self.logger.error(f"Dependency Step {dep_step_num} failed")
                return False
        
        return True
    
    def execute_step(self, step: PipelineStep, dry_run: bool = False) -> ExecutionResult:
        """Execute a single pipeline step."""
        self.logger.info(f"Starting Step {step.step_number}: {step.name}")
        self.logger.info(f"Description: {step.description}")
        
        start_time = time.time()
        self.current_step = step
        
        # Validate dependencies
        if not self.check_dependencies(step):
            return ExecutionResult(
                step=step,
                success=False,
                execution_time=0,
                output_file_created=False,
                error_message="Dependency check failed"
            )
        
        # Validate input file
        if not self.validate_input_file(step):
            return ExecutionResult(
                step=step,
                success=False,
                execution_time=0,
                output_file_created=False,
                error_message="Input file validation failed"
            )
        
        # Prepare command
        # Handle both absolute and relative paths for input
        if step.input_file.startswith('../'):
            # Path relative to project root
            project_root = Path(__file__).parent.parent
            input_path = project_root / step.input_file.lstrip('../')
        else:
            # Path relative to output directory
            input_path = self.base_output_dir / step.input_file
        
        output_path = self.base_output_dir / step.output_file
        
        cmd = [
            "python", step.script_path,
            "--input", str(input_path),
            "--output", str(output_path),
            "--config", self.config.config_path if hasattr(self.config, 'config_path') else "config/curriculum_config.json"
        ]
        
        # Add common flags
        if self.logger.logger.level <= 10:  # DEBUG level
            cmd.append("--verbose")
        
        if dry_run:
            self.logger.info(f"DRY RUN - Would execute: {' '.join(cmd)}")
            return ExecutionResult(
                step=step,
                success=True,
                execution_time=0,
                output_file_created=False,
                error_message="Dry run - not executed"
            )
        
        # Execute the step
        self.logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            execution_time = time.time() - start_time
            
            # Log stdout and stderr
            if result.stdout:
                self.logger.info(f"Step {step.step_number} output:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"Step {step.step_number} errors:\n{result.stderr}")
            
            # Check if execution was successful
            success = result.returncode == 0
            
            # Check if output file was created (for steps 3-6)
            output_file_created = False
            if step.step_number < 7:  # Steps 3-6 create single files
                output_file_created = output_path.exists()
            else:  # Step 7 creates multiple files
                # Check for any of the export formats
                for ext in ['.tsv', '.json', '.dot', '.db']:
                    if Path(str(output_path) + ext).exists():
                        output_file_created = True
                        break
            
            if success and not output_file_created:
                self.logger.warning(f"Step {step.step_number} reported success but output file not found")
            
            return ExecutionResult(
                step=step,
                success=success,
                execution_time=execution_time,
                output_file_created=output_file_created,
                error_message=result.stderr if result.stderr else "",
                return_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Step {step.step_number} timed out after 1 hour"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                step=step,
                success=False,
                execution_time=execution_time,
                output_file_created=False,
                error_message=error_msg,
                return_code=-1
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to execute Step {step.step_number}: {e}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                step=step,
                success=False,
                execution_time=execution_time,
                output_file_created=False,
                error_message=error_msg,
                return_code=-2
            )
    
    def run_pipeline(self, start_step: int = 3, end_step: int = 7, 
                    skip_steps: List[int] = None, dry_run: bool = False) -> bool:
        """Run the complete pipeline or a subset of steps."""
        if skip_steps is None:
            skip_steps = []
        
        self.pipeline_start_time = time.time()
        self.logger.info("Starting Master Curriculum Generation Pipeline")
        self.logger.info(f"Steps to execute: {start_step} to {end_step}")
        self.logger.info(f"Skipping steps: {skip_steps if skip_steps else 'None'}")
        self.logger.info(f"Output directory: {self.base_output_dir}")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - No actual execution")
        
        # Filter steps to execute
        steps_to_run = [
            step for step in self.steps 
            if start_step <= step.step_number <= end_step 
            and step.step_number not in skip_steps
        ]
        
        if not steps_to_run:
            self.logger.error("No steps to execute")
            return False
        
        # Execute steps in sequence
        overall_success = True
        
        for step in steps_to_run:
            result = self.execute_step(step, dry_run)
            self.execution_results.append(result)
            
            if result.success:
                self.logger.info(f"✓ Step {step.step_number} completed successfully in {result.execution_time:.2f}s")
            else:
                self.logger.error(f"✗ Step {step.step_number} failed: {result.error_message}")
                overall_success = False
                
                # Check if this step is required
                if step.required:
                    self.logger.error("Required step failed - stopping pipeline")
                    break
                else:
                    self.logger.warning("Optional step failed - continuing pipeline")
        
        # Generate final report
        self._generate_execution_report()
        
        return overall_success
    
    def _generate_execution_report(self) -> None:
        """Generate a comprehensive execution report."""
        total_time = time.time() - self.pipeline_start_time if self.pipeline_start_time else 0
        
        successful_steps = [r for r in self.execution_results if r.success]
        failed_steps = [r for r in self.execution_results if not r.success]
        
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE EXECUTION REPORT")
        self.logger.info("="*60)
        self.logger.info(f"Total execution time: {total_time:.2f}s")
        self.logger.info(f"Steps executed: {len(self.execution_results)}")
        self.logger.info(f"Successful steps: {len(successful_steps)}")
        self.logger.info(f"Failed steps: {len(failed_steps)}")
        self.logger.info(f"Success rate: {len(successful_steps)/len(self.execution_results)*100:.1f}%")
        
        # Individual step results
        self.logger.info("\nStep-by-step results:")
        for result in self.execution_results:
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            self.logger.info(f"  Step {result.step.step_number}: {status} ({result.execution_time:.2f}s)")
            if not result.success:
                self.logger.info(f"    Error: {result.error_message}")
        
        # Performance breakdown
        if successful_steps:
            self.logger.info("\nPerformance breakdown:")
            for result in sorted(successful_steps, key=lambda r: r.execution_time, reverse=True):
                percentage = (result.execution_time / total_time) * 100 if total_time > 0 else 0
                self.logger.info(f"  Step {result.step.step_number}: {result.execution_time:.2f}s ({percentage:.1f}%)")
        
        # Output files created
        self.logger.info("\nOutput files:")
        for result in self.execution_results:
            if result.output_file_created:
                output_path = self.base_output_dir / result.step.output_file
                if result.step.step_number == 7:  # Multi-format export
                    self.logger.info(f"  Step {result.step.step_number}: Multiple formats in {self.base_output_dir}/")
                else:
                    self.logger.info(f"  Step {result.step.step_number}: {output_path}")
        
        self.logger.info("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Master Curriculum Generation Orchestrator")
    parser.add_argument("--input-dir", "-i", default=".",
                       help="Input directory containing extracted TOCs")
    parser.add_argument("--output-dir", "-o", default="Curriculum",
                       help="Output directory for generated curriculum files")
    parser.add_argument("--config", "-c", default="config/curriculum_config.json",
                       help="Configuration file path")
    parser.add_argument("--start-step", type=int, default=3,
                       help="Starting step number (default: 3)")
    parser.add_argument("--end-step", type=int, default=7,
                       help="Ending step number (default: 7)")
    parser.add_argument("--skip-steps", nargs="+", type=int,
                       help="Steps to skip (e.g., --skip-steps 5 6)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    config.config_path = args.config  # Store for subprocess calls
    
    logger = CurriculumLogger("master_orchestrator", "DEBUG" if args.verbose else "INFO")
    
    # Create orchestrator and run pipeline
    orchestrator = MasterCurriculumOrchestrator(config, logger, args.output_dir)
    
    try:
        success = orchestrator.run_pipeline(
            start_step=args.start_step,
            end_step=args.end_step,
            skip_steps=args.skip_steps or [],
            dry_run=args.dry_run
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())