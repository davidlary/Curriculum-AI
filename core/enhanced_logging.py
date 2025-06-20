"""
Enhanced logging module for OpenBooks with comprehensive error tracking and structured logging.

This module provides enhanced logging capabilities with better error categorization,
context tracking, and performance monitoring for all OpenBooks operations.
"""

import logging
import traceback
import functools
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
from datetime import datetime
import sys


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_context = getattr(logger, '_context', {})
    
    def __enter__(self):
        self.logger._context = {**self.old_context, **self.context}
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._context = self.old_context


class OperationLogger:
    """Enhanced logger with operation tracking and context."""
    
    def __init__(self, name: str, terminal_ui=None):
        self.logger = logging.getLogger(name)
        self.operation_stack = []
        self.error_counts = {}
        self.performance_metrics = {}
        self._lock = threading.Lock()  # Thread safety for shared state
        self.terminal_ui = terminal_ui  # Reference to terminal UI for notifications
        self.session_error_count = 0  # Count of errors in this session
        self.session_warning_count = 0  # Count of warnings in this session
        self.suppress_terminal_notifications = False  # Flag to suppress terminal notifications
    
    def start_operation(self, operation: str, **context):
        """Start tracking an operation."""
        with self._lock:
            op_info = {
                'name': operation,
                'start_time': time.time(),
                'context': context
            }
            self.operation_stack.append(op_info)
            
            context_str = ', '.join(f"{k}={v}" for k, v in context.items()) if context else ""
            self.logger.info(f"🚀 Starting: {operation}" + (f" ({context_str})" if context_str else ""))
            return op_info
    
    def end_operation(self, success: bool = True, **results):
        """End tracking an operation."""
        with self._lock:
            if not self.operation_stack:
                self.logger.warning("Attempted to end operation but no operation in progress")
                return
            
            op_info = self.operation_stack.pop()
            duration = time.time() - op_info['start_time']
            operation = op_info['name']
            
            # Track performance
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = []
            self.performance_metrics[operation].append(duration)
            
            if success:
                results_str = ', '.join(f"{k}={v}" for k, v in results.items()) if results else ""
                self.logger.info(f"✅ Completed: {operation} ({duration:.2f}s)" + 
                               (f" - {results_str}" if results_str else ""))
            else:
                self.logger.error(f"❌ Failed: {operation} ({duration:.2f}s)")
                
                # Track error counts
                self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
    
    def log_error(self, error: Exception, context: str = "", category: str = "general", 
                  severity: str = "error", recoverable: bool = True):
        """Log an error with enhanced context and categorization."""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'category': category,
            'severity': severity,
            'recoverable': recoverable,
            'timestamp': datetime.now().isoformat(),
            'operation': self.operation_stack[-1]['name'] if self.operation_stack else 'unknown'
        }
        
        # Format error message
        if severity == "critical":
            icon = "🔥"
            log_func = self.logger.critical
        elif severity == "error":
            icon = "❌"
            log_func = self.logger.error
        elif severity == "warning":
            icon = "⚠️"
            log_func = self.logger.warning
        else:
            icon = "ℹ️"
            log_func = self.logger.info
        
        message = f"{icon} {category.upper()}: {error_info['message']}"
        if context:
            message += f" (Context: {context})"
        
        log_func(message)
        
        # Track session counts and notify terminal that something was logged to file
        with self._lock:
            if severity in ["error", "critical"]:
                self.session_error_count += 1
            elif severity == "warning":
                self.session_warning_count += 1
        
        # Only show terminal notifications for critical errors during normal operations
        # Suppress all individual notifications during discovery/bulk operations
        if (self.terminal_ui and severity == "critical" and not self.suppress_terminal_notifications):
            brief_msg = error_info['message'][:50] + "..." if len(error_info['message']) > 50 else error_info['message']
            self.terminal_ui.print_log_notification(severity, category, brief_msg)
        
        # Log traceback for errors and critical issues
        if severity in ["error", "critical"]:
            self.logger.debug(f"Traceback for {error_info['type']}: {traceback.format_exc()}")
        
        # Track error patterns
        with self._lock:
            error_key = f"{category}:{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        return error_info
    
    def log_api_error(self, url: str, status_code: int, response_text: str = "", 
                      retry_attempt: int = 0, max_retries: int = 3):
        """Log API-specific errors with detailed context."""
        severity = "warning" if retry_attempt < max_retries else "error"
        recoverable = retry_attempt < max_retries
        
        context = f"URL={url}, Status={status_code}"
        if retry_attempt > 0:
            context += f", Retry={retry_attempt}/{max_retries}"
        
        if 400 <= status_code < 500:
            category = "api_client_error"
            message = f"Client error {status_code} for {url}"
        elif 500 <= status_code < 600:
            category = "api_server_error" 
            message = f"Server error {status_code} for {url}"
        else:
            category = "api_unknown_error"
            message = f"Unknown API error {status_code} for {url}"
        
        if response_text:
            message += f": {response_text[:200]}"
        
        error = Exception(message)
        return self.log_error(error, context, category, severity, recoverable)
    
    def log_file_error(self, file_path: str, operation: str, error: Exception):
        """Log file operation errors with enhanced context."""
        context = f"File={file_path}, Operation={operation}"
        
        if isinstance(error, FileNotFoundError):
            category = "file_not_found"
            severity = "warning"
            recoverable = True
        elif isinstance(error, PermissionError):
            category = "file_permission"
            severity = "error"
            recoverable = False
        elif isinstance(error, OSError):
            category = "file_system"
            severity = "error" 
            recoverable = False
        else:
            category = "file_unknown"
            severity = "error"
            recoverable = True
        
        return self.log_error(error, context, category, severity, recoverable)
    
    def log_git_error(self, repo_path: str, command: str, error: Exception, 
                      stderr: str = "", returncode: int = None):
        """Log Git operation errors with command context."""
        context = f"Repo={repo_path}, Command={command}"
        if returncode is not None:
            context += f", ExitCode={returncode}"
        
        if "authentication" in str(error).lower() or "permission denied" in str(error).lower():
            category = "git_auth"
            severity = "error"
            recoverable = False
        elif "not found" in str(error).lower() or returncode == 128:
            category = "git_not_found"
            severity = "warning"
            recoverable = True
        elif "network" in str(error).lower() or "timeout" in str(error).lower():
            category = "git_network"
            severity = "warning"
            recoverable = True
        else:
            category = "git_command"
            severity = "error"
            recoverable = True
        
        if stderr:
            context += f", Stderr={stderr[:200]}"
        
        return self.log_error(error, context, category, severity, recoverable)
    
    def log_performance_warning(self, operation: str, duration: float, threshold: float):
        """Log performance warnings for slow operations."""
        if duration > threshold:
            self.logger.warning(f"⏱️ PERFORMANCE: {operation} took {duration:.2f}s "
                              f"(threshold: {threshold:.2f}s)")
    
    def log_resource_warning(self, resource: str, usage: float, threshold: float):
        """Log resource usage warnings."""
        if usage > threshold:
            self.logger.warning(f"📊 RESOURCE: {resource} usage at {usage:.1f}% "
                              f"(threshold: {threshold:.1f}%)")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'error_categories': dict(self.error_counts),
            'most_common': sorted(self.error_counts.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        for operation, durations in self.performance_metrics.items():
            summary[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }
        return summary
    
    def get_session_counts(self) -> Dict[str, int]:
        """Get session error and warning counts."""
        with self._lock:
            return {
                'errors': self.session_error_count,
                'warnings': self.session_warning_count
            }
    
    def suppress_notifications(self, suppress: bool = True):
        """Control whether terminal notifications are shown."""
        with self._lock:
            self.suppress_terminal_notifications = suppress


def log_exceptions(category: str = "general", severity: str = "error"):
    """Decorator to automatically log exceptions from functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find a logger in the instance or create one
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            elif args and hasattr(args[0], '__class__'):
                logger_name = f"{args[0].__class__.__module__}.{args[0].__class__.__name__}"
                logger = OperationLogger(logger_name)
            else:
                logger = OperationLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"Function={func.__name__}"
                if args:
                    context += f", Args={len(args)}"
                if kwargs:
                    context += f", Kwargs={list(kwargs.keys())}"
                
                logger.log_error(e, context, category, severity)
                raise
        return wrapper
    return decorator


def create_enhanced_logger(name: str, terminal_ui=None) -> OperationLogger:
    """Factory function to create an enhanced logger."""
    return OperationLogger(name, terminal_ui)


def setup_logging(log_file: str = "openbooks.log", verbose: bool = False):
    """Setup comprehensive logging configuration - logs only to file, not terminal."""
    
    # Create formatter for detailed file logging
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed logging - ALL levels go to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger - NO console handler to prevent terminal output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Remove existing handlers
    root_logger.addHandler(file_handler)  # ONLY file handler
    
    # Disable propagation for our loggers to prevent any console output
    root_logger.propagate = False
    
    # Set levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('git').setLevel(logging.WARNING)
    
    return root_logger