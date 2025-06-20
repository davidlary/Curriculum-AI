"""
Unit tests for core.parallel_processor module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import time
import concurrent.futures
from pathlib import Path
import shutil

from core.parallel_processor import (
    ParallelProcessor, ProcessingTask, ProcessingResult, 
    SystemMonitor, create_parallel_task_batches
)
from core.config import OpenBooksConfig


class TestProcessingTask(unittest.TestCase):
    """Test cases for ProcessingTask dataclass"""

    def test_task_creation(self):
        """Test ProcessingTask creation"""
        task = ProcessingTask(
            task_id="test_task",
            task_type="discovery",
            data={"key": "value"},
            priority=1,
            retry_count=0,
            max_retries=3
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.task_type, "discovery")
        self.assertEqual(task.data, {"key": "value"})
        self.assertEqual(task.priority, 1)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.max_retries, 3)


class TestProcessingResult(unittest.TestCase):
    """Test cases for ProcessingResult dataclass"""

    def test_result_creation(self):
        """Test ProcessingResult creation"""
        result = ProcessingResult(
            task_id="test_task",
            success=True,
            data={"result": "data"},
            error=None,
            duration=1.5
        )
        
        self.assertEqual(result.task_id, "test_task")
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"result": "data"})
        self.assertIsNone(result.error)
        self.assertEqual(result.duration, 1.5)


class TestParallelProcessor(unittest.TestCase):
    """Test cases for ParallelProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OpenBooksConfig()
        self.config.max_workers = 4
        self.config.max_discovery_workers = 2
        self.config.max_clone_workers = 2
        self.config.max_processing_workers = 2
        self.processor = ParallelProcessor(self.config)

    def tearDown(self):
        """Clean up test fixtures"""
        if self.processor.is_running:
            self.processor.stop()

    def test_processor_initialization(self):
        """Test ParallelProcessor initialization"""
        self.assertIsInstance(self.processor.config, OpenBooksConfig)
        self.assertFalse(self.processor.is_running)
        self.assertEqual(self.processor.stats['tasks_completed'], 0)
        self.assertEqual(self.processor.stats['tasks_failed'], 0)
        self.assertIsNone(self.processor.discovery_executor)

    def test_start_and_stop(self):
        """Test starting and stopping processor"""
        # Test start
        self.processor.start()
        
        self.assertTrue(self.processor.is_running)
        self.assertIsNotNone(self.processor.discovery_executor)
        self.assertIsNotNone(self.processor.clone_executor)
        self.assertIsNotNone(self.processor.processing_executor)
        self.assertIsNotNone(self.processor.io_executor)
        
        # Test stop
        self.processor.stop()
        
        self.assertFalse(self.processor.is_running)

    def test_start_already_running(self):
        """Test starting processor when already running"""
        self.processor.start()
        
        # Try to start again
        with patch('core.parallel_processor.logger') as mock_logger:
            self.processor.start()
            mock_logger.warning.assert_called_with("ParallelProcessor already running")

    def test_stop_not_running(self):
        """Test stopping processor when not running"""
        # Should not raise an error
        self.processor.stop()
        self.assertFalse(self.processor.is_running)

    def test_submit_discovery_tasks(self):
        """Test submitting discovery tasks"""
        tasks = [
            ProcessingTask("task1", "github_search_limited", {}),
            ProcessingTask("task2", "subject_search", {"subject": "physics"})
        ]
        
        futures = self.processor.submit_discovery_tasks(tasks)
        
        self.assertEqual(len(futures), 2)
        self.assertTrue(self.processor.is_running)
        
        # Wait for completion
        for future in futures:
            result = future.result(timeout=5)
            self.assertIsInstance(result, ProcessingResult)

    def test_submit_clone_tasks(self):
        """Test submitting clone tasks"""
        tasks = [
            ProcessingTask("clone1", "clone_repository", {
                "book_info": {
                    "repo": "test-repo",
                    "clone_url": "https://github.com/test/repo.git"
                }
            })
        ]
        
        with patch('core.parallel_processor.RepositoryManager') as mock_manager:
            mock_manager.return_value.clone_repository.return_value = True
            
            futures = self.processor.submit_clone_tasks(tasks)
            
            self.assertEqual(len(futures), 1)
            
            # Wait for completion
            result = futures[0].result(timeout=5)
            self.assertIsInstance(result, ProcessingResult)

    def test_submit_processing_tasks(self):
        """Test submitting processing tasks"""
        tasks = [
            ProcessingTask("process1", "extract_text", {
                "book_path": "/tmp/test"
            })
        ]
        
        futures = self.processor.submit_processing_tasks(tasks)
        
        self.assertEqual(len(futures), 1)
        
        # Wait for completion
        result = futures[0].result(timeout=5)
        self.assertIsInstance(result, ProcessingResult)

    def test_execute_discovery_task_github_search(self):
        """Test executing GitHub search discovery task"""
        task = ProcessingTask("test", "github_search_limited", {})
        
        result = self.processor._execute_discovery_task(task)
        
        self.assertTrue(result.success)
        self.assertEqual(result.task_id, "test")
        self.assertIn('message', result.data)

    def test_execute_discovery_task_subject_search(self):
        """Test executing subject search discovery task"""
        task = ProcessingTask("test", "subject_search", {"subject": "physics"})
        
        result = self.processor._execute_discovery_task(task)
        
        self.assertTrue(result.success)
        self.assertEqual(result.data['subject'], "physics")

    def test_execute_discovery_task_unknown_type(self):
        """Test executing unknown discovery task type"""
        task = ProcessingTask("test", "unknown_type", {})
        
        result = self.processor._execute_discovery_task(task)
        
        self.assertFalse(result.success)
        self.assertIn("Unknown discovery task type", result.error)

    def test_execute_clone_task_success(self):
        """Test executing successful clone task"""
        task = ProcessingTask("test", "clone_repository", {
            "book_info": {
                "repo": "test-repo",
                "clone_url": "https://github.com/test/repo.git"
            }
        })
        
        with patch('core.parallel_processor.RepositoryManager') as mock_manager:
            mock_manager.return_value.clone_repository.return_value = True
            
            result = self.processor._execute_clone_task(task)
            
            self.assertTrue(result.success)
            self.assertIn('message', result.data)

    def test_execute_clone_task_already_exists(self):
        """Test executing clone task for repository that already exists"""
        task = ProcessingTask("test", "clone_repository", {
            "book_info": {
                "repo": "test-repo",
                "clone_url": "https://github.com/test/repo.git"
            }
        })
        
        with patch('core.parallel_processor.RepositoryManager') as mock_manager:
            mock_manager.return_value.clone_repository.side_effect = Exception("Repository already exists")
            
            result = self.processor._execute_clone_task(task)
            
            self.assertTrue(result.success)  # Should treat as success
            self.assertIn('already exists', result.data['message'])

    def test_execute_clone_task_no_url(self):
        """Test executing clone task without clone URL"""
        task = ProcessingTask("test", "clone_repository", {
            "book_info": {
                "repo": "test-repo"
                # No clone_url
            }
        })
        
        result = self.processor._execute_clone_task(task)
        
        self.assertFalse(result.success)
        self.assertIn('No clone URL', result.error)

    def test_execute_processing_task_extract_text(self):
        """Test executing text extraction processing task"""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"test content")
        temp_file.close()
        
        try:
            task = ProcessingTask("test", "extract_text", {
                "book_path": temp_file.name
            })
            
            result = self.processor._execute_processing_task(task)
            
            self.assertTrue(result.success)
            self.assertIn('message', result.data)
            self.assertIn('size_mb', result.data)
        finally:
            Path(temp_file.name).unlink()

    def test_execute_processing_task_nonexistent_path(self):
        """Test executing processing task with non-existent path"""
        task = ProcessingTask("test", "extract_text", {
            "book_path": "/nonexistent/path"
        })
        
        result = self.processor._execute_processing_task(task)
        
        self.assertFalse(result.success)
        self.assertIn('does not exist', result.error)

    def test_execute_processing_task_generate_catalog(self):
        """Test executing catalog generation task"""
        task = ProcessingTask("test", "generate_catalog", {})
        
        result = self.processor._execute_processing_task(task)
        
        self.assertTrue(result.success)
        self.assertIn('Catalog generated', result.data['message'])

    def test_process_batch_parallel(self):
        """Test batch parallel processing"""
        discovery_tasks = [
            ProcessingTask("disc1", "github_search_limited", {})
        ]
        
        clone_tasks = [
            ProcessingTask("clone1", "clone_repository", {
                "book_info": {
                    "repo": "test-repo",
                    "clone_url": "https://github.com/test/repo.git"
                }
            })
        ]
        
        processing_tasks = [
            ProcessingTask("proc1", "generate_catalog", {})
        ]
        
        with patch('core.parallel_processor.RepositoryManager') as mock_manager:
            mock_manager.return_value.clone_repository.return_value = True
            
            results = self.processor.process_batch_parallel(
                discovery_tasks=discovery_tasks,
                clone_tasks=clone_tasks,
                processing_tasks=processing_tasks
            )
            
            self.assertIn('discovery', results)
            self.assertIn('clone', results)
            self.assertIn('processing', results)
            
            self.assertEqual(len(results['discovery']), 1)
            self.assertEqual(len(results['clone']), 1)
            self.assertEqual(len(results['processing']), 1)

    def test_get_stats(self):
        """Test getting processor statistics"""
        stats = self.processor.get_stats()
        
        self.assertIn('tasks_completed', stats)
        self.assertIn('tasks_failed', stats)
        self.assertIn('total_duration', stats)
        self.assertIn('runtime', stats)
        self.assertIn('tasks_per_second', stats)
        self.assertIn('is_running', stats)

    def test_get_system_status(self):
        """Test getting system status"""
        status = self.processor.get_system_status()
        
        self.assertIsInstance(status, dict)


class TestSystemMonitor(unittest.TestCase):
    """Test cases for SystemMonitor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.monitor = SystemMonitor()

    def test_monitor_initialization(self):
        """Test SystemMonitor initialization"""
        self.assertIsNotNone(self.monitor.process)
        self.assertGreater(self.monitor.start_time, 0)

    @patch('core.parallel_processor.psutil')
    def test_get_status(self, mock_psutil):
        """Test getting system status"""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.free = 100 * 1024**3  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.getloadavg.return_value = (1.0, 1.5, 2.0)
        
        mock_process_memory = Mock()
        mock_process_memory.rss = 500 * 1024**2  # 500MB
        self.monitor.process.memory_info.return_value = mock_process_memory
        
        status = self.monitor.get_status()
        
        self.assertEqual(status['cpu_percent'], 25.0)
        self.assertEqual(status['cpu_count'], 8)
        self.assertEqual(status['memory_percent'], 60.0)
        self.assertEqual(status['memory_available_gb'], 8.0)
        self.assertEqual(status['disk_free_gb'], 100.0)
        self.assertEqual(status['process_memory_mb'], 500.0)

    @patch('core.parallel_processor.psutil')
    def test_get_status_error(self, mock_psutil):
        """Test getting system status with error"""
        mock_psutil.cpu_percent.side_effect = Exception("Test error")
        
        status = self.monitor.get_status()
        
        self.assertIn('error', status)

    @patch('core.parallel_processor.psutil')
    def test_check_resources_healthy(self, mock_psutil):
        """Test resource check when system is healthy"""
        # Mock healthy system
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.free = 50 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        
        is_healthy, message = self.monitor.check_resources()
        
        self.assertTrue(is_healthy)
        self.assertIn("healthy", message.lower())

    @patch('core.parallel_processor.psutil')
    def test_check_resources_high_cpu(self, mock_psutil):
        """Test resource check with high CPU usage"""
        mock_psutil.cpu_percent.return_value = 95.0
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.free = 50 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        
        is_healthy, message = self.monitor.check_resources()
        
        self.assertFalse(is_healthy)
        self.assertIn("High CPU usage", message)

    @patch('core.parallel_processor.psutil')
    def test_check_resources_high_memory(self, mock_psutil):
        """Test resource check with high memory usage"""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 90.0
        mock_memory.available = 1 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.free = 50 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk
        
        is_healthy, message = self.monitor.check_resources()
        
        self.assertFalse(is_healthy)
        self.assertIn("High memory usage", message)

    @patch('core.parallel_processor.psutil')
    def test_check_resources_low_disk(self, mock_psutil):
        """Test resource check with low disk space"""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.free = 2 * 1024**3  # 2GB - below threshold
        mock_psutil.disk_usage.return_value = mock_disk
        
        is_healthy, message = self.monitor.check_resources()
        
        self.assertFalse(is_healthy)
        self.assertIn("Low disk space", message)


class TestCreateParallelTaskBatches(unittest.TestCase):
    """Test cases for create_parallel_task_batches function"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OpenBooksConfig()
        self.config.books_path = self.temp_dir

    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_create_task_batches_empty(self):
        """Test creating task batches with empty book list"""
        books = []
        
        discovery_tasks, clone_tasks, processing_tasks = create_parallel_task_batches(
            books, self.config
        )
        
        self.assertEqual(len(discovery_tasks), 1)  # Always has limited discovery
        self.assertEqual(len(clone_tasks), 0)
        self.assertEqual(len(processing_tasks), 0)

    def test_create_task_batches_with_books(self):
        """Test creating task batches with books"""
        books = [
            {
                "repo": "osbooks-physics",
                "subject": "physics",
                "clone_url": "https://github.com/openstax/osbooks-physics.git"
            },
            {
                "repo": "osbooks-biology", 
                "subject": "biology",
                "clone_url": "https://github.com/openstax/osbooks-biology.git"
            }
        ]
        
        discovery_tasks, clone_tasks, processing_tasks = create_parallel_task_batches(
            books, self.config
        )
        
        self.assertEqual(len(discovery_tasks), 1)  # Limited discovery
        self.assertEqual(len(clone_tasks), 2)  # One per book
        
        # Check clone task data
        for i, task in enumerate(clone_tasks):
            self.assertEqual(task.task_type, "clone_repository")
            self.assertIn("book_info", task.data)

    def test_create_task_batches_with_existing_books(self):
        """Test creating task batches with existing books directory"""
        # Create test book directories
        books_dir = Path(self.temp_dir)
        
        # Create git repository
        git_repo = books_dir / "osbooks-physics"
        git_repo.mkdir()
        (git_repo / ".git").mkdir()
        (git_repo / "content.txt").write_text("physics content")
        
        # Create substantial directory
        substantial_dir = books_dir / "osbooks-biology"
        substantial_dir.mkdir()
        for i in range(10):
            (substantial_dir / f"file{i}.txt").write_text(f"content {i}")
        
        # Create system files (should be skipped)
        (books_dir / ".DS_Store").write_text("system file")
        (books_dir / "README.md").write_text("readme")
        
        books = []
        
        discovery_tasks, clone_tasks, processing_tasks = create_parallel_task_batches(
            books, self.config
        )
        
        self.assertEqual(len(discovery_tasks), 1)
        self.assertEqual(len(clone_tasks), 0)
        self.assertGreater(len(processing_tasks), 0)  # Should process existing books
        
        # Check that system files are not processed
        processing_paths = [task.data.get('book_path', '') for task in processing_tasks]
        self.assertNotIn(str(books_dir / ".DS_Store"), processing_paths)
        self.assertNotIn(str(books_dir / "README.md"), processing_paths)

    def test_task_priority_assignment(self):
        """Test that physics books get higher priority"""
        books = [
            {
                "repo": "osbooks-physics",
                "subject": "physics",
                "clone_url": "https://github.com/openstax/osbooks-physics.git"
            },
            {
                "repo": "osbooks-biology",
                "subject": "biology", 
                "clone_url": "https://github.com/openstax/osbooks-biology.git"
            }
        ]
        
        discovery_tasks, clone_tasks, processing_tasks = create_parallel_task_batches(
            books, self.config
        )
        
        physics_task = next(t for t in clone_tasks if 'physics' in t.task_id)
        biology_task = next(t for t in clone_tasks if 'biology' in t.task_id)
        
        self.assertEqual(physics_task.priority, 1)  # High priority
        self.assertEqual(biology_task.priority, 2)  # Medium priority


if __name__ == '__main__':
    unittest.main()