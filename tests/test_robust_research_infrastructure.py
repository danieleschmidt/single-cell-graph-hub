"""
Test suite for Robust Research Infrastructure v4.0
Comprehensive testing of fault tolerance, security, and monitoring capabilities
"""

import pytest
import asyncio
import sqlite3
import tempfile
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.scgraph_hub.robust_research_infrastructure import (
    ResearchInfrastructure,
    ResearchDatabaseManager,
    ResearchCacheManager,
    SystemMonitor,
    FaultTolerantTaskExecutor,
    SecurityManager,
    ResearchTask,
    ResearchPhase,
    AlertLevel,
    SystemMetrics,
    SecurityEvent
)


class TestResearchDatabaseManager:
    """Test database management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = ResearchDatabaseManager(self.temp_db.name)
    
    def teardown_method(self):
        """Cleanup test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
        if hasattr(self.db_manager, 'backup_dir'):
            import shutil
            shutil.rmtree(self.db_manager.backup_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database schema initialization."""
        # Check that tables exist
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            
            # Check research_tasks table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_tasks'")
            assert cursor.fetchone() is not None
            
            # Check system_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_metrics'")
            assert cursor.fetchone() is not None
            
            # Check security_events table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='security_events'")
            assert cursor.fetchone() is not None
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_tasks_phase'")
            assert cursor.fetchone() is not None
    
    def test_task_save_and_load(self):
        """Test task saving and loading."""
        task = ResearchTask(
            task_id='test_task_001',
            task_type='hypothesis_generation',
            phase=ResearchPhase.HYPOTHESIS_GENERATION,
            priority=5,
            payload={'test': 'data'},
            created_at=datetime.now(),
            dependencies=['dep1', 'dep2'],
            timeout_seconds=3600
        )
        
        # Save task
        self.db_manager.save_task(task)
        
        # Load task
        loaded_task = self.db_manager.load_task('test_task_001')
        
        assert loaded_task is not None
        assert loaded_task.task_id == task.task_id
        assert loaded_task.task_type == task.task_type
        assert loaded_task.phase == task.phase
        assert loaded_task.priority == task.priority
        assert loaded_task.payload == task.payload
        assert loaded_task.dependencies == task.dependencies
        assert loaded_task.timeout_seconds == task.timeout_seconds
    
    def test_task_update(self):
        """Test task updating."""
        task = ResearchTask(
            task_id='test_task_002',
            task_type='algorithm_discovery',
            phase=ResearchPhase.ALGORITHM_DISCOVERY,
            priority=3,
            payload={'algorithm': 'test'},
            created_at=datetime.now()
        )
        
        # Save initial task
        self.db_manager.save_task(task)
        
        # Update task
        task.started_at = datetime.now()
        task.retry_count = 1
        task.results = {'output': 'test_result'}
        
        self.db_manager.save_task(task)
        
        # Load updated task
        loaded_task = self.db_manager.load_task('test_task_002')
        
        assert loaded_task.started_at is not None
        assert loaded_task.retry_count == 1
        assert loaded_task.results == {'output': 'test_result'}
    
    def test_nonexistent_task_load(self):
        """Test loading nonexistent task."""
        result = self.db_manager.load_task('nonexistent_task')
        assert result is None
    
    def test_metrics_save(self):
        """Test system metrics saving."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=75.5,
            memory_usage=68.2,
            disk_usage=45.1,
            gpu_usage=85.0,
            gpu_memory=4096.0,
            network_io={'bytes_sent': 1024, 'bytes_recv': 2048},
            active_tasks=5,
            queued_tasks=10,
            completed_tasks=50,
            failed_tasks=2,
            system_load=2.5,
            research_throughput=1.2
        )
        
        # Save metrics
        self.db_manager.save_metrics(metrics)
        
        # Verify saved in database
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM system_metrics")
            count = cursor.fetchone()[0]
            assert count == 1
            
            cursor.execute("SELECT cpu_usage, memory_usage FROM system_metrics")
            row = cursor.fetchone()
            assert row[0] == 75.5
            assert row[1] == 68.2
    
    def test_security_event_save(self):
        """Test security event saving."""
        event = SecurityEvent(
            event_id='sec_001',
            timestamp=datetime.now(),
            event_type='suspicious_activity',
            severity=AlertLevel.WARNING,
            source='security_scanner',
            description='Suspicious pattern detected',
            affected_resources=['task_001', 'resource_002'],
            metadata={'pattern': 'rm -rf', 'confidence': 0.95}
        )
        
        # Save event
        self.db_manager.save_security_event(event)
        
        # Verify saved in database
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events")
            count = cursor.fetchone()[0]
            assert count == 1
            
            cursor.execute("SELECT event_type, severity FROM security_events")
            row = cursor.fetchone()
            assert row[0] == 'suspicious_activity'
            assert row[1] == 'warning'
    
    def test_database_backup(self):
        """Test database backup functionality."""
        # Add some data
        task = ResearchTask(
            task_id='backup_test',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={'test': 'backup'},
            created_at=datetime.now()
        )
        self.db_manager.save_task(task)
        
        # Create backup
        backup_path = self.db_manager.backup_database()
        
        assert Path(backup_path).exists()
        assert backup_path.endswith('.gz')
        
        # Verify backup contains data
        import gzip
        import sqlite3
        
        # Extract and verify backup
        extracted_path = backup_path[:-3]  # Remove .gz extension
        with gzip.open(backup_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        with sqlite3.connect(extracted_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task_id FROM research_tasks WHERE task_id='backup_test'")
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == 'backup_test'
        
        # Cleanup
        Path(extracted_path).unlink(missing_ok=True)
        Path(backup_path).unlink(missing_ok=True)
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        task = ResearchTask(
            task_id='transaction_test',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={'test': 'transaction'},
            created_at=datetime.now()
        )
        
        # Save task first
        self.db_manager.save_task(task)
        
        # Test transaction that should fail
        try:
            with self.db_manager.transaction() as conn:
                # Valid operation
                conn.execute("UPDATE research_tasks SET priority = 10 WHERE task_id = 'transaction_test'")
                
                # Invalid operation that should cause rollback
                conn.execute("INSERT INTO nonexistent_table (id) VALUES (1)")
        except sqlite3.OperationalError:
            pass  # Expected error
        
        # Verify rollback occurred - priority should still be 1
        loaded_task = self.db_manager.load_task('transaction_test')
        assert loaded_task.priority == 1


class TestResearchCacheManager:
    """Test cache management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        # Use memory cache (Redis not available in tests)
        self.cache_manager = ResearchCacheManager(redis_host='nonexistent')
        assert not self.cache_manager.redis_available
    
    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}
        
        # Set data
        success = self.cache_manager.set('test_key', test_data, ttl=60)
        assert success
        
        # Get data
        retrieved_data = self.cache_manager.get('test_key')
        assert retrieved_data == test_data
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        test_data = 'expiring_data'
        
        # Set with short TTL
        self.cache_manager.set('expiring_key', test_data, ttl=1)
        
        # Should be available immediately
        assert self.cache_manager.get('expiring_key') == test_data
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert self.cache_manager.get('expiring_key') is None
    
    def test_cache_delete(self):
        """Test cache deletion."""
        self.cache_manager.set('delete_test', 'delete_me', ttl=60)
        
        # Verify exists
        assert self.cache_manager.get('delete_test') == 'delete_me'
        
        # Delete
        success = self.cache_manager.delete('delete_test')
        assert success
        
        # Verify deleted
        assert self.cache_manager.get('delete_test') is None
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        initial_stats = self.cache_manager.get_stats()
        
        # Perform cache operations
        self.cache_manager.set('stats_test', 'data')
        self.cache_manager.get('stats_test')  # Hit
        self.cache_manager.get('nonexistent')  # Miss
        self.cache_manager.delete('stats_test')
        
        final_stats = self.cache_manager.get_stats()
        
        assert final_stats['operations']['sets'] > initial_stats['operations']['sets']
        assert final_stats['operations']['hits'] > initial_stats['operations']['hits']
        assert final_stats['operations']['misses'] > initial_stats['operations']['misses']
        assert final_stats['operations']['deletes'] > initial_stats['operations']['deletes']
    
    def test_cache_pattern_deletion(self):
        """Test pattern-based cache deletion."""
        # Set multiple keys
        self.cache_manager.set('pattern_test_1', 'data1')
        self.cache_manager.set('pattern_test_2', 'data2')
        self.cache_manager.set('other_key', 'data3')
        
        # Delete pattern
        deleted_count = self.cache_manager.clear_pattern('pattern_test_*')
        assert deleted_count == 2
        
        # Verify deletion
        assert self.cache_manager.get('pattern_test_1') is None
        assert self.cache_manager.get('pattern_test_2') is None
        assert self.cache_manager.get('other_key') == 'data3'
    
    def test_encryption_functionality(self):
        """Test data encryption in cache."""
        sensitive_data = {'password': 'secret123', 'api_key': 'abc123'}
        
        # Set encrypted data
        self.cache_manager.set('encrypted_test', sensitive_data)
        
        # Check that raw data in memory cache is encrypted
        cache_key = 'encrypted_test'
        if cache_key in self.cache_manager.memory_cache:
            raw_data, _ = self.cache_manager.memory_cache[cache_key]
            # Raw data should be bytes (encrypted)
            assert isinstance(raw_data, bytes)
        
        # Retrieved data should be decrypted
        retrieved_data = self.cache_manager.get('encrypted_test')
        assert retrieved_data == sensitive_data


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = ResearchDatabaseManager(self.temp_db.name)
        self.monitor = SystemMonitor(self.db_manager)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert not self.monitor.monitoring_active
        assert self.monitor.monitor_thread is None
        assert isinstance(self.monitor.alert_handlers, list)
        assert isinstance(self.monitor.thresholds, dict)
        
        # Check default thresholds
        assert 'cpu_usage' in self.monitor.thresholds
        assert 'memory_usage' in self.monitor.thresholds
        assert self.monitor.thresholds['cpu_usage'] > 0
    
    def test_alert_handler_registration(self):
        """Test alert handler registration."""
        alert_received = []
        
        def test_handler(level, message, data):
            alert_received.append((level, message, data))
        
        self.monitor.add_alert_handler(test_handler)
        assert len(self.monitor.alert_handlers) == 1
        
        # Test alert sending
        self.monitor._send_alert(AlertLevel.WARNING, "Test alert", {"test": True})
        
        assert len(alert_received) == 1
        assert alert_received[0][0] == AlertLevel.WARNING
        assert alert_received[0][1] == "Test alert"
        assert alert_received[0][2] == {"test": True}
    
    def test_metrics_collection(self):
        """Test system metrics collection."""
        metrics = self.monitor._collect_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert isinstance(metrics.timestamp, datetime)
        assert 0 <= metrics.cpu_usage <= 100
        assert 0 <= metrics.memory_usage <= 100
        assert 0 <= metrics.disk_usage <= 100
        assert metrics.system_load >= 0
        assert isinstance(metrics.network_io, dict)
        assert 'bytes_sent' in metrics.network_io
        assert 'bytes_recv' in metrics.network_io
    
    def test_threshold_checking(self):
        """Test threshold checking and alerting."""
        alerts_received = []
        
        def capture_alerts(level, message, data):
            alerts_received.append((level, message, data))
        
        self.monitor.add_alert_handler(capture_alerts)
        
        # Create metrics that exceed thresholds
        high_usage_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=95.0,  # Exceeds threshold
            memory_usage=90.0,  # Exceeds threshold
            disk_usage=95.0,  # Exceeds threshold
            gpu_usage=None,
            gpu_memory=None,
            network_io={},
            active_tasks=0,
            queued_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            system_load=5.0,  # Exceeds threshold
            research_throughput=0.0
        )
        
        self.monitor._check_thresholds(high_usage_metrics)
        
        # Should have generated multiple alerts
        assert len(alerts_received) > 0
        
        # Check alert content
        alert_messages = [alert[1] for alert in alerts_received]
        assert any('CPU' in msg for msg in alert_messages)
        assert any('memory' in msg for msg in alert_messages)
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop."""
        # Start monitoring
        self.monitor.start_monitoring(interval=1)
        
        assert self.monitor.monitoring_active
        assert self.monitor.monitor_thread is not None
        assert self.monitor.monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        assert not self.monitor.monitoring_active
        
        # Thread should stop
        time.sleep(1)
        assert not self.monitor.monitor_thread.is_alive()


class TestFaultTolerantTaskExecutor:
    """Test fault-tolerant task execution."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = ResearchDatabaseManager(self.temp_db.name)
        self.cache_manager = ResearchCacheManager(redis_host='nonexistent')
        self.executor = FaultTolerantTaskExecutor(self.db_manager, self.cache_manager, max_workers=2)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if hasattr(self.executor, 'worker_threads'):
            self.executor.stop()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        assert self.executor.db_manager == self.db_manager
        assert self.executor.cache_manager == self.cache_manager
        assert self.executor.max_workers == 2
        assert isinstance(self.executor.active_tasks, dict)
    
    def test_task_submission(self):
        """Test task submission."""
        task = ResearchTask(
            task_id='submit_test',
            task_type='test_task',
            phase=ResearchPhase.INITIALIZATION,
            priority=5,
            payload={'test': 'submission'},
            created_at=datetime.now()
        )
        
        # Submit task
        task_id = self.executor.submit_task(task)
        
        assert task_id == 'submit_test'
        
        # Verify task was saved to database
        loaded_task = self.db_manager.load_task('submit_test')
        assert loaded_task is not None
        assert loaded_task.task_id == 'submit_test'
    
    def test_dependency_checking(self):
        """Test dependency checking."""
        # Create dependency task
        dep_task = ResearchTask(
            task_id='dependency',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now()
        )
        self.db_manager.save_task(dep_task)
        
        # Create dependent task
        dependent_task = ResearchTask(
            task_id='dependent',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            dependencies=['dependency']
        )
        
        # Dependencies not met (dependency not completed)
        assert not self.executor._check_dependencies(dependent_task)
        
        # Complete dependency
        dep_task.completed_at = datetime.now()
        self.db_manager.save_task(dep_task)
        
        # Dependencies should now be met
        assert self.executor._check_dependencies(dependent_task)
    
    def test_task_phases(self):
        """Test task phase identification."""
        hypothesis_task = ResearchTask(
            task_id='hyp_task',
            task_type='hypothesis_generation',
            phase=ResearchPhase.HYPOTHESIS_GENERATION,
            priority=1,
            payload={},
            created_at=datetime.now()
        )
        
        phases = self.executor._get_task_phases(hypothesis_task)
        
        assert isinstance(phases, list)
        assert len(phases) > 0
        assert 'init' in phases
        assert 'finalize' in phases
        
        # Check specific phases for hypothesis generation
        expected_phases = ['init', 'knowledge_base_load', 'generate', 'validate', 'finalize']
        assert phases == expected_phases
    
    def test_phase_execution(self):
        """Test individual phase execution."""
        task = ResearchTask(
            task_id='phase_test',
            task_type='algorithm_discovery',
            phase=ResearchPhase.ALGORITHM_DISCOVERY,
            priority=1,
            payload={'test': 'phase'},
            created_at=datetime.now()
        )
        
        # Execute a phase
        phase_result = self.executor._execute_phase(task, 'generate', {})
        
        assert isinstance(phase_result, dict)
        assert 'generate_completed' in phase_result
        assert 'generate_execution_time' in phase_result
        assert 'generate_timestamp' in phase_result
        assert phase_result['generate_completed'] is True
        assert phase_result['generate_execution_time'] > 0


class TestSecurityManager:
    """Test security management functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_manager = ResearchDatabaseManager(self.temp_db.name)
        self.security_manager = SecurityManager(self.db_manager)
    
    def teardown_method(self):
        """Cleanup test environment."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        assert self.security_manager.security_enabled
        assert isinstance(self.security_manager.failed_attempts, dict)
        assert isinstance(self.security_manager.blocked_ips, set)
        assert self.security_manager.max_failed_attempts > 0
        assert self.security_manager.block_duration > 0
    
    def test_task_security_validation_safe(self):
        """Test task security validation with safe content."""
        safe_task = ResearchTask(
            task_id='safe_task',
            task_type='hypothesis_generation',
            phase=ResearchPhase.HYPOTHESIS_GENERATION,
            priority=1,
            payload={'research_domain': 'biology', 'parameters': {'learning_rate': 0.01}},
            created_at=datetime.now()
        )
        
        # Should pass validation
        is_valid = self.security_manager.validate_task_security(safe_task)
        assert is_valid
    
    def test_task_security_validation_suspicious(self):
        """Test task security validation with suspicious content."""
        suspicious_task = ResearchTask(
            task_id='suspicious_task',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={'command': 'rm -rf /', 'script': 'import os; os.system("format c:")'},
            created_at=datetime.now()
        )
        
        # Should fail validation
        is_valid = self.security_manager.validate_task_security(suspicious_task)
        assert not is_valid
    
    def test_task_security_validation_oversized(self):
        """Test task security validation with oversized payload."""
        large_payload = {'data': 'x' * (2 * 1024 * 1024)}  # 2MB payload
        
        oversized_task = ResearchTask(
            task_id='oversized_task',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload=large_payload,
            created_at=datetime.now()
        )
        
        # Should fail validation due to size
        is_valid = self.security_manager.validate_task_security(oversized_task)
        assert not is_valid
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        test_ip = '192.168.1.1'
        
        # Initially should not be blocked
        assert self.security_manager.check_rate_limiting(test_ip)
        
        # Record multiple failed attempts
        for _ in range(self.security_manager.max_failed_attempts):
            self.security_manager.record_failed_attempt(test_ip, 'test_failure')
        
        # Should now be blocked
        assert not self.security_manager.check_rate_limiting(test_ip)
        assert test_ip in self.security_manager.blocked_ips
    
    def test_security_event_logging(self):
        """Test security event logging."""
        # Trigger a security event
        suspicious_task = ResearchTask(
            task_id='log_test',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={'dangerous': 'eval("malicious_code")'},
            created_at=datetime.now()
        )
        
        self.security_manager.validate_task_security(suspicious_task)
        
        # Check that security event was logged
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events")
            count = cursor.fetchone()[0]
            assert count > 0
            
            cursor.execute("SELECT event_type, severity FROM security_events LIMIT 1")
            row = cursor.fetchone()
            assert row[0] == 'suspicious_pattern'
            assert row[1] == 'warning'


class TestResearchInfrastructure:
    """Test main research infrastructure orchestrator."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'database': {'path': ':memory:', 'backup_interval': 3600},
            'cache': {'redis_host': 'nonexistent', 'redis_port': 6379},
            'execution': {'max_workers': 2, 'task_timeout': 60},
            'monitoring': {'interval': 1, 'alert_thresholds': {'cpu_usage': 80.0}},
            'security': {'enabled': True, 'max_failed_attempts': 3}
        }
        self.infrastructure = ResearchInfrastructure(self.config)
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.infrastructure.running:
            asyncio.run(self.infrastructure.stop())
    
    def test_infrastructure_initialization(self):
        """Test infrastructure initialization."""
        assert self.infrastructure.config == self.config
        assert self.infrastructure.db_manager is not None
        assert self.infrastructure.cache_manager is not None
        assert self.infrastructure.monitor is not None
        assert self.infrastructure.task_executor is not None
        assert self.infrastructure.security_manager is not None
        
        assert not self.infrastructure.running
        assert self.infrastructure.start_time is None
    
    @pytest.mark.asyncio
    async def test_infrastructure_start_stop(self):
        """Test infrastructure start and stop."""
        # Start infrastructure
        await self.infrastructure.start()
        
        assert self.infrastructure.running
        assert self.infrastructure.start_time is not None
        assert self.infrastructure.monitor.monitoring_active
        
        # Stop infrastructure
        await self.infrastructure.stop()
        
        assert not self.infrastructure.running
        assert not self.infrastructure.monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test research task submission."""
        await self.infrastructure.start()
        
        try:
            task_id = self.infrastructure.submit_research_task(
                task_type='hypothesis_generation',
                payload={'domain': 'test', 'parameters': {}},
                priority=5,
                dependencies=[]
            )
            
            assert isinstance(task_id, str)
            assert task_id.startswith('task_')
            
            # Verify task was saved
            task_status = self.infrastructure.get_task_status(task_id)
            assert task_status is not None
            assert task_status['task_id'] == task_id
            assert task_status['task_type'] == 'hypothesis_generation'
            
        finally:
            await self.infrastructure.stop()
    
    @pytest.mark.asyncio
    async def test_task_submission_security_validation(self):
        """Test task submission with security validation."""
        await self.infrastructure.start()
        
        try:
            # Should reject suspicious task
            with pytest.raises(ValueError, match="failed security validation"):
                self.infrastructure.submit_research_task(
                    task_type='malicious',
                    payload={'command': 'rm -rf /', 'script': 'os.system("bad")'},
                    priority=1
                )
        
        finally:
            await self.infrastructure.stop()
    
    def test_task_status_retrieval(self):
        """Test task status retrieval."""
        # Create and save a task directly
        task = ResearchTask(
            task_id='status_test',
            task_type='algorithm_discovery',
            phase=ResearchPhase.ALGORITHM_DISCOVERY,
            priority=3,
            payload={'test': 'status'},
            created_at=datetime.now(),
            started_at=datetime.now(),
            checkpoints=[{'phase': 'init', 'timestamp': datetime.now().isoformat()}]
        )
        
        self.infrastructure.db_manager.save_task(task)
        
        # Retrieve status
        status = self.infrastructure.get_task_status('status_test')
        
        assert status is not None
        assert status['task_id'] == 'status_test'
        assert status['task_type'] == 'algorithm_discovery'
        assert status['status'] == 'running'  # Has started_at but not completed_at
        assert status['checkpoints'] == 1
        assert status['retry_count'] == 0
        assert 'progress' in status
    
    def test_nonexistent_task_status(self):
        """Test status retrieval for nonexistent task."""
        status = self.infrastructure.get_task_status('nonexistent')
        assert status is None
    
    def test_system_status_retrieval(self):
        """Test system status retrieval."""
        status = self.infrastructure.get_system_status()
        
        assert 'infrastructure' in status
        assert 'system_metrics' in status
        assert 'cache_stats' in status
        assert 'task_stats' in status
        assert 'components' in status
        
        # Check infrastructure status
        infra_status = status['infrastructure']
        assert 'running' in infra_status
        assert 'start_time' in infra_status
        assert 'uptime' in infra_status
        
        # Check components status
        components = status['components']
        assert 'database' in components
        assert 'cache' in components
        assert 'monitoring' in components
        assert 'task_executor' in components
        assert 'security' in components
    
    def test_task_phase_mapping(self):
        """Test task type to phase mapping."""
        assert self.infrastructure._get_task_phase('hypothesis_generation') == ResearchPhase.HYPOTHESIS_GENERATION
        assert self.infrastructure._get_task_phase('algorithm_discovery') == ResearchPhase.ALGORITHM_DISCOVERY
        assert self.infrastructure._get_task_phase('experimental_validation') == ResearchPhase.EXPERIMENTAL_VALIDATION
        assert self.infrastructure._get_task_phase('statistical_analysis') == ResearchPhase.STATISTICAL_ANALYSIS
        assert self.infrastructure._get_task_phase('unknown_type') == ResearchPhase.INITIALIZATION
    
    def test_task_status_string_mapping(self):
        """Test task status string mapping."""
        # Completed task
        completed_task = ResearchTask(
            task_id='completed',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        assert self.infrastructure._get_task_status_string(completed_task) == 'completed'
        
        # Failed task
        failed_task = ResearchTask(
            task_id='failed',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            failed_at=datetime.now(),
            retry_count=5,
            max_retries=3
        )
        assert self.infrastructure._get_task_status_string(failed_task) == 'failed'
        
        # Running task
        running_task = ResearchTask(
            task_id='running',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            started_at=datetime.now()
        )
        assert self.infrastructure._get_task_status_string(running_task) == 'running'
        
        # Queued task
        queued_task = ResearchTask(
            task_id='queued',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now()
        )
        assert self.infrastructure._get_task_status_string(queued_task) == 'queued'
    
    def test_progress_calculation(self):
        """Test task progress calculation."""
        # Completed task
        completed_task = ResearchTask(
            task_id='completed',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        assert self.infrastructure._calculate_progress(completed_task) == 100.0
        
        # Failed task
        failed_task = ResearchTask(
            task_id='failed',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            failed_at=datetime.now(),
            retry_count=5,
            max_retries=3
        )
        assert self.infrastructure._calculate_progress(failed_task) == 0.0
        
        # Queued task
        queued_task = ResearchTask(
            task_id='queued',
            task_type='test',
            phase=ResearchPhase.INITIALIZATION,
            priority=1,
            payload={},
            created_at=datetime.now()
        )
        assert self.infrastructure._calculate_progress(queued_task) == 0.0
        
        # Running task with checkpoints
        running_task = ResearchTask(
            task_id='running',
            task_type='hypothesis_generation',
            phase=ResearchPhase.HYPOTHESIS_GENERATION,
            priority=1,
            payload={},
            created_at=datetime.now(),
            started_at=datetime.now(),
            checkpoints=[
                {'phase': 'init', 'timestamp': datetime.now().isoformat()},
                {'phase': 'generate', 'timestamp': datetime.now().isoformat()}
            ]
        )
        progress = self.infrastructure._calculate_progress(running_task)
        assert 0 < progress < 100


class TestIntegrationAndPerformance:
    """Integration and performance tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_execution(self):
        """Test end-to-end task execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'database': {'path': f'{temp_dir}/test.db'},
                'cache': {'redis_host': 'nonexistent'},
                'execution': {'max_workers': 1},
                'monitoring': {'interval': 10},
                'security': {'enabled': True}
            }
            
            infrastructure = ResearchInfrastructure(config)
            
            try:
                await infrastructure.start()
                
                # Submit a task
                task_id = infrastructure.submit_research_task(
                    task_type='hypothesis_generation',
                    payload={'domain': 'integration_test', 'num_hypotheses': 3},
                    priority=1
                )
                
                # Wait briefly for task processing
                await asyncio.sleep(2)
                
                # Check task status
                status = infrastructure.get_task_status(task_id)
                assert status is not None
                assert status['task_id'] == task_id
                
                # Check system status
                system_status = infrastructure.get_system_status()
                assert system_status['infrastructure']['running']
                
            finally:
                await infrastructure.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test concurrent task execution."""
        config = {
            'database': {'path': ':memory:'},
            'cache': {'redis_host': 'nonexistent'},
            'execution': {'max_workers': 3},
            'monitoring': {'interval': 10},
            'security': {'enabled': True}
        }
        
        infrastructure = ResearchInfrastructure(config)
        
        try:
            await infrastructure.start()
            
            # Submit multiple tasks
            task_ids = []
            for i in range(5):
                task_id = infrastructure.submit_research_task(
                    task_type='algorithm_discovery',
                    payload={'algorithm_type': f'test_algo_{i}'},
                    priority=i + 1
                )
                task_ids.append(task_id)
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Check all tasks were submitted
            for task_id in task_ids:
                status = infrastructure.get_task_status(task_id)
                assert status is not None
                assert status['task_id'] == task_id
        
        finally:
            await infrastructure.stop()
    
    def test_memory_efficiency(self):
        """Test memory efficiency of infrastructure."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create multiple infrastructure instances
        infrastructures = []
        for i in range(3):
            config = {
                'database': {'path': ':memory:'},
                'cache': {'redis_host': 'nonexistent'},
                'execution': {'max_workers': 1},
                'monitoring': {'interval': 60},
                'security': {'enabled': False}
            }
            infra = ResearchInfrastructure(config)
            infrastructures.append(infra)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 100MB per instance)
        memory_per_instance = peak / len(infrastructures)
        assert memory_per_instance < 100 * 1024 * 1024  # 100MB


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness."""
    
    def test_invalid_database_path(self):
        """Test handling of invalid database path."""
        # Should handle gracefully and not crash
        try:
            db_manager = ResearchDatabaseManager('/invalid/path/database.db')
            # May succeed with some SQLite configurations
        except Exception:
            # Should fail gracefully, not crash
            pass
    
    def test_infrastructure_double_start(self):
        """Test double start of infrastructure."""
        config = {'database': {'path': ':memory:'}}
        infrastructure = ResearchInfrastructure(config)
        
        async def test_double_start():
            await infrastructure.start()
            
            # Second start should be handled gracefully
            await infrastructure.start()  # Should not cause error
            
            assert infrastructure.running
            
            await infrastructure.stop()
        
        asyncio.run(test_double_start())
    
    def test_task_submission_without_start(self):
        """Test task submission without starting infrastructure."""
        infrastructure = ResearchInfrastructure()
        
        with pytest.raises(RuntimeError, match="Infrastructure not running"):
            infrastructure.submit_research_task('test', {})
    
    def test_cache_corruption_handling(self):
        """Test handling of cache corruption."""
        cache_manager = ResearchCacheManager(redis_host='nonexistent')
        
        # Manually corrupt cache data
        cache_manager.memory_cache['corrupted'] = (b'invalid_encrypted_data', time.time() + 3600)
        
        # Should handle corruption gracefully
        result = cache_manager.get('corrupted')
        assert result is None  # Should return None instead of crashing


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])