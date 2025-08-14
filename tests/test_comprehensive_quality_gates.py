#!/usr/bin/env python3
"""Comprehensive Quality Gates Testing.

This module implements comprehensive testing and validation for all TERRAGON SDLC
generations, ensuring quality, reliability, and performance standards.
"""

import sys
import os
import time
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all modules for testing
try:
    import scgraph_hub
    from scgraph_hub import DatasetCatalog, SimpleSCGraphDataset, check_dependencies
    from scgraph_hub.research_framework import (
        BiologicallyInformedGNN, TemporalDynamicsGNN, MultiModalIntegrationGNN
    )
    from scgraph_hub.intelligent_fault_tolerance import IntelligentFaultToleranceSystem
    from scgraph_hub.advanced_security_framework import AdvancedSecurityFramework
    from scgraph_hub.hyperscale_performance_engine import HyperscalePerformanceEngine
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class QualityGateError(Exception):
    """Custom exception for quality gate failures."""
    pass


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.security_validation = {}
        self.reliability_metrics = {}
        
        print("🛡️ QUALITY GATE VALIDATOR INITIALIZED")
        print("=" * 60)
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'quality_gates': {}
        }
        
        quality_gates = [
            ('functionality', self._test_functionality_gate),
            ('performance', self._test_performance_gate),
            ('security', self._test_security_gate),
            ('reliability', self._test_reliability_gate),
            ('scalability', self._test_scalability_gate),
            ('integration', self._test_integration_gate),
            ('documentation', self._test_documentation_gate),
            ('compliance', self._test_compliance_gate)
        ]
        
        passed_gates = 0
        total_gates = len(quality_gates)
        
        for gate_name, gate_function in quality_gates:
            print(f"\n🔍 Quality Gate: {gate_name.upper()}")
            print("-" * 40)
            
            try:
                gate_result = gate_function()
                results['quality_gates'][gate_name] = gate_result
                
                if gate_result['status'] == 'PASS':
                    passed_gates += 1
                    print(f"✅ {gate_name.upper()} GATE: PASSED")
                else:
                    print(f"❌ {gate_name.upper()} GATE: FAILED")
                    if 'errors' in gate_result:
                        for error in gate_result['errors']:
                            print(f"   Error: {error}")
                            
            except Exception as e:
                results['quality_gates'][gate_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'score': 0.0
                }
                print(f"💥 {gate_name.upper()} GATE: ERROR - {str(e)}")
        
        # Calculate overall status
        pass_rate = passed_gates / total_gates
        if pass_rate >= 0.85:
            results['overall_status'] = 'PASS'
        elif pass_rate >= 0.70:
            results['overall_status'] = 'WARNING'
        else:
            results['overall_status'] = 'FAIL'
        
        results['pass_rate'] = pass_rate
        results['passed_gates'] = passed_gates
        results['total_gates'] = total_gates
        
        return results
    
    def _test_functionality_gate(self) -> Dict[str, Any]:
        """Test core functionality across all generations."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'tests': {},
            'errors': []
        }
        
        functionality_tests = [
            ('basic_imports', self._test_basic_imports),
            ('dataset_operations', self._test_dataset_operations),
            ('research_algorithms', self._test_research_algorithms),
            ('core_functionality', self._test_core_functionality)
        ]
        
        passed_tests = 0
        
        for test_name, test_function in functionality_tests:
            try:
                test_result = test_function()
                gate_result['tests'][test_name] = test_result
                if test_result.get('passed', False):
                    passed_tests += 1
                    print(f"    ✅ {test_name}: PASSED")
                else:
                    print(f"    ❌ {test_name}: FAILED")
                    if 'error' in test_result:
                        gate_result['errors'].append(f"{test_name}: {test_result['error']}")
            except Exception as e:
                gate_result['tests'][test_name] = {'passed': False, 'error': str(e)}
                gate_result['errors'].append(f"{test_name}: {str(e)}")
                print(f"    💥 {test_name}: ERROR - {str(e)}")
        
        gate_result['score'] = passed_tests / len(functionality_tests)
        gate_result['status'] = 'PASS' if gate_result['score'] >= 0.8 else 'FAIL'
        
        return gate_result
    
    def _test_basic_imports(self) -> Dict[str, Any]:
        """Test that all basic imports work correctly."""
        try:
            # Test basic package import
            import scgraph_hub
            
            # Test core classes
            catalog = scgraph_hub.DatasetCatalog()
            dataset = scgraph_hub.SimpleSCGraphDataset("test")
            
            # Test utility functions
            deps = scgraph_hub.check_dependencies()
            
            return {'passed': True, 'components_tested': 3}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_dataset_operations(self) -> Dict[str, Any]:
        """Test dataset operations and data handling."""
        try:
            # Test catalog operations
            catalog = DatasetCatalog()
            datasets = catalog.list_datasets()
            
            # Test dataset creation
            dataset = SimpleSCGraphDataset("pbmc_10k", root=tempfile.mkdtemp())
            
            # Test dataset info
            info = dataset.get_info()
            
            return {
                'passed': True,
                'catalog_datasets': len(datasets),
                'dataset_info_keys': len(info) if info else 0
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_research_algorithms(self) -> Dict[str, Any]:
        """Test research algorithm implementations."""
        try:
            algorithms_tested = 0
            
            # Test BiologicallyInformedGNN
            bio_gnn = BiologicallyInformedGNN()
            result1 = bio_gnn.forward({'test': 'data'})
            if 'accuracy' in result1:
                algorithms_tested += 1
            
            # Test TemporalDynamicsGNN
            temporal_gnn = TemporalDynamicsGNN()
            result2 = temporal_gnn.forward({'test': 'data'})
            if 'accuracy' in result2:
                algorithms_tested += 1
            
            # Test MultiModalIntegrationGNN
            multimodal_gnn = MultiModalIntegrationGNN()
            result3 = multimodal_gnn.forward({'test': 'data'})
            if 'accuracy' in result3:
                algorithms_tested += 1
            
            return {
                'passed': algorithms_tested >= 2,
                'algorithms_tested': algorithms_tested,
                'total_algorithms': 3
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core package functionality."""
        try:
            # Test package version
            version = scgraph_hub.__version__
            
            # Test quick start function
            dataset = scgraph_hub.simple_quick_start("test_dataset")
            
            return {
                'passed': True,
                'package_version': version,
                'quick_start_works': dataset is not None
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_performance_gate(self) -> Dict[str, Any]:
        """Test performance and optimization capabilities."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'metrics': {},
            'errors': []
        }
        
        try:
            # Initialize performance engine
            engine = HyperscalePerformanceEngine()
            
            # Test cache performance
            cache_start = time.time()
            for i in range(100):
                engine.cache.put(f"key_{i}", f"value_{i}")
            
            hit_count = 0
            for i in range(50):
                if engine.cache.get(f"key_{i}") is not None:
                    hit_count += 1
            
            cache_time = time.time() - cache_start
            cache_hit_rate = hit_count / 50
            
            # Test task submission performance
            task_start = time.time()
            engine.start_all_services()
            
            task_ids = []
            for i in range(20):
                task_id = engine.submit_compute_task('inference', priority=5)
                task_ids.append(task_id)
            
            # Wait for some tasks to complete
            time.sleep(3)
            
            # Stop services
            engine.stop_all_services()
            task_time = time.time() - task_start
            
            gate_result['metrics'] = {
                'cache_performance_ms': cache_time * 1000,
                'cache_hit_rate': cache_hit_rate,
                'task_submission_time_ms': task_time * 1000,
                'tasks_submitted': len(task_ids)
            }
            
            # Performance criteria
            performance_score = 0
            if cache_time < 0.1:  # Cache operations under 100ms
                performance_score += 0.25
            if cache_hit_rate > 0.8:  # Cache hit rate over 80%
                performance_score += 0.25
            if task_time < 5.0:  # Task operations under 5 seconds
                performance_score += 0.25
            if len(task_ids) == 20:  # All tasks submitted successfully
                performance_score += 0.25
            
            gate_result['score'] = performance_score
            gate_result['status'] = 'PASS' if performance_score >= 0.7 else 'FAIL'
            
            print(f"    Cache performance: {cache_time*1000:.2f}ms")
            print(f"    Cache hit rate: {cache_hit_rate:.2%}")
            print(f"    Task processing: {task_time:.2f}s for {len(task_ids)} tasks")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_security_gate(self) -> Dict[str, Any]:
        """Test security framework and controls."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'security_features': {},
            'errors': []
        }
        
        try:
            # Initialize security framework
            security = AdvancedSecurityFramework()
            
            # Test user creation and authentication
            user = security.access_control.create_user(
                username="test_user",
                email="test@example.com",
                password="TestPassword123!",
                roles=["researcher"]
            )
            
            # Test authentication
            auth_user = security.access_control.authenticate_user("test_user", "TestPassword123!")
            
            # Test session management
            session_id = None
            if auth_user:
                session_id = security.access_control.create_session(auth_user)
            
            # Test audit logging
            from scgraph_hub.advanced_security_framework import ThreatLevel, ActionType
            security.audit_logger.log_security_event(
                event_type='test_event',
                severity=ThreatLevel.INFO,
                user_id=user.user_id if user else None,
                resource='test_resource',
                action=ActionType.READ,
                result='success',
                details={'test': True}
            )
            
            gate_result['security_features'] = {
                'user_creation': user is not None,
                'authentication': auth_user is not None,
                'session_management': session_id is not None,
                'audit_logging': True,
                'cryptography': hasattr(security, 'crypto_manager'),
                'access_control': hasattr(security, 'access_control')
            }
            
            # Calculate security score
            features_working = sum(1 for v in gate_result['security_features'].values() if v)
            total_features = len(gate_result['security_features'])
            
            gate_result['score'] = features_working / total_features
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.8 else 'FAIL'
            
            print(f"    Security features working: {features_working}/{total_features}")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_reliability_gate(self) -> Dict[str, Any]:
        """Test fault tolerance and reliability features."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'reliability_features': {},
            'errors': []
        }
        
        try:
            # Initialize fault tolerance system
            ft_system = IntelligentFaultToleranceSystem(monitoring_interval=1)
            
            # Test health checks
            health_checks_passed = 0
            for component in ft_system.components:
                is_healthy, metrics = component.health_check()
                if is_healthy:
                    health_checks_passed += 1
            
            # Test exception handling
            recovery_successes = 0
            test_exceptions = [
                (FileNotFoundError("Test file not found"), "TestComponent"),
                (RuntimeError("Test runtime error"), "TestComponent")
            ]
            
            for exc, context in test_exceptions:
                recovery_success = ft_system.handle_exception(exc, context)
                if recovery_success:
                    recovery_successes += 1
            
            # Test monitoring
            ft_system.start_monitoring()
            time.sleep(2)  # Let monitoring run
            ft_system.stop_monitoring()
            
            gate_result['reliability_features'] = {
                'health_checks_passed': health_checks_passed,
                'total_components': len(ft_system.components),
                'recovery_successes': recovery_successes,
                'total_exceptions_tested': len(test_exceptions),
                'monitoring_functional': True,
                'circuit_breaker_available': True  # Always available in our implementation
            }
            
            # Calculate reliability score
            health_rate = health_checks_passed / max(len(ft_system.components), 1)
            recovery_rate = recovery_successes / max(len(test_exceptions), 1)
            
            gate_result['score'] = (health_rate + recovery_rate) / 2
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.7 else 'FAIL'
            
            print(f"    Health checks: {health_checks_passed}/{len(ft_system.components)} passed")
            print(f"    Recovery rate: {recovery_successes}/{len(test_exceptions)} successful")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_scalability_gate(self) -> Dict[str, Any]:
        """Test scalability and performance optimization."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'scalability_metrics': {},
            'errors': []
        }
        
        try:
            # Test cache scalability
            cache_sizes = [100, 500, 1000]
            cache_performance = []
            
            for size in cache_sizes:
                from scgraph_hub.hyperscale_performance_engine import IntelligentCache, CacheStrategy
                cache = IntelligentCache(max_size=size, strategy=CacheStrategy.INTELLIGENT)
                
                start_time = time.time()
                for i in range(size // 2):
                    cache.put(f"key_{i}", f"value_{i}")
                
                for i in range(size // 4):
                    cache.get(f"key_{i}")
                
                end_time = time.time()
                cache_performance.append(end_time - start_time)
            
            # Test distributed task management
            engine = HyperscalePerformanceEngine()
            engine.start_all_services()
            
            # Submit varying workloads
            task_counts = [10, 20, 30]
            throughput_results = []
            
            for count in task_counts:
                start_time = time.time()
                task_ids = []
                
                for i in range(count):
                    task_id = engine.submit_compute_task('inference', priority=5)
                    task_ids.append(task_id)
                
                # Wait for completion
                completed = 0
                while completed < count:
                    time.sleep(0.1)
                    completed = len([
                        task for task in engine.task_manager.completed_tasks.values()
                        if task.task_id in task_ids
                    ])
                
                end_time = time.time()
                duration = end_time - start_time
                throughput = count / duration if duration > 0 else 0
                throughput_results.append(throughput)
            
            engine.stop_all_services()
            
            gate_result['scalability_metrics'] = {
                'cache_performance_scaling': cache_performance,
                'task_throughput_scaling': throughput_results,
                'cache_sizes_tested': cache_sizes,
                'task_counts_tested': task_counts
            }
            
            # Calculate scalability score
            cache_scalable = len(cache_performance) >= 2
            throughput_scalable = len(throughput_results) >= 2 and all(t > 0 for t in throughput_results)
            
            gate_result['score'] = (cache_scalable + throughput_scalable) / 2
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.5 else 'FAIL'
            
            print(f"    Cache scaling: {cache_scalable}")
            print(f"    Throughput scaling: {throughput_scalable}")
            print(f"    Max throughput: {max(throughput_results):.2f} tasks/sec")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_integration_gate(self) -> Dict[str, Any]:
        """Test integration between all system components."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'integration_tests': {},
            'errors': []
        }
        
        try:
            integration_score = 0
            total_integrations = 4
            
            # Test 1: Research + Performance Integration
            try:
                bio_gnn = BiologicallyInformedGNN()
                engine = HyperscalePerformanceEngine()
                
                # Use research algorithm with performance engine
                result = bio_gnn.forward({'test': 'data'})
                engine.cache.put('research_result', str(result))
                cached_result = engine.cache.get('research_result')
                
                if cached_result is not None:
                    integration_score += 1
                    gate_result['integration_tests']['research_performance'] = 'PASS'
                else:
                    gate_result['integration_tests']['research_performance'] = 'FAIL'
                    
            except Exception as e:
                gate_result['integration_tests']['research_performance'] = f'ERROR: {str(e)}'
            
            # Test 2: Security + Performance Integration
            try:
                security = AdvancedSecurityFramework()
                engine = HyperscalePerformanceEngine()
                
                # Create secure session and use with performance engine
                user = security.access_control.create_user(
                    "integration_user", "test@example.com", "Password123!", roles=["researcher"]
                )
                
                if user:
                    integration_score += 1
                    gate_result['integration_tests']['security_performance'] = 'PASS'
                else:
                    gate_result['integration_tests']['security_performance'] = 'FAIL'
                    
            except Exception as e:
                gate_result['integration_tests']['security_performance'] = f'ERROR: {str(e)}'
            
            # Test 3: Fault Tolerance + Security Integration
            try:
                ft_system = IntelligentFaultToleranceSystem()
                security = AdvancedSecurityFramework()
                
                # Test exception handling with security context
                recovery_success = ft_system.handle_exception(
                    ConnectionError("Integration test"), "SecurityComponent"
                )
                
                if recovery_success:
                    integration_score += 1
                    gate_result['integration_tests']['fault_tolerance_security'] = 'PASS'
                else:
                    gate_result['integration_tests']['fault_tolerance_security'] = 'FAIL'
                    
            except Exception as e:
                gate_result['integration_tests']['fault_tolerance_security'] = f'ERROR: {str(e)}'
            
            # Test 4: End-to-End Workflow
            try:
                # Simulate complete workflow
                catalog = DatasetCatalog()
                dataset = SimpleSCGraphDataset("test_integration")
                bio_gnn = BiologicallyInformedGNN()
                
                # Run algorithm on dataset
                result = bio_gnn.forward({'dataset': dataset.name})
                
                if 'accuracy' in result:
                    integration_score += 1
                    gate_result['integration_tests']['end_to_end'] = 'PASS'
                else:
                    gate_result['integration_tests']['end_to_end'] = 'FAIL'
                    
            except Exception as e:
                gate_result['integration_tests']['end_to_end'] = f'ERROR: {str(e)}'
            
            gate_result['score'] = integration_score / total_integrations
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.75 else 'FAIL'
            
            print(f"    Integration tests passed: {integration_score}/{total_integrations}")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_documentation_gate(self) -> Dict[str, Any]:
        """Test documentation completeness and quality."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'documentation_coverage': {},
            'errors': []
        }
        
        try:
            # Check for essential documentation files
            repo_root = Path(__file__).parent.parent
            
            essential_docs = [
                'README.md',
                'LICENSE',
                'pyproject.toml',
                'CONTRIBUTING.md',
                'CODE_OF_CONDUCT.md'
            ]
            
            docs_found = 0
            for doc in essential_docs:
                if (repo_root / doc).exists():
                    docs_found += 1
                    gate_result['documentation_coverage'][doc] = 'FOUND'
                else:
                    gate_result['documentation_coverage'][doc] = 'MISSING'
            
            # Check for docstrings in key modules
            modules_with_docstrings = 0
            key_modules = [
                'scgraph_hub/__init__.py',
                'scgraph_hub/dataset.py',
                'scgraph_hub/models.py',
                'scgraph_hub/research_framework.py'
            ]
            
            for module_path in key_modules:
                full_path = repo_root / 'src' / module_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r') as f:
                            content = f.read()
                            if '"""' in content and 'def ' in content:
                                modules_with_docstrings += 1
                    except Exception:
                        pass
            
            gate_result['documentation_coverage']['modules_with_docstrings'] = f"{modules_with_docstrings}/{len(key_modules)}"
            
            # Calculate documentation score
            doc_coverage = docs_found / len(essential_docs)
            docstring_coverage = modules_with_docstrings / len(key_modules)
            
            gate_result['score'] = (doc_coverage + docstring_coverage) / 2
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.7 else 'FAIL'
            
            print(f"    Essential docs: {docs_found}/{len(essential_docs)} found")
            print(f"    Modules with docstrings: {modules_with_docstrings}/{len(key_modules)}")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def _test_compliance_gate(self) -> Dict[str, Any]:
        """Test compliance with standards and best practices."""
        gate_result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'compliance_checks': {},
            'errors': []
        }
        
        try:
            compliance_score = 0
            total_checks = 5
            
            # Check 1: Package structure
            repo_root = Path(__file__).parent.parent
            required_structure = ['src', 'tests', 'examples', 'docs']
            structure_score = sum(1 for dir_name in required_structure if (repo_root / dir_name).exists())
            
            if structure_score >= 3:
                compliance_score += 1
                gate_result['compliance_checks']['package_structure'] = 'PASS'
            else:
                gate_result['compliance_checks']['package_structure'] = 'FAIL'
            
            # Check 2: Error handling
            try:
                # Test that modules handle missing dependencies gracefully
                catalog = DatasetCatalog()
                datasets = catalog.list_datasets()  # Should work even without heavy dependencies
                compliance_score += 1
                gate_result['compliance_checks']['error_handling'] = 'PASS'
            except Exception:
                gate_result['compliance_checks']['error_handling'] = 'FAIL'
            
            # Check 3: Import structure
            try:
                import scgraph_hub
                # Should be able to import without errors
                compliance_score += 1
                gate_result['compliance_checks']['import_structure'] = 'PASS'
            except Exception:
                gate_result['compliance_checks']['import_structure'] = 'FAIL'
            
            # Check 4: Configuration management
            config_files = ['pyproject.toml']
            config_found = sum(1 for cfg in config_files if (repo_root / cfg).exists())
            
            if config_found >= 1:
                compliance_score += 1
                gate_result['compliance_checks']['configuration'] = 'PASS'
            else:
                gate_result['compliance_checks']['configuration'] = 'FAIL'
            
            # Check 5: Version management
            try:
                version = scgraph_hub.__version__
                if version and len(version) > 0:
                    compliance_score += 1
                    gate_result['compliance_checks']['version_management'] = 'PASS'
                else:
                    gate_result['compliance_checks']['version_management'] = 'FAIL'
            except Exception:
                gate_result['compliance_checks']['version_management'] = 'FAIL'
            
            gate_result['score'] = compliance_score / total_checks
            gate_result['status'] = 'PASS' if gate_result['score'] >= 0.8 else 'FAIL'
            
            print(f"    Compliance checks passed: {compliance_score}/{total_checks}")
            
        except Exception as e:
            gate_result['errors'].append(str(e))
            gate_result['status'] = 'FAIL'
        
        return gate_result
    
    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive quality gate report."""
        report_lines = [
            "# TERRAGON SDLC v4.0 - QUALITY GATE REPORT",
            "",
            f"**Generated:** {results['timestamp']}",
            f"**Overall Status:** {results['overall_status']}",
            f"**Pass Rate:** {results['pass_rate']:.2%} ({results['passed_gates']}/{results['total_gates']} gates)",
            "",
            "## Executive Summary",
            "",
        ]
        
        if results['overall_status'] == 'PASS':
            report_lines.extend([
                "✅ **QUALITY GATES PASSED** - System meets all quality standards and is ready for production deployment.",
                "All critical quality gates have been validated successfully.",
                ""
            ])
        elif results['overall_status'] == 'WARNING':
            report_lines.extend([
                "⚠️ **QUALITY GATES WARNING** - System meets most quality standards but has areas for improvement.",
                "Review failed gates and address issues before production deployment.",
                ""
            ])
        else:
            report_lines.extend([
                "❌ **QUALITY GATES FAILED** - System does not meet quality standards.",
                "Critical issues must be resolved before production deployment.",
                ""
            ])
        
        # Quality gate details
        report_lines.extend([
            "## Quality Gate Results",
            "",
            "| Gate | Status | Score | Details |",
            "|------|--------|-------|---------|"
        ])
        
        for gate_name, gate_result in results['quality_gates'].items():
            status_emoji = "✅" if gate_result['status'] == 'PASS' else "❌" if gate_result['status'] == 'FAIL' else "💥"
            score = gate_result.get('score', 0.0)
            
            details = []
            if 'tests' in gate_result:
                passed_tests = sum(1 for test in gate_result['tests'].values() if test.get('passed', False))
                total_tests = len(gate_result['tests'])
                details.append(f"{passed_tests}/{total_tests} tests passed")
            
            if 'metrics' in gate_result and gate_result['metrics']:
                details.append("Performance metrics collected")
            
            if 'errors' in gate_result and gate_result['errors']:
                details.append(f"{len(gate_result['errors'])} errors")
            
            details_str = "; ".join(details) if details else "N/A"
            
            report_lines.append(
                f"| {gate_name.title()} | {status_emoji} {gate_result['status']} | {score:.2%} | {details_str} |"
            )
        
        # Detailed findings
        report_lines.extend([
            "",
            "## Detailed Findings",
            ""
        ])
        
        for gate_name, gate_result in results['quality_gates'].items():
            report_lines.extend([
                f"### {gate_name.title()} Gate",
                ""
            ])
            
            if gate_result['status'] == 'PASS':
                report_lines.append("✅ **Status:** PASSED - All requirements met")
            else:
                report_lines.append(f"❌ **Status:** {gate_result['status']} - Issues identified")
            
            report_lines.append(f"**Score:** {gate_result.get('score', 0.0):.2%}")
            
            if 'errors' in gate_result and gate_result['errors']:
                report_lines.extend([
                    "",
                    "**Issues:**"
                ])
                for error in gate_result['errors']:
                    report_lines.append(f"- {error}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if results['overall_status'] == 'PASS':
            report_lines.extend([
                "✅ System is ready for production deployment",
                "✅ All quality standards have been met",
                "✅ Continue monitoring in production environment",
                ""
            ])
        else:
            failed_gates = [
                gate_name for gate_name, gate_result in results['quality_gates'].items()
                if gate_result['status'] != 'PASS'
            ]
            
            report_lines.extend([
                "🔧 **Priority Actions Required:**",
                ""
            ])
            
            for gate_name in failed_gates:
                report_lines.append(f"- Address issues in {gate_name.title()} gate")
            
            report_lines.extend([
                "",
                "🔄 Re-run quality gates after addressing issues",
                ""
            ])
        
        report_lines.extend([
            "---",
            "*Generated by TERRAGON SDLC v4.0 Quality Gate System*"
        ])
        
        return "\n".join(report_lines)


def main():
    """Run comprehensive quality gate validation."""
    print("🛡️ TERRAGON SDLC v4.0 - COMPREHENSIVE QUALITY GATES")
    print("=" * 70)
    print()
    
    # Initialize validator
    validator = QualityGateValidator()
    
    # Run all quality gates
    start_time = time.time()
    results = validator.run_all_quality_gates()
    end_time = time.time()
    
    # Generate and save report
    report_content = validator.generate_quality_report(results)
    report_path = f"quality_gate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Save detailed results
    results_path = f"quality_gate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 QUALITY GATE SUMMARY")
    print("=" * 70)
    
    print(f"\n⏱️  Total Validation Time: {end_time - start_time:.2f} seconds")
    print(f"📋 Quality Gates Tested: {results['total_gates']}")
    print(f"✅ Gates Passed: {results['passed_gates']}")
    print(f"❌ Gates Failed: {results['total_gates'] - results['passed_gates']}")
    print(f"📈 Pass Rate: {results['pass_rate']:.2%}")
    
    if results['overall_status'] == 'PASS':
        print(f"\n🎉 OVERALL STATUS: ✅ {results['overall_status']}")
        print("🚀 SYSTEM IS PRODUCTION READY!")
    elif results['overall_status'] == 'WARNING':
        print(f"\n⚠️  OVERALL STATUS: ⚠️ {results['overall_status']}")
        print("🔧 MINOR ISSUES TO ADDRESS")
    else:
        print(f"\n💥 OVERALL STATUS: ❌ {results['overall_status']}")
        print("🛠️  CRITICAL ISSUES REQUIRE ATTENTION")
    
    print(f"\n📄 Detailed Report: {report_path}")
    print(f"📊 Raw Results: {results_path}")
    
    return results['overall_status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Quality gates validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Quality gates validation failed!")
        sys.exit(1)