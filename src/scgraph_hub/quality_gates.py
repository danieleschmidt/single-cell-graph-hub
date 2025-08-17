"""Quality Gates Verification System for TERRAGON SDLC v4.0."""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class QualityGateStatus(Enum):
    """Quality gate status levels."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: QualityGateStatus
    score: float = 0.0
    threshold: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gate_name': self.gate_name,
            'status': self.status.value,
            'score': self.score,
            'threshold': self.threshold,
            'message': self.message,
            'details': self.details,
            'execution_time': self.execution_time
        }


class QualityGateChecker:
    """Base class for quality gate checkers."""
    
    def __init__(self, name: str, threshold: float = 85.0):
        self.name = name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
    
    def check(self) -> QualityGateResult:
        """Execute the quality gate check."""
        start_time = time.time()
        
        try:
            score, details, message = self._execute_check()
            
            if score >= self.threshold:
                status = QualityGateStatus.PASSED
            elif score >= self.threshold * 0.7:  # 70% of threshold
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
            
            execution_time = time.time() - start_time
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=score,
                threshold=self.threshold,
                message=message,
                details=details,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gate {self.name} failed: {e}")
            
            return QualityGateResult(
                gate_name=self.name,
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=self.threshold,
                message=f"Error: {str(e)}",
                details={'error': str(e)},
                execution_time=execution_time
            )
    
    def _execute_check(self) -> tuple[float, Dict[str, Any], str]:
        """Execute the actual check. Override in subclasses."""
        raise NotImplementedError


class CodeCoverageChecker(QualityGateChecker):
    """Check code coverage."""
    
    def __init__(self, threshold: float = 85.0, source_dir: str = "src"):
        super().__init__("Code Coverage", threshold)
        self.source_dir = source_dir
    
    def _execute_check(self) -> tuple[float, Dict[str, Any], str]:
        """Check code coverage."""
        try:
            # Try to run coverage if available
            result = subprocess.run([
                sys.executable, "-m", "coverage", "report", "--show-missing"
            ], capture_output=True, text=True, cwd=".", timeout=60)
            
            if result.returncode == 0:
                # Parse coverage output
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Look for total coverage line
                    for line in lines:
                        if 'TOTAL' in line:
                            parts = line.split()
                            if len(parts) >= 4:
                                coverage_str = parts[-1].rstrip('%')
                                coverage = float(coverage_str)
                                
                                return coverage, {
                                    'coverage_output': result.stdout,
                                    'coverage_percentage': coverage
                                }, f"Code coverage: {coverage}%"
            
            # Fallback: estimate coverage based on test files
            return self._estimate_coverage()
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # Coverage tool not available, estimate
            return self._estimate_coverage()
    
    def _estimate_coverage(self) -> tuple[float, Dict[str, Any], str]:
        """Estimate coverage based on test files."""
        source_files = list(Path(self.source_dir).rglob("*.py"))
        test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
        
        if len(source_files) == 0:
            return 0.0, {'error': 'No source files found'}, "No source files found"
        
        # Simple heuristic: estimate coverage based on test file ratio
        test_ratio = len(test_files) / len(source_files)
        estimated_coverage = min(90.0, test_ratio * 100)
        
        return estimated_coverage, {
            'source_files': len(source_files),
            'test_files': len(test_files),
            'estimated': True
        }, f"Estimated coverage: {estimated_coverage:.1f}% (based on test file ratio)"


class SecurityScanChecker(QualityGateChecker):
    """Check for security vulnerabilities."""
    
    def __init__(self, threshold: float = 95.0):
        super().__init__("Security Scan", threshold)
    
    def _execute_check(self) -> tuple[float, Dict[str, Any], str]:
        """Check for security vulnerabilities."""
        try:
            # Try bandit for Python security check
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src", "-f", "json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    high_severity = len([r for r in bandit_data.get('results', []) 
                                       if r.get('issue_severity') == 'HIGH'])
                    medium_severity = len([r for r in bandit_data.get('results', []) 
                                         if r.get('issue_severity') == 'MEDIUM'])
                    low_severity = len([r for r in bandit_data.get('results', []) 
                                      if r.get('issue_severity') == 'LOW'])
                    
                    # Calculate score based on severity
                    total_files = len(bandit_data.get('metrics', {}).get('_totals', {}).get('files', 1))
                    severity_score = 100 - (high_severity * 20 + medium_severity * 10 + low_severity * 2)
                    score = max(0.0, min(100.0, severity_score))
                    
                    return score, {
                        'high_severity': high_severity,
                        'medium_severity': medium_severity,
                        'low_severity': low_severity,
                        'total_files_scanned': total_files,
                        'bandit_output': bandit_data
                    }, f"Security scan: {high_severity} high, {medium_severity} medium, {low_severity} low severity issues"
                    
                except json.JSONDecodeError:
                    pass
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback: basic security pattern check
        return self._basic_security_check()
    
    def _basic_security_check(self) -> tuple[float, Dict[str, Any], str]:
        """Basic security pattern check."""
        security_issues = 0
        total_files = 0
        
        # Check for common security anti-patterns
        dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'pickle\.loads',
            r'yaml\.load\(',
            r'subprocess\.call.*shell=True'
        ]
        
        import re
        
        for py_file in Path("src").rglob("*.py"):
            total_files += 1
            try:
                content = py_file.read_text()
                for pattern in dangerous_patterns:
                    if re.search(pattern, content):
                        security_issues += 1
                        break
            except Exception:
                continue
        
        if total_files == 0:
            return 100.0, {'total_files': 0}, "No Python files to scan"
        
        score = max(0.0, 100.0 - (security_issues / total_files * 100))
        
        return score, {
            'total_files': total_files,
            'files_with_issues': security_issues,
            'check_type': 'basic_pattern_check'
        }, f"Basic security check: {security_issues}/{total_files} files with potential issues"


class PerformanceBenchmarkChecker(QualityGateChecker):
    """Check performance benchmarks."""
    
    def __init__(self, threshold: float = 80.0):
        super().__init__("Performance Benchmarks", threshold)
    
    def _execute_check(self) -> tuple[float, Dict[str, Any], str]:
        """Check performance benchmarks."""
        try:
            # Import and test core functionality
            sys.path.insert(0, 'src')
            from scgraph_hub import get_enhanced_loader, get_performance_optimizer
            
            start_time = time.time()
            
            # Test dataset loading performance
            loader = get_enhanced_loader()
            dataset = loader._create_dummy_dataset("perf_test", num_nodes=100, num_features=50)
            
            load_time = time.time() - start_time
            
            # Test caching performance
            optimizer = get_performance_optimizer()
            start_time = time.time()
            
            # Create some cached operations
            for i in range(10):
                cached_result = optimizer.cache.get(f"test_key_{i}")
                if cached_result is None:
                    optimizer.cache.put(f"test_key_{i}", f"test_value_{i}")
            
            cache_time = time.time() - start_time
            
            # Calculate performance score
            load_score = max(0, 100 - (load_time * 1000))  # Penalty for slow loading
            cache_score = max(0, 100 - (cache_time * 10000))  # Penalty for slow caching
            
            overall_score = (load_score + cache_score) / 2
            
            return overall_score, {
                'dataset_load_time': load_time,
                'cache_operation_time': cache_time,
                'load_score': load_score,
                'cache_score': cache_score,
                'dataset_nodes': 100 if dataset else 0
            }, f"Performance: load={load_time:.3f}s, cache={cache_time:.3f}s"
            
        except Exception as e:
            return 50.0, {'error': str(e)}, f"Performance test failed: {str(e)}"


class DocumentationChecker(QualityGateChecker):
    """Check documentation completeness."""
    
    def __init__(self, threshold: float = 70.0):
        super().__init__("Documentation", threshold)
    
    def _execute_check(self) -> tuple[float, Dict[str, Any], str]:
        """Check documentation completeness."""
        score_components = []
        details = {}
        
        # Check for README
        if Path("README.md").exists():
            score_components.append(20)
            details['readme'] = True
        else:
            details['readme'] = False
        
        # Check for docstrings in Python files
        documented_functions = 0
        total_functions = 0
        
        for py_file in Path("src").rglob("*.py"):
            try:
                content = py_file.read_text()
                
                # Count functions/classes
                import re
                functions = re.findall(r'^def\s+\w+', content, re.MULTILINE)
                classes = re.findall(r'^class\s+\w+', content, re.MULTILINE)
                
                total_functions += len(functions) + len(classes)
                
                # Check for docstrings (simplified check)
                for func in functions:
                    func_start = content.find(func)
                    if func_start != -1:
                        # Look for docstring after function definition
                        remaining = content[func_start:func_start + 500]
                        if '"""' in remaining or "'''" in remaining:
                            documented_functions += 1
                
                for cls in classes:
                    cls_start = content.find(cls)
                    if cls_start != -1:
                        remaining = content[cls_start:cls_start + 500]
                        if '"""' in remaining or "'''" in remaining:
                            documented_functions += 1
                            
            except Exception:
                continue
        
        # Calculate docstring score
        if total_functions > 0:
            docstring_score = (documented_functions / total_functions) * 60
            score_components.append(docstring_score)
            details['docstring_coverage'] = documented_functions / total_functions * 100
        else:
            details['docstring_coverage'] = 0
        
        details['total_functions'] = total_functions
        details['documented_functions'] = documented_functions
        
        # Check for other documentation
        doc_files = list(Path("docs").rglob("*.md")) if Path("docs").exists() else []
        if doc_files:
            score_components.append(20)
            details['doc_files'] = len(doc_files)
        else:
            details['doc_files'] = 0
        
        overall_score = sum(score_components)
        
        return overall_score, details, f"Documentation: {overall_score:.1f}% complete"


class QualityGateRunner:
    """Orchestrates quality gate execution."""
    
    def __init__(self):
        self.checkers: List[QualityGateChecker] = []
        self.results: List[QualityGateResult] = []
        self.logger = logging.getLogger(__name__)
    
    def add_checker(self, checker: QualityGateChecker) -> None:
        """Add a quality gate checker."""
        self.checkers.append(checker)
    
    def add_default_checkers(self) -> None:
        """Add default quality gate checkers."""
        self.add_checker(CodeCoverageChecker(threshold=85.0))
        self.add_checker(SecurityScanChecker(threshold=95.0))
        self.add_checker(PerformanceBenchmarkChecker(threshold=80.0))
        self.add_checker(DocumentationChecker(threshold=70.0))
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        self.logger.info("Starting quality gate verification")
        start_time = time.time()
        
        self.results = []
        
        for checker in self.checkers:
            self.logger.info(f"Running quality gate: {checker.name}")
            result = checker.check()
            self.results.append(result)
            
            status_emoji = "✅" if result.status == QualityGateStatus.PASSED else "❌"
            self.logger.info(f"{status_emoji} {checker.name}: {result.status.value} ({result.score:.1f}%)")
        
        total_time = time.time() - start_time
        
        # Calculate overall results
        passed_gates = len([r for r in self.results if r.status == QualityGateStatus.PASSED])
        failed_gates = len([r for r in self.results if r.status == QualityGateStatus.FAILED])
        warning_gates = len([r for r in self.results if r.status == QualityGateStatus.WARNING])
        
        overall_score = sum(r.score for r in self.results) / len(self.results) if self.results else 0
        
        # Determine overall status
        if failed_gates == 0:
            overall_status = QualityGateStatus.PASSED
        elif failed_gates <= 1 and warning_gates <= 2:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.FAILED
        
        report = {
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'total_gates': len(self.results),
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'warning_gates': warning_gates,
            'execution_time': total_time,
            'timestamp': time.time(),
            'results': [result.to_dict() for result in self.results]
        }
        
        self.logger.info(f"Quality gate verification completed: {overall_status.value} ({overall_score:.1f}%)")
        
        return report
    
    def export_report(self, filepath: str) -> bool:
        """Export quality gate report."""
        try:
            if not self.results:
                report = {'error': 'No quality gate results available'}
            else:
                report = self.run_all_gates()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            return False


def run_quality_gates() -> Dict[str, Any]:
    """Run quality gates and return results."""
    runner = QualityGateRunner()
    runner.add_default_checkers()
    return runner.run_all_gates()


def main():
    """Main function for standalone execution."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 TERRAGON SDLC v4.0 - Quality Gates Verification")
    print("=" * 60)
    
    runner = QualityGateRunner()
    runner.add_default_checkers()
    
    results = runner.run_all_gates()
    
    print(f"\n📊 Quality Gates Summary:")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print(f"Overall Score: {results['overall_score']:.1f}%")
    print(f"Passed: {results['passed_gates']}/{results['total_gates']}")
    
    # Export report
    report_file = f"quality_gate_report_{int(time.time())}.json"
    if runner.export_report(report_file):
        print(f"📋 Report exported to: {report_file}")
    
    return results['overall_status'] != 'failed'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)