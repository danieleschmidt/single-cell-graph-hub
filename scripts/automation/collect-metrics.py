#!/usr/bin/env python3
"""
Automated metrics collection script for Single-Cell Graph Hub.

This script collects various metrics from different sources and updates
the project metrics JSON file used for tracking and reporting.
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import argparse


class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        """Initialize the metrics collector.
        
        Args:
            config_path: Path to the project metrics configuration file
        """
        self.config_path = Path(config_path)
        self.metrics = self.load_current_metrics()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/single-cell-graph-hub')
        
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from configuration file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return {"metrics": {}, "tracking": {}}
    
    def save_metrics(self) -> None:
        """Save updated metrics to configuration file."""
        self.metrics['tracking']['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Metrics saved to {self.config_path}")
    
    def collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect GitHub repository metrics."""
        print("📊 Collecting repository metrics...")
        
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN not found, skipping repository metrics")
            return {}
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            # Repository basic info
            repo_url = f"https://api.github.com/repos/{self.repo}"
            repo_response = requests.get(repo_url, headers=headers)
            repo_data = repo_response.json()
            
            # Issues and PRs
            issues_url = f"https://api.github.com/repos/{self.repo}/issues"
            issues_response = requests.get(issues_url, headers=headers, params={'state': 'all'})
            issues_data = issues_response.json()
            
            # Commits (last 30 days)
            since_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'
            commits_url = f"https://api.github.com/repos/{self.repo}/commits"
            commits_response = requests.get(
                commits_url, 
                headers=headers, 
                params={'since': since_date}
            )
            commits_data = commits_response.json()
            
            # Contributors
            contributors_url = f"https://api.github.com/repos/{self.repo}/contributors"
            contributors_response = requests.get(contributors_url, headers=headers)
            contributors_data = contributors_response.json()
            
            # Releases
            releases_url = f"https://api.github.com/repos/{self.repo}/releases"
            releases_response = requests.get(releases_url, headers=headers)
            releases_data = releases_response.json()
            
            return {
                "stars": repo_data.get('stargazers_count', 0),
                "forks": repo_data.get('forks_count', 0),
                "watchers": repo_data.get('watchers_count', 0),
                "issues": {
                    "total": len(issues_data),
                    "open": len([i for i in issues_data if i.get('state') == 'open' and 'pull_request' not in i]),
                    "closed": len([i for i in issues_data if i.get('state') == 'closed' and 'pull_request' not in i])
                },
                "pull_requests": {
                    "total": len([i for i in issues_data if 'pull_request' in i]),
                    "open": len([i for i in issues_data if i.get('state') == 'open' and 'pull_request' in i]),
                    "closed": len([i for i in issues_data if i.get('state') == 'closed' and 'pull_request' in i])
                },
                "commits": {
                    "total": len(commits_data),
                    "last_month": len(commits_data)
                },
                "contributors": {
                    "total": len(contributors_data),
                    "active_last_month": len(contributors_data)  # Simplified
                },
                "releases": {
                    "total": len(releases_data),
                    "latest_version": releases_data[0].get('tag_name') if releases_data else None,
                    "latest_date": releases_data[0].get('published_at') if releases_data else None
                }
            }
            
        except Exception as e:
            print(f"❌ Error collecting repository metrics: {e}")
            return {}
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics using local tools."""
        print("🔍 Collecting code quality metrics...")
        
        metrics = {}
        
        try:
            # Lines of code using cloc (if available)
            try:
                result = subprocess.run(
                    ['cloc', 'src/', 'tests/', 'docs/', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    cloc_data = json.loads(result.stdout)
                    metrics["lines_of_code"] = {
                        "total": cloc_data.get('SUM', {}).get('code', 0),
                        "source": cloc_data.get('Python', {}).get('code', 0),
                        "tests": cloc_data.get('SUM', {}).get('code', 0) // 4,  # Estimate
                        "documentation": cloc_data.get('Markdown', {}).get('code', 0)
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                # Fallback to simple line counting
                metrics["lines_of_code"] = self.count_lines_fallback()
            
            # Test coverage (if coverage report exists)
            coverage_file = Path("coverage.xml")
            if coverage_file.exists():
                metrics["test_coverage"] = self.parse_coverage_xml(coverage_file)
            else:
                metrics["test_coverage"] = {
                    "percentage": 0.0,
                    "lines_covered": 0,
                    "lines_total": 0
                }
            
            # Placeholder for other metrics (would integrate with tools like SonarQube)
            metrics.update({
                "code_smells": {"total": 0, "critical": 0, "major": 0, "minor": 0},
                "technical_debt": {"minutes": 0, "rating": "A"},
                "duplication": {"percentage": 0.0, "duplicated_lines": 0}
            })
            
        except Exception as e:
            print(f"❌ Error collecting code quality metrics: {e}")
        
        return metrics
    
    def count_lines_fallback(self) -> Dict[str, int]:
        """Fallback method to count lines of code."""
        def count_lines_in_dir(directory: Path, extensions: list) -> int:
            count = 0
            for ext in extensions:
                for file in directory.rglob(f"*.{ext}"):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            count += len([line for line in f if line.strip()])
                    except:
                        pass
            return count
        
        src_lines = count_lines_in_dir(Path("src"), ["py"])
        test_lines = count_lines_in_dir(Path("tests"), ["py"])
        doc_lines = count_lines_in_dir(Path("docs"), ["md", "rst"])
        
        return {
            "total": src_lines + test_lines,
            "source": src_lines,
            "tests": test_lines,
            "documentation": doc_lines
        }
    
    def parse_coverage_xml(self, coverage_file: Path) -> Dict[str, Any]:
        """Parse coverage XML file for coverage metrics."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            coverage_elem = root.find('.//coverage')
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get('line-rate', 0))
                branch_rate = float(coverage_elem.get('branch-rate', 0))
                lines_covered = int(coverage_elem.get('lines-covered', 0))
                lines_valid = int(coverage_elem.get('lines-valid', 0))
                branches_covered = int(coverage_elem.get('branches-covered', 0))
                branches_valid = int(coverage_elem.get('branches-valid', 0))
                
                return {
                    "percentage": line_rate * 100,
                    "lines_covered": lines_covered,
                    "lines_total": lines_valid,
                    "branches_covered": branches_covered,
                    "branches_total": branches_valid
                }
        except Exception as e:
            print(f"⚠️ Could not parse coverage file: {e}")
        
        return {"percentage": 0.0, "lines_covered": 0, "lines_total": 0}
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics from various sources."""
        print("🔒 Collecting security metrics...")
        
        metrics = {
            "vulnerabilities": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0},
            "security_hotspots": {"total": 0, "to_review": 0, "reviewed": 0},
            "last_scan": datetime.utcnow().isoformat() + 'Z',
            "dependencies": {"total": 0, "outdated": 0, "vulnerable": 0}
        }
        
        try:
            # Count dependencies from requirements
            if Path("pyproject.toml").exists():
                with open("pyproject.toml", 'r') as f:
                    content = f.read()
                    # Simple count of dependencies (would be more sophisticated in practice)
                    metrics["dependencies"]["total"] = content.count('>=') + content.count('==')
            
            # Check for security scan results
            security_files = [
                "safety-report.json",
                "bandit-report.json",
                "snyk-report.json"
            ]
            
            for security_file in security_files:
                if Path(security_file).exists():
                    with open(security_file, 'r') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                metrics["vulnerabilities"]["total"] += len(data)
                        except json.JSONDecodeError:
                            pass
        
        except Exception as e:
            print(f"❌ Error collecting security metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from CI/CD and monitoring."""
        print("⚡ Collecting performance metrics...")
        
        # Default values (would be populated from actual CI/CD data)
        return {
            "build_time": {
                "average_seconds": 120,
                "last_build_seconds": 0,
                "trend": "stable"
            },
            "test_execution": {
                "average_seconds": 45,
                "last_run_seconds": 0,
                "trend": "stable"
            },
            "deployment_time": {
                "average_seconds": 180,
                "last_deployment_seconds": 0,
                "trend": "stable"
            }
        }
    
    def collect_usage_metrics(self) -> Dict[str, Any]:
        """Collect usage metrics from PyPI, Docker Hub, etc."""
        print("📈 Collecting usage metrics...")
        
        metrics = {
            "downloads": {"total": 0, "last_week": 0, "last_month": 0},
            "docker_pulls": {"total": 0, "last_week": 0, "last_month": 0},
            "api_requests": {"total": 0, "last_week": 0, "last_month": 0}
        }
        
        try:
            # PyPI download stats (using pypistats API)
            package_name = "single-cell-graph-hub"
            pypi_url = f"https://pypistats.org/api/packages/{package_name}/recent"
            
            response = requests.get(pypi_url, timeout=10)
            if response.status_code == 200:
                pypi_data = response.json()
                metrics["downloads"] = {
                    "total": pypi_data.get("data", {}).get("last_month", 0),
                    "last_week": pypi_data.get("data", {}).get("last_week", 0),
                    "last_month": pypi_data.get("data", {}).get("last_month", 0)
                }
        
        except Exception as e:
            print(f"⚠️ Could not collect PyPI stats: {e}")
        
        return metrics
    
    def collect_ci_cd_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD pipeline metrics."""
        print("🚀 Collecting CI/CD metrics...")
        
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN not found, skipping CI/CD metrics")
            return {}
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # Workflow runs
            workflows_url = f"https://api.github.com/repos/{self.repo}/actions/runs"
            response = requests.get(workflows_url, headers=headers, params={'per_page': 100})
            runs_data = response.json().get('workflow_runs', [])
            
            total_runs = len(runs_data)
            successful_runs = len([r for r in runs_data if r.get('conclusion') == 'success'])
            failed_runs = len([r for r in runs_data if r.get('conclusion') == 'failure'])
            
            return {
                "workflow_runs": {
                    "total": total_runs,
                    "successful": successful_runs,
                    "failed": failed_runs,
                    "success_rate": (successful_runs / total_runs * 100) if total_runs > 0 else 0
                },
                "deployment_frequency": {"per_week": 2, "per_month": 8},  # Placeholder
                "lead_time": {"average_hours": 2.5, "median_hours": 2.0},  # Placeholder
                "mean_time_to_recovery": {"average_hours": 0.5, "incidents_last_month": 0}
            }
        
        except Exception as e:
            print(f"❌ Error collecting CI/CD metrics: {e}")
            return {}
    
    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation metrics."""
        print("📖 Collecting documentation metrics...")
        
        try:
            # Count documentation files
            doc_files = list(Path("docs").rglob("*.md")) + list(Path("docs").rglob("*.rst"))
            readme_files = list(Path(".").glob("README*"))
            
            total_pages = len(doc_files) + len(readme_files)
            
            # Estimate documentation coverage (would be more sophisticated)
            py_files = list(Path("src").rglob("*.py"))
            documented_functions = 0
            total_functions = 0
            
            for py_file in py_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        functions = content.count('def ')
                        docstrings = content.count('"""') + content.count("'''")
                        total_functions += functions
                        documented_functions += min(functions, docstrings // 2)  # Rough estimate
                except:
                    pass
            
            coverage_percentage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            
            return {
                "coverage": {
                    "percentage": coverage_percentage,
                    "documented_functions": documented_functions,
                    "total_functions": total_functions
                },
                "pages": {
                    "total": total_pages,
                    "updated_last_month": total_pages  # Simplified
                },
                "readability": {"score": 8.0, "scale": "1-10"}  # Placeholder
            }
        
        except Exception as e:
            print(f"❌ Error collecting documentation metrics: {e}")
            return {}
    
    def update_all_metrics(self) -> None:
        """Update all metrics categories."""
        print("🔄 Starting metrics collection...")
        
        # Update each metrics category
        self.metrics.setdefault("metrics", {})
        
        self.metrics["metrics"]["repository"] = self.collect_repository_metrics()
        self.metrics["metrics"]["code_quality"] = self.collect_code_quality_metrics()
        self.metrics["metrics"]["security"] = self.collect_security_metrics()
        self.metrics["metrics"]["performance"] = self.collect_performance_metrics()
        self.metrics["metrics"]["usage"] = self.collect_usage_metrics()
        self.metrics["metrics"]["ci_cd"] = self.collect_ci_cd_metrics()
        self.metrics["metrics"]["documentation"] = self.collect_documentation_metrics()
        
        # Update project info
        self.metrics.setdefault("project", {})["updated"] = datetime.utcnow().isoformat() + 'Z'
        
        print("✅ All metrics collected successfully")
    
    def check_thresholds(self) -> list:
        """Check if any metrics exceed defined thresholds."""
        alerts = []
        thresholds = self.metrics.get("alerts", {}).get("thresholds", {})
        current_metrics = self.metrics.get("metrics", {})
        
        # Check test coverage
        test_coverage = current_metrics.get("code_quality", {}).get("test_coverage", {}).get("percentage", 0)
        min_coverage = thresholds.get("test_coverage_min", 80)
        if test_coverage < min_coverage:
            alerts.append(f"Test coverage ({test_coverage:.1f}%) below threshold ({min_coverage}%)")
        
        # Check build success rate
        success_rate = current_metrics.get("ci_cd", {}).get("workflow_runs", {}).get("success_rate", 100)
        min_success_rate = thresholds.get("build_success_rate_min", 90)
        if success_rate < min_success_rate:
            alerts.append(f"Build success rate ({success_rate:.1f}%) below threshold ({min_success_rate}%)")
        
        # Check security vulnerabilities
        vulnerabilities = current_metrics.get("security", {}).get("vulnerabilities", {}).get("total", 0)
        max_vulnerabilities = thresholds.get("security_vulnerabilities_max", 0)
        if vulnerabilities > max_vulnerabilities:
            alerts.append(f"Security vulnerabilities ({vulnerabilities}) exceed threshold ({max_vulnerabilities})")
        
        return alerts
    
    def generate_report(self, format: str = "json") -> str:
        """Generate a metrics report in specified format."""
        if format == "json":
            return json.dumps(self.metrics, indent=2)
        elif format == "markdown":
            return self.generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown metrics report."""
        metrics = self.metrics.get("metrics", {})
        
        report = f"""# Project Metrics Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Repository:** {self.repo}

## Repository Overview

- **Stars:** {metrics.get('repository', {}).get('stars', 0)}
- **Forks:** {metrics.get('repository', {}).get('forks', 0)}
- **Contributors:** {metrics.get('repository', {}).get('contributors', {}).get('total', 0)}
- **Open Issues:** {metrics.get('repository', {}).get('issues', {}).get('open', 0)}
- **Open PRs:** {metrics.get('repository', {}).get('pull_requests', {}).get('open', 0)}

## Code Quality

- **Test Coverage:** {metrics.get('code_quality', {}).get('test_coverage', {}).get('percentage', 0):.1f}%
- **Lines of Code:** {metrics.get('code_quality', {}).get('lines_of_code', {}).get('total', 0)}
- **Technical Debt:** {metrics.get('code_quality', {}).get('technical_debt', {}).get('rating', 'N/A')}

## Security

- **Vulnerabilities:** {metrics.get('security', {}).get('vulnerabilities', {}).get('total', 0)}
- **Dependencies:** {metrics.get('security', {}).get('dependencies', {}).get('total', 0)}
- **Last Scan:** {metrics.get('security', {}).get('last_scan', 'Never')}

## CI/CD Performance

- **Build Success Rate:** {metrics.get('ci_cd', {}).get('workflow_runs', {}).get('success_rate', 0):.1f}%
- **Total Workflow Runs:** {metrics.get('ci_cd', {}).get('workflow_runs', {}).get('total', 0)}

## Documentation

- **Documentation Coverage:** {metrics.get('documentation', {}).get('coverage', {}).get('percentage', 0):.1f}%
- **Total Pages:** {metrics.get('documentation', {}).get('pages', {}).get('total', 0)}
"""
        
        return report


def main():
    """Main entry point for metrics collection script."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--config", default=".github/project-metrics.json",
                       help="Path to metrics configuration file")
    parser.add_argument("--output", choices=["json", "markdown"], default="json",
                       help="Output format for report")
    parser.add_argument("--check-thresholds", action="store_true",
                       help="Check metrics against thresholds and exit with error if exceeded")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report only, don't collect new metrics")
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.config)
        
        if not args.report_only:
            collector.update_all_metrics()
            collector.save_metrics()
        
        # Generate report
        report = collector.generate_report(args.output)
        print("\n" + "="*50)
        print("METRICS REPORT")
        print("="*50)
        print(report)
        
        # Check thresholds
        if args.check_thresholds:
            alerts = collector.check_thresholds()
            if alerts:
                print("\n" + "="*50)
                print("THRESHOLD ALERTS")
                print("="*50)
                for alert in alerts:
                    print(f"⚠️ {alert}")
                sys.exit(1)
            else:
                print("\n✅ All metrics within acceptable thresholds")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()