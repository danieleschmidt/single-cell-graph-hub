#!/usr/bin/env python3
"""
Setup verification script for Single-Cell Graph Hub.

This script verifies that all components of the SDLC implementation
are properly configured and functioning.
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import urllib.request
import urllib.error
from datetime import datetime

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SetupVerifier:
    """Comprehensive setup verification for Single-Cell Graph Hub."""
    
    def __init__(self):
        """Initialize the setup verifier."""
        self.repo_root = Path('.')
        self.results = {
            'overall_score': 0,
            'max_score': 0,
            'categories': {},
            'issues': [],
            'recommendations': []
        }
        
    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")
    
    def print_success(self, text: str) -> None:
        """Print success message."""
        print(f"{Colors.GREEN}✅ {text}{Colors.END}")
    
    def print_warning(self, text: str) -> None:
        """Print warning message."""
        print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")
    
    def print_error(self, text: str) -> None:
        """Print error message."""
        print(f"{Colors.RED}❌ {text}{Colors.END}")
    
    def print_info(self, text: str) -> None:
        """Print info message."""
        print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")
    
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run a shell command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False, ""
    
    def check_file_exists(self, path: str, description: str) -> bool:
        """Check if a file exists and report result."""
        file_path = self.repo_root / path
        if file_path.exists():
            self.print_success(f"{description}: {path}")
            return True
        else:
            self.print_error(f"{description} missing: {path}")
            self.results['issues'].append(f"Missing file: {path}")
            return False
    
    def check_directory_exists(self, path: str, description: str) -> bool:
        """Check if a directory exists and report result."""
        dir_path = self.repo_root / path
        if dir_path.exists() and dir_path.is_dir():
            file_count = len(list(dir_path.rglob('*')))
            self.print_success(f"{description}: {path} ({file_count} files)")
            return True
        else:
            self.print_error(f"{description} missing: {path}")
            self.results['issues'].append(f"Missing directory: {path}")
            return False
    
    def verify_project_structure(self) -> Dict[str, Any]:
        """Verify the basic project structure."""
        self.print_header("PROJECT STRUCTURE VERIFICATION")
        
        score = 0
        max_score = 20
        
        # Essential files
        essential_files = [
            ('README.md', 'Main README'),
            ('pyproject.toml', 'Python project configuration'),
            ('Dockerfile', 'Docker configuration'),
            ('docker-compose.yml', 'Docker Compose configuration'),
            ('.gitignore', 'Git ignore rules'),
            ('.env.template', 'Environment template'),
            ('LICENSE', 'License file'),
        ]
        
        for file_path, description in essential_files:
            if self.check_file_exists(file_path, description):
                score += 2
        
        # Essential directories
        essential_dirs = [
            ('src/', 'Source code'),
            ('tests/', 'Test suite'),
            ('docs/', 'Documentation'),
            ('scripts/', 'Utility scripts'),
            ('.github/', 'GitHub configuration'),
        ]
        
        for dir_path, description in essential_dirs:
            if self.check_directory_exists(dir_path, description):
                score += 1
        
        # Additional structure checks
        if (self.repo_root / 'src' / 'scgraph_hub').exists():
            self.print_success("Source package structure: src/scgraph_hub/")
            score += 1
        else:
            self.print_warning("Source package structure not found: src/scgraph_hub/")
        
        self.results['categories']['project_structure'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['project_structure']
    
    def verify_development_environment(self) -> Dict[str, Any]:
        """Verify development environment setup."""
        self.print_header("DEVELOPMENT ENVIRONMENT VERIFICATION")
        
        score = 0
        max_score = 15
        
        # Check Python version
        success, python_version = self.run_command(['python', '--version'])
        if success and 'Python 3.' in python_version:
            version_parts = python_version.split()[1].split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            if major == 3 and minor >= 8:
                self.print_success(f"Python version: {python_version}")
                score += 3
            else:
                self.print_warning(f"Python version may be too old: {python_version}")
                score += 1
        else:
            self.print_error("Python not found or version check failed")
        
        # Check essential tools
        tools = [
            ('git', 'Git version control'),
            ('docker', 'Docker containerization'),
            ('docker-compose', 'Docker Compose'),
        ]
        
        for tool, description in tools:
            success, output = self.run_command([tool, '--version'])
            if success:
                self.print_success(f"{description}: Available")
                score += 2
            else:
                self.print_error(f"{description}: Not available")
                self.results['issues'].append(f"Missing tool: {tool}")
        
        # Check pre-commit
        if self.check_file_exists('.pre-commit-config.yaml', 'Pre-commit configuration'):
            success, _ = self.run_command(['pre-commit', '--version'])
            if success:
                self.print_success("Pre-commit: Available")
                score += 2
            else:
                self.print_warning("Pre-commit: Not installed")
                score += 1
        
        # Check development configuration files
        dev_files = [
            ('.editorconfig', 'Editor configuration'),
            ('.devcontainer/devcontainer.json', 'Development container'),
        ]
        
        for file_path, description in dev_files:
            if self.check_file_exists(file_path, description):
                score += 1
        
        self.results['categories']['development_environment'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['development_environment']
    
    def verify_testing_infrastructure(self) -> Dict[str, Any]:
        """Verify testing infrastructure."""
        self.print_header("TESTING INFRASTRUCTURE VERIFICATION")
        
        score = 0
        max_score = 12
        
        # Check test directories and files
        test_structure = [
            ('tests/conftest.py', 'Pytest configuration'),
            ('tests/unit/', 'Unit tests directory'),
            ('tests/integration/', 'Integration tests directory'),
            ('tests/fixtures/', 'Test fixtures'),
        ]
        
        for path, description in test_structure:
            if self.check_file_exists(path, description) or self.check_directory_exists(path, description):
                score += 2
        
        # Check for pytest and test dependencies
        try:
            import pytest
            self.print_success("Pytest: Available")
            score += 2
        except ImportError:
            self.print_warning("Pytest: Not installed")
            score += 1
        
        # Check test configuration in pyproject.toml
        try:
            with open('pyproject.toml', 'r') as f:
                content = f.read()
                if '[tool.pytest' in content:
                    self.print_success("Pytest configuration: Found in pyproject.toml")
                    score += 2
                else:
                    self.print_warning("Pytest configuration: Not found in pyproject.toml")
        except FileNotFoundError:
            pass
        
        self.results['categories']['testing_infrastructure'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['testing_infrastructure']
    
    def verify_containerization(self) -> Dict[str, Any]:
        """Verify containerization setup."""
        self.print_header("CONTAINERIZATION VERIFICATION")
        
        score = 0
        max_score = 10
        
        # Check Docker files
        docker_files = [
            ('Dockerfile', 'Main Dockerfile'),
            ('docker-compose.yml', 'Docker Compose configuration'),
            ('.dockerignore', 'Docker ignore rules'),
        ]
        
        for file_path, description in docker_files:
            if self.check_file_exists(file_path, description):
                score += 2
        
        # Analyze Dockerfile
        dockerfile_path = self.repo_root / 'Dockerfile'
        if dockerfile_path.exists():
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                
            # Check for multi-stage build
            if 'FROM' in dockerfile_content and dockerfile_content.count('FROM') > 1:
                self.print_success("Dockerfile: Multi-stage build detected")
                score += 1
            
            # Check for security practices
            if 'USER' in dockerfile_content:
                self.print_success("Dockerfile: Non-root user configured")
                score += 1
            else:
                self.print_warning("Dockerfile: Consider adding non-root user")
            
            # Check for health check
            if 'HEALTHCHECK' in dockerfile_content:
                self.print_success("Dockerfile: Health check configured")
                score += 1
            else:
                self.print_info("Dockerfile: Health check not configured (optional)")
        
        # Test Docker build (if Docker is available)
        success, _ = self.run_command(['docker', '--version'])
        if success:
            self.print_info("Docker: Available for testing")
            score += 1
        
        self.results['categories']['containerization'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['containerization']
    
    def verify_documentation(self) -> Dict[str, Any]:
        """Verify documentation completeness."""
        self.print_header("DOCUMENTATION VERIFICATION")
        
        score = 0
        max_score = 15
        
        # Check main documentation files
        main_docs = [
            ('README.md', 'Main README'),
            ('docs/ARCHITECTURE.md', 'Architecture documentation'),
            ('docs/SETUP_REQUIRED.md', 'Setup instructions'),
            ('PROJECT_CHARTER.md', 'Project charter'),
        ]
        
        for file_path, description in main_docs:
            if self.check_file_exists(file_path, description):
                score += 2
        
        # Check README content
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            readme_sections = [
                ('installation', 'Installation instructions'),
                ('usage', 'Usage examples'),
                ('contributing', 'Contributing guidelines'),
                ('license', 'License information'),
            ]
            
            for section, description in readme_sections:
                if section.lower() in readme_content.lower():
                    self.print_success(f"README: {description} found")
                    score += 1
                else:
                    self.print_warning(f"README: {description} missing")
        
        # Check for additional documentation
        doc_dirs = [
            ('docs/runbooks/', 'Operational runbooks'),
            ('docs/workflows/', 'Workflow documentation'),
            ('docs/adr/', 'Architecture Decision Records'),
        ]
        
        for dir_path, description in doc_dirs:
            if self.check_directory_exists(dir_path, description):
                score += 1
        
        self.results['categories']['documentation'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['documentation']
    
    def verify_workflows_and_automation(self) -> Dict[str, Any]:
        """Verify workflow and automation setup."""
        self.print_header("WORKFLOWS & AUTOMATION VERIFICATION")
        
        score = 0
        max_score = 12
        
        # Check for workflow templates
        workflow_templates = [
            ('docs/workflows/examples/ci.yml', 'CI pipeline template'),
            ('docs/workflows/examples/security-scan.yml', 'Security scan template'),
            ('docs/workflows/examples/dependency-update.yml', 'Dependency update template'),
        ]
        
        for file_path, description in workflow_templates:
            if self.check_file_exists(file_path, description):
                score += 2
        
        # Check automation scripts
        automation_scripts = [
            ('scripts/automation/collect-metrics.py', 'Metrics collection'),
            ('scripts/automation/dependency-checker.py', 'Dependency checking'),
            ('scripts/automation/repository-maintenance.py', 'Repository maintenance'),
        ]
        
        for file_path, description in automation_scripts:
            if self.check_file_exists(file_path, description):
                script_path = self.repo_root / file_path
                if os.access(script_path, os.X_OK):
                    self.print_success(f"{description}: Executable")
                    score += 2
                else:
                    self.print_warning(f"{description}: Not executable")
                    score += 1
        
        self.results['categories']['workflows_automation'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['workflows_automation']
    
    def verify_security_configuration(self) -> Dict[str, Any]:
        """Verify security configuration."""
        self.print_header("SECURITY CONFIGURATION VERIFICATION")
        
        score = 0
        max_score = 10
        
        # Check security files
        security_files = [
            ('SECURITY.md', 'Security policy'),
            ('.gitignore', 'Git ignore (prevents secrets)'),
            ('.env.template', 'Environment template (not .env)'),
        ]
        
        for file_path, description in security_files:
            if self.check_file_exists(file_path, description):
                score += 2
        
        # Check that .env is NOT committed
        env_path = self.repo_root / '.env'
        if not env_path.exists():
            self.print_success("Environment file: .env not present (good)")
            score += 2
        else:
            self.print_warning("Environment file: .env found - ensure it's in .gitignore")
            score += 1
        
        # Check .gitignore content
        gitignore_path = self.repo_root / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            sensitive_patterns = ['.env', '*.key', '*.pem', '__pycache__']
            ignored_patterns = sum(1 for pattern in sensitive_patterns if pattern in gitignore_content)
            
            if ignored_patterns >= 3:
                self.print_success(f"Gitignore: {ignored_patterns}/{len(sensitive_patterns)} sensitive patterns covered")
                score += 2
            else:
                self.print_warning(f"Gitignore: Only {ignored_patterns}/{len(sensitive_patterns)} sensitive patterns covered")
                score += 1
        
        self.results['categories']['security_configuration'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['security_configuration']
    
    def verify_monitoring_setup(self) -> Dict[str, Any]:
        """Verify monitoring and observability setup."""
        self.print_header("MONITORING & OBSERVABILITY VERIFICATION")
        
        score = 0
        max_score = 8
        
        # Check monitoring configuration files
        monitoring_files = [
            ('monitoring/prometheus.yml', 'Prometheus configuration'),
            ('monitoring/grafana/dashboards/', 'Grafana dashboards'),
            ('.github/project-metrics.json', 'Project metrics configuration'),
        ]
        
        for file_path, description in monitoring_files:
            if (self.repo_root / file_path).exists():
                if file_path.endswith('/'):
                    if self.check_directory_exists(file_path, description):
                        score += 2
                else:
                    if self.check_file_exists(file_path, description):
                        score += 2
        
        # Check docker-compose for monitoring services
        docker_compose_path = self.repo_root / 'docker-compose.yml'
        if docker_compose_path.exists():
            with open(docker_compose_path, 'r') as f:
                compose_content = f.read()
            
            monitoring_services = ['prometheus', 'grafana']
            found_services = sum(1 for service in monitoring_services if service in compose_content)
            
            if found_services > 0:
                self.print_success(f"Docker Compose: {found_services}/{len(monitoring_services)} monitoring services configured")
                score += 2
            else:
                self.print_info("Docker Compose: No monitoring services found")
        
        self.results['categories']['monitoring_setup'] = {
            'score': score,
            'max_score': max_score,
            'percentage': (score / max_score) * 100
        }
        
        return self.results['categories']['monitoring_setup']
    
    def generate_recommendations(self) -> None:
        """Generate recommendations based on verification results."""
        recommendations = []
        
        # Analyze scores and generate specific recommendations
        for category, results in self.results['categories'].items():
            percentage = results['percentage']
            
            if percentage < 50:
                recommendations.append(f"🔴 {category.replace('_', ' ').title()}: Needs significant improvement ({percentage:.1f}%)")
            elif percentage < 80:
                recommendations.append(f"🟡 {category.replace('_', ' ').title()}: Good but could be better ({percentage:.1f}%)")
            else:
                recommendations.append(f"🟢 {category.replace('_', ' ').title()}: Excellent setup ({percentage:.1f}%)")
        
        # Add specific improvement suggestions
        if len(self.results['issues']) > 0:
            recommendations.append("\n📋 Priority Actions:")
            for issue in self.results['issues'][:5]:  # Top 5 issues
                recommendations.append(f"   • {issue}")
        
        # Add general recommendations
        recommendations.extend([
            "\n🚀 Next Steps:",
            "   • Review and address any missing files or directories",
            "   • Ensure all automation scripts are executable",
            "   • Complete manual setup steps in docs/SETUP_REQUIRED.md",
            "   • Test the CI pipeline with a sample pull request",
            "   • Configure monitoring dashboards for your environment"
        ])
        
        self.results['recommendations'] = recommendations
    
    def print_summary(self) -> None:
        """Print verification summary."""
        self.print_header("VERIFICATION SUMMARY")
        
        # Calculate overall score
        total_score = sum(cat['score'] for cat in self.results['categories'].values())
        total_max_score = sum(cat['max_score'] for cat in self.results['categories'].values())
        overall_percentage = (total_score / total_max_score) * 100 if total_max_score > 0 else 0
        
        self.results['overall_score'] = total_score
        self.results['max_score'] = total_max_score
        
        # Print overall score
        if overall_percentage >= 90:
            color = Colors.GREEN
            status = "EXCELLENT"
        elif overall_percentage >= 75:
            color = Colors.YELLOW
            status = "GOOD"
        elif overall_percentage >= 50:
            color = Colors.YELLOW
            status = "FAIR"
        else:
            color = Colors.RED
            status = "NEEDS WORK"
        
        print(f"{color}{Colors.BOLD}Overall Score: {total_score}/{total_max_score} ({overall_percentage:.1f}%) - {status}{Colors.END}\n")
        
        # Print category breakdown
        print(f"{Colors.BOLD}Category Breakdown:{Colors.END}")
        for category, results in self.results['categories'].items():
            category_name = category.replace('_', ' ').title()
            score = results['score']
            max_score = results['max_score']
            percentage = results['percentage']
            
            if percentage >= 80:
                color = Colors.GREEN
            elif percentage >= 60:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            
            print(f"  {color}• {category_name:<25} {score:>2}/{max_score:<2} ({percentage:>5.1f}%){Colors.END}")
        
        # Print recommendations
        if self.results['recommendations']:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
            for rec in self.results['recommendations']:
                print(rec)
        
        # Print final message
        print(f"\n{Colors.BOLD}Verification completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}{Colors.END}")
        
        if overall_percentage >= 80:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 Single-Cell Graph Hub setup is ready for development!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ Please address the recommendations above before proceeding.{Colors.END}")
    
    def save_results(self, output_file: str) -> None:
        """Save verification results to JSON file."""
        results_data = {
            **self.results,
            'verification_date': datetime.utcnow().isoformat(),
            'verifier_version': '1.0.0'
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.print_info(f"Verification results saved to: {output_file}")
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification checks."""
        print(f"{Colors.BOLD}{Colors.PURPLE}")
        print("Single-Cell Graph Hub - Setup Verification")
        print("==========================================")
        print(f"Verification started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{Colors.END}")
        
        # Run all verification categories
        self.verify_project_structure()
        self.verify_development_environment()
        self.verify_testing_infrastructure()
        self.verify_containerization()
        self.verify_documentation()
        self.verify_workflows_and_automation()
        self.verify_security_configuration()
        self.verify_monitoring_setup()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Print summary
        self.print_summary()
        
        return self.results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Single-Cell Graph Hub setup")
    parser.add_argument(
        '--output', '-o',
        help='Output file for verification results (JSON)',
        default='setup-verification-results.json'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Run verification
    verifier = SetupVerifier()
    results = verifier.run_all_verifications()
    
    # Save results
    verifier.save_results(args.output)
    
    # Exit with appropriate code
    overall_percentage = (results['overall_score'] / results['max_score']) * 100
    if overall_percentage < 50:
        sys.exit(1)  # Significant issues
    elif overall_percentage < 80:
        sys.exit(2)  # Minor issues
    else:
        sys.exit(0)  # All good

if __name__ == "__main__":
    main()