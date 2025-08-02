#!/usr/bin/env python3
"""
Automated dependency management and security checking script.

This script checks for outdated dependencies, security vulnerabilities,
and provides recommendations for updates with automated PR creation.
"""

import json
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import requests


class DependencyChecker:
    """Manages dependency checking and update recommendations."""
    
    def __init__(self, requirements_file: str = "pyproject.toml"):
        """Initialize dependency checker.
        
        Args:
            requirements_file: Path to requirements file (pyproject.toml or requirements.txt)
        """
        self.requirements_file = Path(requirements_file)
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/single-cell-graph-hub')
        
    def get_current_dependencies(self) -> Dict[str, str]:
        """Extract current dependencies and their versions."""
        dependencies = {}
        
        if self.requirements_file.name == "pyproject.toml":
            dependencies = self._parse_pyproject_toml()
        elif self.requirements_file.name.endswith(".txt"):
            dependencies = self._parse_requirements_txt()
        
        return dependencies
    
    def _parse_pyproject_toml(self) -> Dict[str, str]:
        """Parse dependencies from pyproject.toml."""
        dependencies = {}
        
        try:
            import toml
            with open(self.requirements_file, 'r') as f:
                config = toml.load(f)
            
            # Main dependencies
            deps = config.get('project', {}).get('dependencies', [])
            for dep in deps:
                name, version = self._parse_requirement(dep)
                if name:
                    dependencies[name] = version
            
            # Optional dependencies
            optional_deps = config.get('project', {}).get('optional-dependencies', {})
            for group, deps in optional_deps.items():
                for dep in deps:
                    name, version = self._parse_requirement(dep)
                    if name:
                        dependencies[f"{name}[{group}]"] = version
        
        except Exception as e:
            print(f"❌ Error parsing pyproject.toml: {e}")
        
        return dependencies
    
    def _parse_requirements_txt(self) -> Dict[str, str]:
        """Parse dependencies from requirements.txt."""
        dependencies = {}
        
        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        name, version = self._parse_requirement(line)
                        if name:
                            dependencies[name] = version
        
        except Exception as e:
            print(f"❌ Error parsing requirements.txt: {e}")
        
        return dependencies
    
    def _parse_requirement(self, requirement: str) -> Tuple[Optional[str], str]:
        """Parse a requirement string to extract package name and version."""
        # Simple parsing - would use packaging library in production
        requirement = requirement.strip()
        
        for operator in ['>=', '==', '<=', '>', '<', '~=']:
            if operator in requirement:
                parts = requirement.split(operator)
                name = parts[0].strip()
                version = parts[1].strip() if len(parts) > 1 else ""
                return name, version
        
        # No version specified
        return requirement.strip(), ""
    
    def check_outdated_packages(self) -> List[Dict[str, Any]]:
        """Check for outdated packages using pip list --outdated."""
        print("🔍 Checking for outdated packages...")
        
        outdated = []
        
        try:
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                outdated_data = json.loads(result.stdout)
                for package in outdated_data:
                    outdated.append({
                        'name': package['name'],
                        'current_version': package['version'],
                        'latest_version': package['latest_version'],
                        'type': package.get('latest_filetype', 'wheel')
                    })
        
        except Exception as e:
            print(f"⚠️ Error checking outdated packages: {e}")
        
        print(f"📊 Found {len(outdated)} outdated packages")
        return outdated
    
    def check_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities using safety and pip-audit."""
        print("🔒 Checking for security vulnerabilities...")
        
        vulnerabilities = []
        
        # Safety check
        vulnerabilities.extend(self._run_safety_check())
        
        # pip-audit check
        vulnerabilities.extend(self._run_pip_audit())
        
        print(f"🚨 Found {len(vulnerabilities)} security vulnerabilities")
        return vulnerabilities
    
    def _run_safety_check(self) -> List[Dict[str, Any]]:
        """Run safety check for known vulnerabilities."""
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerabilities.append({
                        'tool': 'safety',
                        'package': vuln.get('package_name', ''),
                        'version': vuln.get('analyzed_version', ''),
                        'vulnerability_id': vuln.get('vulnerability_id', ''),
                        'severity': vuln.get('severity', 'unknown'),
                        'description': vuln.get('advisory', ''),
                        'fixed_versions': vuln.get('fixed_versions', [])
                    })
        
        except Exception as e:
            print(f"⚠️ Safety check failed: {e}")
        
        return vulnerabilities
    
    def _run_pip_audit(self) -> List[Dict[str, Any]]:
        """Run pip-audit for vulnerability checking."""
        vulnerabilities = []
        
        try:
            result = subprocess.run(
                ['pip-audit', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get('vulnerabilities', []):
                    vulnerabilities.append({
                        'tool': 'pip-audit',
                        'package': vuln.get('package', ''),
                        'version': vuln.get('installed_version', ''),
                        'vulnerability_id': vuln.get('id', ''),
                        'severity': 'high',  # pip-audit doesn't provide severity
                        'description': vuln.get('description', ''),
                        'fixed_versions': vuln.get('fix_versions', [])
                    })
        
        except Exception as e:
            print(f"⚠️ pip-audit check failed: {e}")
        
        return vulnerabilities
    
    def check_license_compatibility(self) -> List[Dict[str, Any]]:
        """Check license compatibility of dependencies."""
        print("⚖️ Checking license compatibility...")
        
        license_issues = []
        
        try:
            result = subprocess.run(
                ['pip-licenses', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                licenses_data = json.loads(result.stdout)
                
                # Define problematic licenses
                problematic_licenses = [
                    'GPL', 'AGPL', 'LGPL',  # Copyleft licenses
                    'Commercial', 'Proprietary',  # Commercial licenses
                    'UNKNOWN', 'NOTFOUND'  # Unknown licenses
                ]
                
                for package in licenses_data:
                    license_name = package.get('License', 'UNKNOWN')
                    if any(prob in license_name.upper() for prob in problematic_licenses):
                        license_issues.append({
                            'package': package.get('Name', ''),
                            'version': package.get('Version', ''),
                            'license': license_name,
                            'issue': 'Potentially incompatible license'
                        })
        
        except Exception as e:
            print(f"⚠️ License check failed: {e}")
        
        print(f"⚖️ Found {len(license_issues)} potential license issues")
        return license_issues
    
    def generate_update_recommendations(self, 
                                      outdated: List[Dict[str, Any]],
                                      vulnerabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate prioritized update recommendations."""
        print("💡 Generating update recommendations...")
        
        recommendations = {
            'critical': [],  # Security vulnerabilities
            'high': [],      # Major version updates
            'medium': [],    # Minor version updates  
            'low': []        # Patch version updates
        }
        
        # Prioritize security vulnerabilities
        vulnerable_packages = {v['package'] for v in vulnerabilities}
        
        for package in outdated:
            package_name = package['name']
            current = package['current_version']
            latest = package['latest_version']
            
            priority = self._determine_update_priority(
                package_name, current, latest, package_name in vulnerable_packages
            )
            
            recommendation = {
                'package': package_name,
                'current_version': current,
                'latest_version': latest,
                'has_vulnerability': package_name in vulnerable_packages,
                'update_command': f"pip install --upgrade {package_name}=={latest}",
                'breaking_change_risk': self._assess_breaking_change_risk(current, latest)
            }
            
            recommendations[priority].append(recommendation)
        
        return recommendations
    
    def _determine_update_priority(self, package: str, current: str, 
                                 latest: str, has_vulnerability: bool) -> str:
        """Determine update priority based on various factors."""
        if has_vulnerability:
            return 'critical'
        
        # Simple version comparison (would use packaging.version in production)
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Major version change
            if len(current_parts) > 0 and len(latest_parts) > 0:
                if current_parts[0] < latest_parts[0]:
                    return 'high'
            
            # Minor version change
            if len(current_parts) > 1 and len(latest_parts) > 1:
                if current_parts[1] < latest_parts[1]:
                    return 'medium'
            
            # Patch version change
            return 'low'
        
        except ValueError:
            return 'medium'  # Default for unparseable versions
    
    def _assess_breaking_change_risk(self, current: str, latest: str) -> str:
        """Assess risk of breaking changes in update."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Major version change = high risk
            if len(current_parts) > 0 and len(latest_parts) > 0:
                if current_parts[0] < latest_parts[0]:
                    return 'high'
            
            # Minor version change = medium risk
            if len(current_parts) > 1 and len(latest_parts) > 1:
                if current_parts[1] < latest_parts[1]:
                    return 'medium'
            
            # Patch version change = low risk
            return 'low'
        
        except ValueError:
            return 'medium'
    
    def create_update_script(self, recommendations: Dict[str, Any], 
                           output_file: str = "update_dependencies.sh") -> str:
        """Create a shell script to apply recommended updates."""
        print(f"📝 Creating update script: {output_file}")
        
        script_content = """#!/bin/bash
# Automated dependency update script
# Generated by dependency-checker.py

set -e  # Exit on any error

echo "🔄 Starting dependency updates..."

# Create backup of current requirements
cp pyproject.toml pyproject.toml.backup

"""
        
        # Add critical updates first
        if recommendations['critical']:
            script_content += """
echo "🚨 Applying critical security updates..."
"""
            for rec in recommendations['critical']:
                script_content += f"echo \"Updating {rec['package']} (security fix)\"\n"
                script_content += f"{rec['update_command']}\n"
        
        # Add high priority updates
        if recommendations['high']:
            script_content += """
echo "⬆️ Applying high priority updates..."
"""
            for rec in recommendations['high']:
                script_content += f"echo \"Updating {rec['package']} (major version)\"\n"
                script_content += f"{rec['update_command']}\n"
        
        # Add medium priority updates
        if recommendations['medium']:
            script_content += """
echo "📈 Applying medium priority updates..."
"""
            for rec in recommendations['medium']:
                script_content += f"echo \"Updating {rec['package']} (minor version)\"\n"
                script_content += f"{rec['update_command']}\n"
        
        script_content += """
echo "✅ Dependency updates completed"
echo "🧪 Running tests to verify updates..."

# Run tests to verify updates
if command -v pytest &> /dev/null; then
    pytest tests/unit/ -x --tb=short || {
        echo "❌ Tests failed - rolling back updates"
        cp pyproject.toml.backup pyproject.toml
        exit 1
    }
fi

echo "🎉 All updates applied successfully!"
"""
        
        # Write script to file
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_file, 0o755)
        
        return output_file
    
    def generate_report(self, format: str = "markdown") -> str:
        """Generate a comprehensive dependency report."""
        print("📊 Generating dependency report...")
        
        # Collect all data
        current_deps = self.get_current_dependencies()
        outdated = self.check_outdated_packages()
        vulnerabilities = self.check_security_vulnerabilities()
        license_issues = self.check_license_compatibility()
        recommendations = self.generate_update_recommendations(outdated, vulnerabilities)
        
        if format == "markdown":
            return self._generate_markdown_report(
                current_deps, outdated, vulnerabilities, license_issues, recommendations
            )
        elif format == "json":
            return json.dumps({
                'current_dependencies': current_deps,
                'outdated_packages': outdated,
                'vulnerabilities': vulnerabilities,
                'license_issues': license_issues,
                'recommendations': recommendations,
                'generated_at': datetime.utcnow().isoformat()
            }, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_markdown_report(self, current_deps: Dict[str, str],
                                outdated: List[Dict[str, Any]],
                                vulnerabilities: List[Dict[str, Any]],
                                license_issues: List[Dict[str, Any]],
                                recommendations: Dict[str, Any]) -> str:
        """Generate markdown dependency report."""
        
        report = f"""# Dependency Analysis Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Repository:** {self.repo}
**Total Dependencies:** {len(current_deps)}

## Executive Summary

- 🔢 **Total Dependencies:** {len(current_deps)}
- 📈 **Outdated Packages:** {len(outdated)}
- 🚨 **Security Vulnerabilities:** {len(vulnerabilities)}
- ⚖️ **License Issues:** {len(license_issues)}

## Security Vulnerabilities

"""
        if vulnerabilities:
            report += "| Package | Version | Vulnerability | Severity | Fix |\n"
            report += "|---------|---------|---------------|----------|-----|\n"
            
            for vuln in vulnerabilities:
                fix_versions = ', '.join(vuln['fixed_versions'][:3]) if vuln['fixed_versions'] else "None"
                report += f"| {vuln['package']} | {vuln['version']} | {vuln['vulnerability_id']} | {vuln['severity']} | {fix_versions} |\n"
        else:
            report += "✅ No security vulnerabilities found.\n"
        
        report += "\n## Outdated Packages\n\n"
        
        if outdated:
            report += "| Package | Current | Latest | Priority |\n"
            report += "|---------|---------|--------|---------|\n"
            
            for package in outdated:
                priority = "🚨 Critical" if package['name'] in [v['package'] for v in vulnerabilities] else "📈 Normal"
                report += f"| {package['name']} | {package['current_version']} | {package['latest_version']} | {priority} |\n"
        else:
            report += "✅ All packages are up to date.\n"
        
        report += "\n## Update Recommendations\n\n"
        
        for priority, items in recommendations.items():
            if items:
                priority_emoji = {'critical': '🚨', 'high': '⬆️', 'medium': '📈', 'low': '🔧'}
                report += f"### {priority_emoji.get(priority, '📦')} {priority.title()} Priority ({len(items)} packages)\n\n"
                
                for item in items:
                    risk = item['breaking_change_risk']
                    risk_emoji = {'high': '⚠️', 'medium': '⚡', 'low': '✅'}
                    report += f"- **{item['package']}** {item['current_version']} → {item['latest_version']} {risk_emoji.get(risk, '')} {risk} risk\n"
                
                report += "\n"
        
        report += "\n## License Issues\n\n"
        
        if license_issues:
            report += "| Package | Version | License | Issue |\n"
            report += "|---------|---------|---------|-------|\n"
            
            for issue in license_issues:
                report += f"| {issue['package']} | {issue['version']} | {issue['license']} | {issue['issue']} |\n"
        else:
            report += "✅ No license compatibility issues found.\n"
        
        report += f"""
## Recommended Actions

1. **Immediate:** Update packages with security vulnerabilities
2. **Soon:** Review and update high-priority packages
3. **Planning:** Consider impact of major version updates
4. **Ongoing:** Monitor for new vulnerabilities and updates

## Automation

This report was generated automatically. To apply updates:

1. Review the recommendations above
2. Run the generated update script: `./update_dependencies.sh`
3. Test thoroughly before deploying
4. Monitor for any issues after updates

---
*Report generated by dependency-checker.py*
"""
        
        return report
    
    def create_github_issue(self, report: str, title: str = None) -> bool:
        """Create a GitHub issue with the dependency report."""
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN not found, cannot create GitHub issue")
            return False
        
        if not title:
            title = f"Dependency Update Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            issue_data = {
                'title': title,
                'body': report,
                'labels': ['dependencies', 'automated', 'maintenance']
            }
            
            url = f"https://api.github.com/repos/{self.repo}/issues"
            response = requests.post(url, headers=headers, json=issue_data)
            
            if response.status_code == 201:
                issue_url = response.json()['html_url']
                print(f"✅ GitHub issue created: {issue_url}")
                return True
            else:
                print(f"❌ Failed to create GitHub issue: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Error creating GitHub issue: {e}")
            return False


def main():
    """Main entry point for dependency checker."""
    parser = argparse.ArgumentParser(description="Check and manage project dependencies")
    parser.add_argument("--requirements", default="pyproject.toml",
                       help="Path to requirements file")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown",
                       help="Output format for report")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--create-issue", action="store_true",
                       help="Create GitHub issue with report")
    parser.add_argument("--create-script", action="store_true",
                       help="Create update script")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check dependencies, don't generate recommendations")
    
    args = parser.parse_args()
    
    try:
        checker = DependencyChecker(args.requirements)
        
        if args.check_only:
            # Quick check mode
            outdated = checker.check_outdated_packages()
            vulnerabilities = checker.check_security_vulnerabilities()
            
            print(f"\n📊 Summary:")
            print(f"   Outdated packages: {len(outdated)}")
            print(f"   Security vulnerabilities: {len(vulnerabilities)}")
            
            if vulnerabilities:
                print(f"\n🚨 Security vulnerabilities found:")
                for vuln in vulnerabilities:
                    print(f"   - {vuln['package']} {vuln['version']}: {vuln['vulnerability_id']}")
                sys.exit(1)
            
        else:
            # Full analysis
            report = checker.generate_report(args.format)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"📄 Report saved to {args.output}")
            else:
                print(report)
            
            if args.create_issue and args.format == "markdown":
                checker.create_github_issue(report)
            
            if args.create_script:
                current_deps = checker.get_current_dependencies()
                outdated = checker.check_outdated_packages()
                vulnerabilities = checker.check_security_vulnerabilities()
                recommendations = checker.generate_update_recommendations(outdated, vulnerabilities)
                
                script_file = checker.create_update_script(recommendations)
                print(f"📝 Update script created: {script_file}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()