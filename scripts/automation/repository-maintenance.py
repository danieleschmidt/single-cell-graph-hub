#!/usr/bin/env python3
"""
Repository maintenance automation script.

This script performs various maintenance tasks including:
- Cleaning up old branches
- Updating documentation
- Checking repository health
- Managing releases
- Optimizing repository structure
"""

import json
import os
import sys
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import requests


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self):
        """Initialize repository maintainer."""
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/single-cell-graph-hub')
        self.repo_path = Path('.')
        
    def cleanup_old_branches(self, days_old: int = 30, dry_run: bool = True) -> List[str]:
        """Clean up old merged branches.
        
        Args:
            days_old: Remove branches older than this many days
            dry_run: If True, only show what would be deleted
            
        Returns:
            List of branches that were (or would be) deleted
        """
        print(f"🧹 Cleaning up branches older than {days_old} days...")
        
        deleted_branches = []
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            # Get list of merged branches
            result = subprocess.run(
                ['git', 'branch', '-r', '--merged', 'main'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print("❌ Failed to get merged branches")
                return deleted_branches
            
            merged_branches = [
                branch.strip().replace('origin/', '') 
                for branch in result.stdout.strip().split('\n')
                if branch.strip() and 'origin/' in branch and 'main' not in branch
            ]
            
            for branch in merged_branches:
                # Get last commit date for branch
                try:
                    date_result = subprocess.run(
                        ['git', 'log', '-1', '--format=%ci', f'origin/{branch}'],
                        capture_output=True,
                        text=True
                    )
                    
                    if date_result.returncode == 0:
                        last_commit_str = date_result.stdout.strip()
                        last_commit_date = datetime.fromisoformat(last_commit_str.replace(' +', '+'))
                        
                        if last_commit_date < cutoff_date:
                            if dry_run:
                                print(f"   Would delete: {branch} (last commit: {last_commit_date.date()})")
                            else:
                                # Delete remote branch
                                delete_result = subprocess.run(
                                    ['git', 'push', 'origin', '--delete', branch],
                                    capture_output=True,
                                    text=True
                                )
                                
                                if delete_result.returncode == 0:
                                    print(f"   ✅ Deleted: {branch}")
                                    deleted_branches.append(branch)
                                else:
                                    print(f"   ❌ Failed to delete: {branch}")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing branch {branch}: {e}")
        
        except Exception as e:
            print(f"❌ Error during branch cleanup: {e}")
        
        if dry_run:
            print(f"🔍 Dry run complete. {len(deleted_branches)} branches would be deleted.")
        else:
            print(f"✅ Branch cleanup complete. {len(deleted_branches)} branches deleted.")
        
        return deleted_branches
    
    def update_readme_badges(self) -> bool:
        """Update README badges with current information."""
        print("🏷️ Updating README badges...")
        
        readme_path = Path('README.md')
        if not readme_path.exists():
            print("❌ README.md not found")
            return False
        
        try:
            with open(readme_path, 'r') as f:
                content = f.read()
            
            # Define badge templates
            badges = {
                'version': f'![Version](https://img.shields.io/pypi/v/single-cell-graph-hub.svg)',
                'python': f'![Python](https://img.shields.io/pypi/pyversions/single-cell-graph-hub.svg)',
                'license': f'![License](https://img.shields.io/github/license/{self.repo}.svg)',
                'build': f'![Build](https://img.shields.io/github/actions/workflow/status/{self.repo}/ci.yml?branch=main)',
                'coverage': f'![Coverage](https://img.shields.io/codecov/c/github/{self.repo}.svg)',
                'downloads': f'![Downloads](https://img.shields.io/pypi/dm/single-cell-graph-hub.svg)',
                'stars': f'![Stars](https://img.shields.io/github/stars/{self.repo}.svg?style=social)',
            }
            
            # Update badges section (assumes badges are at the top)
            lines = content.split('\n')
            updated_lines = []
            in_badges_section = False
            
            for line in lines:
                if line.startswith('![') and any(badge in line for badge in ['Version', 'Python', 'License', 'Build']):
                    if not in_badges_section:
                        in_badges_section = True
                        # Add all badges
                        updated_lines.extend([
                            badges['version'],
                            badges['python'], 
                            badges['license'],
                            badges['build'],
                            badges['coverage'],
                            badges['downloads'],
                            badges['stars'],
                            ''  # Empty line after badges
                        ])
                    # Skip existing badge lines
                    continue
                elif in_badges_section and line.strip() == '':
                    in_badges_section = False
                    continue  # Skip empty line after badges (already added)
                else:
                    updated_lines.append(line)
            
            # Write updated content
            with open(readme_path, 'w') as f:
                f.write('\n'.join(updated_lines))
            
            print("✅ README badges updated")
            return True
        
        except Exception as e:
            print(f"❌ Error updating README badges: {e}")
            return False
    
    def check_repository_health(self) -> Dict[str, Any]:
        """Check overall repository health and provide recommendations."""
        print("🏥 Checking repository health...")
        
        health_report = {
            'score': 0,
            'max_score': 100,
            'checks': {},
            'recommendations': []
        }
        
        # Check 1: README exists and is comprehensive (10 points)
        readme_path = Path('README.md')
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                readme_content = f.read()
                readme_length = len(readme_content)
                
            if readme_length > 1000:
                health_report['checks']['readme'] = {'score': 10, 'status': 'excellent'}
            elif readme_length > 500:
                health_report['checks']['readme'] = {'score': 7, 'status': 'good'}
            else:
                health_report['checks']['readme'] = {'score': 3, 'status': 'basic'}
                health_report['recommendations'].append("Expand README with more detailed documentation")
        else:
            health_report['checks']['readme'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Create comprehensive README.md")
        
        # Check 2: License exists (10 points)
        license_files = list(Path('.').glob('LICENSE*'))
        if license_files:
            health_report['checks']['license'] = {'score': 10, 'status': 'present'}
        else:
            health_report['checks']['license'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Add LICENSE file")
        
        # Check 3: Contributing guidelines (10 points)
        contributing_files = list(Path('.').glob('CONTRIBUTING*'))
        if contributing_files:
            health_report['checks']['contributing'] = {'score': 10, 'status': 'present'}
        else:
            health_report['checks']['contributing'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Add CONTRIBUTING.md guidelines")
        
        # Check 4: Code of Conduct (5 points)
        coc_files = list(Path('.').glob('CODE_OF_CONDUCT*'))
        if coc_files:
            health_report['checks']['code_of_conduct'] = {'score': 5, 'status': 'present'}
        else:
            health_report['checks']['code_of_conduct'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Add CODE_OF_CONDUCT.md")
        
        # Check 5: Tests exist (15 points)
        test_dirs = [Path('tests'), Path('test')]
        test_files = []
        for test_dir in test_dirs:
            if test_dir.exists():
                test_files.extend(list(test_dir.rglob('test_*.py')))
                test_files.extend(list(test_dir.rglob('*_test.py')))
        
        if len(test_files) >= 10:
            health_report['checks']['tests'] = {'score': 15, 'status': 'comprehensive'}
        elif len(test_files) >= 5:
            health_report['checks']['tests'] = {'score': 10, 'status': 'good'}
        elif len(test_files) > 0:
            health_report['checks']['tests'] = {'score': 5, 'status': 'basic'}
        else:
            health_report['checks']['tests'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Add comprehensive test suite")
        
        # Check 6: CI/CD pipeline (15 points)
        github_workflows = Path('.github/workflows')
        if github_workflows.exists():
            workflow_files = list(github_workflows.glob('*.yml')) + list(github_workflows.glob('*.yaml'))
            if len(workflow_files) >= 3:
                health_report['checks']['cicd'] = {'score': 15, 'status': 'comprehensive'}
            elif len(workflow_files) >= 1:
                health_report['checks']['cicd'] = {'score': 10, 'status': 'basic'}
        else:
            health_report['checks']['cicd'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Set up CI/CD pipeline with GitHub Actions")
        
        # Check 7: Documentation (10 points)
        docs_dir = Path('docs')
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob('*.md')) + list(docs_dir.rglob('*.rst'))
            if len(doc_files) >= 5:
                health_report['checks']['documentation'] = {'score': 10, 'status': 'comprehensive'}
            elif len(doc_files) >= 2:
                health_report['checks']['documentation'] = {'score': 7, 'status': 'good'}
            else:
                health_report['checks']['documentation'] = {'score': 3, 'status': 'basic'}
        else:
            health_report['checks']['documentation'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Create comprehensive documentation")
        
        # Check 8: Dependencies management (10 points)
        dep_files = ['pyproject.toml', 'requirements.txt', 'Pipfile', 'poetry.lock']
        dep_file_exists = any(Path(f).exists() for f in dep_files)
        if dep_file_exists:
            health_report['checks']['dependencies'] = {'score': 10, 'status': 'managed'}
        else:
            health_report['checks']['dependencies'] = {'score': 0, 'status': 'missing'}
            health_report['recommendations'].append("Add proper dependency management")
        
        # Check 9: Security (10 points)
        security_files = list(Path('.').glob('SECURITY*'))
        if security_files:
            health_report['checks']['security'] = {'score': 10, 'status': 'documented'}
        else:
            health_report['checks']['security'] = {'score': 5, 'status': 'basic'}
            health_report['recommendations'].append("Add SECURITY.md with vulnerability reporting process")
        
        # Check 10: Release management (5 points)
        changelog_files = list(Path('.').glob('CHANGELOG*')) + list(Path('.').glob('HISTORY*'))
        if changelog_files:
            health_report['checks']['releases'] = {'score': 5, 'status': 'tracked'}
        else:
            health_report['checks']['releases'] = {'score': 0, 'status': 'untracked'}
            health_report['recommendations'].append("Add CHANGELOG.md to track releases")
        
        # Calculate total score
        total_score = sum(check['score'] for check in health_report['checks'].values())
        health_report['score'] = total_score
        
        # Determine health level
        if total_score >= 80:
            health_level = "Excellent"
        elif total_score >= 60:
            health_level = "Good"
        elif total_score >= 40:
            health_level = "Fair"
        else:
            health_level = "Needs Improvement"
        
        health_report['level'] = health_level
        
        print(f"📊 Repository health: {health_level} ({total_score}/{health_report['max_score']})")
        
        return health_report
    
    def optimize_repository_structure(self) -> List[str]:
        """Optimize repository structure and organization."""
        print("🏗️ Optimizing repository structure...")
        
        optimizations = []
        
        # Create standard directories if they don't exist
        standard_dirs = [
            'docs',
            'tests',
            'scripts',
            '.github/workflows',
            '.github/ISSUE_TEMPLATE'
        ]
        
        for dir_path in standard_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                optimizations.append(f"Created directory: {dir_path}")
        
        # Move misplaced files
        misplaced_files = {
            'test_*.py': 'tests/',
            '*_test.py': 'tests/',
            'docs/*.py': 'scripts/',
        }
        
        for pattern, target_dir in misplaced_files.items():
            files = list(Path('.').glob(pattern))
            for file in files:
                if not str(file).startswith(target_dir):
                    target_path = Path(target_dir) / file.name
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file), str(target_path))
                    optimizations.append(f"Moved {file} to {target_path}")
        
        # Clean up empty directories
        for root, dirs, files in os.walk('.', topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    try:
                        dir_path.rmdir()
                        optimizations.append(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass  # Directory not empty or permission issue
        
        print(f"✅ Repository structure optimized ({len(optimizations)} changes)")
        return optimizations
    
    def update_github_topics(self, topics: List[str]) -> bool:
        """Update GitHub repository topics."""
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN not found, cannot update topics")
            return False
        
        print(f"🏷️ Updating GitHub topics: {', '.join(topics)}")
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.mercy-preview+json'
            }
            
            data = {'names': topics}
            
            url = f"https://api.github.com/repos/{self.repo}/topics"
            response = requests.put(url, headers=headers, json=data)
            
            if response.status_code == 200:
                print("✅ GitHub topics updated successfully")
                return True
            else:
                print(f"❌ Failed to update topics: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Error updating GitHub topics: {e}")
            return False
    
    def generate_maintenance_report(self) -> str:
        """Generate a comprehensive maintenance report."""
        print("📋 Generating maintenance report...")
        
        # Collect all maintenance information
        health_report = self.check_repository_health()
        
        report = f"""# Repository Maintenance Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Repository:** {self.repo}

## Health Score: {health_report['score']}/{health_report['max_score']} ({health_report['level']})

### Health Checks

"""
        
        for check_name, check_data in health_report['checks'].items():
            status_emoji = {
                'excellent': '🟢',
                'good': '🟡', 
                'basic': '🟠',
                'present': '✅',
                'missing': '❌',
                'comprehensive': '🟢',
                'managed': '✅',
                'documented': '✅',
                'tracked': '✅',
                'untracked': '❌'
            }
            
            emoji = status_emoji.get(check_data['status'], '⚪')
            report += f"- {emoji} **{check_name.replace('_', ' ').title()}**: {check_data['score']} points ({check_data['status']})\n"
        
        report += "\n### Recommendations\n\n"
        
        if health_report['recommendations']:
            for i, recommendation in enumerate(health_report['recommendations'], 1):
                report += f"{i}. {recommendation}\n"
        else:
            report += "🎉 No recommendations - repository is in excellent condition!\n"
        
        report += f"""
## Repository Statistics

- **Total Files:** {len(list(Path('.').rglob('*')))}
- **Python Files:** {len(list(Path('.').rglob('*.py')))}
- **Documentation Files:** {len(list(Path('.').rglob('*.md')) + list(Path('.').rglob('*.rst')))}
- **Test Files:** {len(list(Path('.').rglob('test_*.py')) + list(Path('.').rglob('*_test.py')))}

## Maintenance Actions Suggested

1. **Immediate** (Critical for repository health):
   - Fix any missing essential files (LICENSE, README, etc.)
   - Address security vulnerabilities
   
2. **Short-term** (Improve developer experience):
   - Enhance documentation
   - Improve test coverage
   - Set up automated workflows

3. **Long-term** (Optimize and maintain):
   - Regular dependency updates
   - Performance optimizations
   - Community building

---
*Report generated by repository maintenance automation*
"""
        
        return report
    
    def create_maintenance_issue(self, report: str) -> bool:
        """Create a GitHub issue with maintenance recommendations."""
        if not self.github_token:
            print("⚠️ GITHUB_TOKEN not found, cannot create maintenance issue")
            return False
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            title = f"Repository Maintenance Report - {datetime.utcnow().strftime('%Y-%m-%d')}"
            
            issue_data = {
                'title': title,
                'body': report,
                'labels': ['maintenance', 'automated', 'housekeeping']
            }
            
            url = f"https://api.github.com/repos/{self.repo}/issues"
            response = requests.post(url, headers=headers, json=issue_data)
            
            if response.status_code == 201:
                issue_url = response.json()['html_url']
                print(f"✅ Maintenance issue created: {issue_url}")
                return True
            else:
                print(f"❌ Failed to create maintenance issue: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ Error creating maintenance issue: {e}")
            return False


def main():
    """Main entry point for repository maintenance."""
    parser = argparse.ArgumentParser(description="Automated repository maintenance")
    parser.add_argument("--cleanup-branches", type=int, metavar="DAYS",
                       help="Clean up merged branches older than DAYS")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--update-badges", action="store_true",
                       help="Update README badges")
    parser.add_argument("--health-check", action="store_true",
                       help="Check repository health")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize repository structure")
    parser.add_argument("--topics", nargs='+',
                       help="Update GitHub repository topics")
    parser.add_argument("--full-maintenance", action="store_true",
                       help="Run all maintenance tasks")
    parser.add_argument("--create-issue", action="store_true",
                       help="Create GitHub issue with maintenance report")
    parser.add_argument("--output", help="Output file for maintenance report")
    
    args = parser.parse_args()
    
    maintainer = RepositoryMaintainer()
    
    try:
        maintenance_performed = []
        
        if args.cleanup_branches is not None or args.full_maintenance:
            days = args.cleanup_branches or 30
            deleted_branches = maintainer.cleanup_old_branches(days, args.dry_run)
            maintenance_performed.append(f"Branch cleanup: {len(deleted_branches)} branches processed")
        
        if args.update_badges or args.full_maintenance:
            if maintainer.update_readme_badges():
                maintenance_performed.append("README badges updated")
        
        if args.health_check or args.full_maintenance:
            health_report = maintainer.check_repository_health()
            maintenance_performed.append(f"Health check: {health_report['level']} ({health_report['score']}/100)")
        
        if args.optimize or args.full_maintenance:
            optimizations = maintainer.optimize_repository_structure()
            maintenance_performed.append(f"Structure optimization: {len(optimizations)} changes")
        
        if args.topics:
            if maintainer.update_github_topics(args.topics):
                maintenance_performed.append(f"Topics updated: {', '.join(args.topics)}")
        
        if args.full_maintenance or args.health_check or args.create_issue:
            report = maintainer.generate_maintenance_report()
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"📄 Maintenance report saved to {args.output}")
            
            if args.create_issue:
                maintainer.create_maintenance_issue(report)
            
            if not args.output and not args.create_issue:
                print("\n" + "="*60)
                print("MAINTENANCE REPORT")
                print("="*60)
                print(report)
        
        # Summary
        if maintenance_performed:
            print(f"\n✅ Maintenance completed:")
            for task in maintenance_performed:
                print(f"   - {task}")
        else:
            print("ℹ️ No maintenance tasks specified. Use --help for options.")
    
    except Exception as e:
        print(f"❌ Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()