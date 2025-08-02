# CI/CD Workflows Documentation

This directory contains comprehensive documentation and templates for GitHub Actions workflows. Due to GitHub App permission limitations, these files serve as templates that repository maintainers must manually implement.

## Required Manual Setup

**⚠️ IMPORTANT**: Repository maintainers must manually create these workflow files from the templates provided in the `examples/` directory due to GitHub App permission restrictions.

### Setup Instructions

1. **Create `.github/workflows/` directory** in the repository root
2. **Copy workflow files** from `docs/workflows/examples/` to `.github/workflows/`
3. **Configure secrets** in repository settings
4. **Enable branch protection rules** as documented below
5. **Test workflows** with pull requests

## Workflow Overview

Our CI/CD pipeline implements a comprehensive software development lifecycle with multiple stages:

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Pull Request  │───▶│  CI Pipeline │───▶│ Security Checks │
│   (Trigger)     │    │  (Testing)   │    │  (SAST/DAST)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Code Quality   │◀───│    Merge     │───▶│   Deployment    │
│   (Linting)     │    │   (main)     │    │  (Staging/Prod) │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## Workflow Files

### Core Workflows

1. **`ci.yml`** - Pull Request validation and testing
2. **`cd.yml`** - Continuous deployment for main branch
3. **`security-scan.yml`** - Comprehensive security scanning
4. **`dependency-update.yml`** - Automated dependency management

### Specialized Workflows

5. **`release.yml`** - Automated release creation and publishing
6. **`docs.yml`** - Documentation building and deployment
7. **`performance.yml`** - Performance testing and benchmarking
8. **`cleanup.yml`** - Repository cleanup and maintenance

## Workflow Triggers

### Pull Request Events
- **opened**: Run full CI pipeline
- **synchronize**: Re-run tests on new commits
- **ready_for_review**: Run security scans

### Push Events
- **main branch**: Deploy to staging, run security scans
- **tags**: Create releases, deploy to production

### Scheduled Events
- **Daily**: Dependency updates, security scans
- **Weekly**: Performance benchmarks, cleanup tasks

### Manual Events
- **workflow_dispatch**: Manual workflow triggers with parameters

## Security Configuration

### Required Secrets

Add these secrets in repository settings (`Settings > Secrets and variables > Actions`):

```bash
# PyPI Publishing
PYPI_TOKEN=pypi-xxxxx

# Docker Registry
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password
DOCKER_REGISTRY=ghcr.io

# Cloud Deployment (AWS)
AWS_ACCESS_KEY_ID=AKIAXXXXX
AWS_SECRET_ACCESS_KEY=xxxxx
AWS_DEFAULT_REGION=us-west-2

# Security Scanning
SNYK_TOKEN=xxxxx
SONAR_TOKEN=xxxxx

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/xxxxx
```

### Branch Protection Rules

Configure branch protection for `main` branch:

```yaml
# Required status checks
required_status_checks:
  - ci-tests
  - security-scan
  - code-quality

# Required reviews
required_pull_request_reviews:
  required_approving_review_count: 1
  dismiss_stale_reviews: true
  require_code_owner_reviews: true

# Additional restrictions
enforce_admins: true
allow_force_pushes: false
allow_deletions: false
```

## Environment Configuration

### Staging Environment
- **URL**: https://staging.scgraphhub.org
- **Deploy on**: Push to `main` branch
- **Auto-deploy**: Enabled with approval gates

### Production Environment
- **URL**: https://scgraphhub.org
- **Deploy on**: Tagged releases
- **Auto-deploy**: Manual approval required

### Review Environments
- **URL**: https://pr-{number}.scgraphhub.org
- **Deploy on**: Pull requests (for testing)
- **Cleanup**: Automatic after PR merge/close

## Workflow Features

### Parallel Execution
- **Test Matrix**: Multiple Python versions, OS combinations
- **Dependency Caching**: Faster builds with cached dependencies
- **Artifact Sharing**: Efficient artifact passing between jobs

### Quality Gates
- **Test Coverage**: Minimum 80% coverage required
- **Code Quality**: No critical issues allowed
- **Security**: No high/critical vulnerabilities
- **Performance**: No regression in key metrics

### Notifications
- **Slack Integration**: Build status notifications
- **Email Alerts**: Failure notifications to maintainers
- **PR Comments**: Automated feedback on pull requests

## Monitoring and Observability

### Workflow Metrics
- **Build Duration**: Track build time trends
- **Success Rate**: Monitor pipeline reliability
- **Failure Analysis**: Automated failure categorization

### Integration Monitoring
- **Deployment Health**: Post-deployment verification
- **Performance Impact**: Monitor performance after deployments
- **Security Posture**: Continuous security monitoring

## Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Check workflow logs
   gh run list --workflow=ci.yml
   gh run view <run-id>
   
   # Re-run failed jobs
   gh run rerun <run-id>
   ```

2. **Security Scan Failures**
   ```bash
   # Check security scan results
   gh run view <run-id> --log
   
   # Download security artifacts
   gh run download <run-id>
   ```

3. **Deployment Issues**
   ```bash
   # Check deployment logs
   kubectl logs -f deployment/scgraph-hub -n production
   
   # Rollback if needed
   kubectl rollout undo deployment/scgraph-hub -n production
   ```

### Debug Mode

Enable debug logging by setting `ACTIONS_STEP_DEBUG` secret to `true`:

```yaml
env:
  ACTIONS_STEP_DEBUG: ${{ secrets.ACTIONS_STEP_DEBUG }}
```

## Performance Optimization

### Caching Strategy
- **Dependencies**: Cache pip, Docker layers
- **Build Artifacts**: Cache build outputs
- **Test Data**: Cache test datasets

### Resource Limits
- **Concurrent Jobs**: Limit to prevent resource exhaustion
- **Timeout Settings**: Prevent hanging jobs
- **Resource Quotas**: Manage GitHub Actions minutes

### Build Optimization
- **Multi-stage Builds**: Efficient Docker builds
- **Parallel Testing**: Run tests in parallel
- **Selective Execution**: Skip unnecessary jobs

## Compliance and Governance

### Audit Trail
- **Workflow History**: Complete execution history
- **Change Tracking**: Git-based workflow versioning
- **Approval Records**: Deployment approval tracking

### Security Compliance
- **SLSA Level 3**: Build provenance and integrity
- **SBOM Generation**: Software bill of materials
- **Vulnerability Scanning**: Continuous security assessment

### Quality Assurance
- **Code Reviews**: Mandatory for workflow changes
- **Testing**: Comprehensive workflow testing
- **Documentation**: Complete workflow documentation

## Migration Guide

### From Other CI Systems

#### Jenkins
```bash
# Convert Jenkinsfile to GitHub Actions
# Use GitHub's importer or manual conversion
# Reference: https://docs.github.com/en/actions/migrating-to-github-actions/migrating-from-jenkins-to-github-actions
```

#### GitLab CI
```bash
# Convert .gitlab-ci.yml to GitHub Actions
# Use conversion tools or migrate step-by-step
# Reference: https://docs.github.com/en/actions/migrating-to-github-actions/migrating-from-gitlab-cicd-to-github-actions  
```

### Best Practices for Migration
1. **Incremental Migration**: Migrate workflows gradually
2. **Parallel Running**: Run both systems during transition
3. **Testing**: Thoroughly test migrated workflows
4. **Documentation**: Update all documentation

## Support and Resources

### Documentation Links
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides)

### Community Resources
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)
- [Actions Toolkit](https://github.com/actions/toolkit)

### Getting Help
1. **Check workflow logs** for detailed error messages
2. **Review documentation** for syntax and configuration
3. **Search GitHub Community** for similar issues
4. **Open GitHub Support ticket** for platform issues

## Workflow Templates

All workflow templates are available in the `examples/` directory:

- `ci.yml` - Continuous Integration
- `cd.yml` - Continuous Deployment  
- `security-scan.yml` - Security Scanning
- `dependency-update.yml` - Dependency Updates
- `release.yml` - Release Management
- `docs.yml` - Documentation
- `performance.yml` - Performance Testing
- `cleanup.yml` - Repository Maintenance

**Remember**: These templates must be manually copied to `.github/workflows/` to be active.