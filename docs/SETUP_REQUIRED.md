# Manual Setup Required

This document outlines the manual setup steps required to complete the Single-Cell Graph Hub SDLC implementation, as certain operations require permissions not available to the automated setup process.

## GitHub Workflows Setup

### Required Actions

**⚠️ CRITICAL**: The following GitHub workflows must be manually created due to GitHub App permission limitations:

#### 1. Copy Workflow Files

Copy all workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

#### 2. Required Secrets Configuration

Add the following secrets in **Repository Settings > Secrets and variables > Actions**:

##### Package Publishing
```
PYPI_TOKEN=pypi-xxxxx
DOCKER_USERNAME=your-docker-username  
DOCKER_PASSWORD=your-docker-token
```

##### Security Scanning
```
SNYK_TOKEN=your-snyk-token
SEMGREP_APP_TOKEN=your-semgrep-token
CODECOV_TOKEN=your-codecov-token
```

##### Cloud Deployment (if applicable)
```
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_DEFAULT_REGION=us-west-2
```

##### Notifications
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/your-webhook
SECURITY_SLACK_WEBHOOK_URL=https://hooks.slack.com/security-webhook
```

#### 3. Branch Protection Rules

Configure branch protection for `main` branch in **Settings > Branches**:

- **Require a pull request before merging**: ✅
  - Require approvals: 1
  - Dismiss stale PR approvals when new commits are pushed: ✅
  - Require review from code owners: ✅

- **Require status checks to pass before merging**: ✅
  - Require branches to be up to date before merging: ✅
  - Status checks that are required:
    - `ci-tests`
    - `security-scan`
    - `code-quality`

- **Require conversation resolution before merging**: ✅
- **Require signed commits**: ✅ (recommended)
- **Require linear history**: ✅ (recommended)
- **Do not allow bypassing the above settings**: ✅
- **Restrict pushes that create files larger than 100MB**: ✅

### Workflow Files to Create

The following workflow files need to be manually created in `.github/workflows/`:

1. **`ci.yml`** - Continuous Integration pipeline
2. **`security-scan.yml`** - Comprehensive security scanning
3. **`dependency-update.yml`** - Automated dependency management
4. **`cd.yml`** - Continuous deployment (template in docs/workflows/examples/)
5. **`release.yml`** - Release automation (template in docs/workflows/examples/)

## Repository Settings Configuration

### General Settings

In **Settings > General**:

- **Default branch**: `main`
- **Allow merge commits**: ✅
- **Allow squash merging**: ✅ 
- **Allow rebase merging**: ❌
- **Automatically delete head branches**: ✅

### Code Security and Analysis

Enable the following in **Settings > Code security and analysis**:

- **Dependency graph**: ✅
- **Dependabot alerts**: ✅
- **Dependabot security updates**: ✅
- **Dependabot version updates**: ✅
- **Code scanning**: ✅
- **Secret scanning**: ✅

### CODEOWNERS File

Create `.github/CODEOWNERS` file:

```
# Global owners
* @danieleschmidt

# Documentation
/docs/ @danieleschmidt
/README.md @danieleschmidt

# Security-sensitive files
/docker/ @danieleschmidt
/.github/ @danieleschmidt
/scripts/ @danieleschmidt

# Core source code
/src/ @danieleschmidt
/tests/ @danieleschmidt

# Configuration files
*.yml @danieleschmidt
*.yaml @danieleschmidt
pyproject.toml @danieleschmidt
Dockerfile @danieleschmidt
docker-compose.yml @danieleschmidt
```

## Issue and PR Templates

### Issue Templates

Create `.github/ISSUE_TEMPLATE/` directory with these files:

#### Bug Report Template
Create `.github/ISSUE_TEMPLATE/bug_report.yml`:

```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  
  - type: input
    id: contact
    attributes:
      label: Contact Details
      description: How can we get in touch with you if we need more info?
      placeholder: ex. email@example.com
    validations:
      required: false
  
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Please provide step-by-step instructions
      placeholder: |
        1. Install package with `pip install single-cell-graph-hub`
        2. Run the following code...
        3. See error
    validations:
      required: true
  
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: Please provide your environment details
      placeholder: |
        - OS: [e.g. Ubuntu 20.04]
        - Python version: [e.g. 3.11]
        - Package version: [e.g. 0.1.0]
        - PyTorch version: [e.g. 2.0.0]
    validations:
      required: true
  
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code, so no need for backticks.
      render: shell
```

#### Feature Request Template
Create `.github/ISSUE_TEMPLATE/feature_request.yml`:

```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of what the problem is.
      placeholder: I'm always frustrated when...
    validations:
      required: true
  
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear and concise description of what you want to happen.
    validations:
      required: true
  
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear and concise description of any alternative solutions or features you've considered.
  
  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Add any other context or screenshots about the feature request here.
```

### Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of the changes in this PR.

## Type of Change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Testing

- [ ] Unit tests pass locally
- [ ] Integration tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)

Add screenshots to help explain your changes.

## Additional Notes

Any additional information that reviewers should know.
```

## Environment Setup

### Required Environment Variables

Add these to your local development environment:

```bash
# Development
export SCGRAPH_DEV_MODE=true
export LOG_LEVEL=DEBUG

# Database (for local development)
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/scgraph_hub
export REDIS_URL=redis://localhost:6379/0

# Testing
export PYTEST_TIMEOUT=300
export SKIP_GPU_TESTS=true  # if no GPU available
```

### Local Development Setup

1. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

2. **Set up development database**:
   ```bash
   docker-compose up -d postgres redis
   ```

3. **Run initial tests**:
   ```bash
   make test-fast
   ```

## Documentation Deployment

### GitHub Pages Setup

If you want to deploy documentation to GitHub Pages:

1. Go to **Settings > Pages**
2. Select **Deploy from a branch**
3. Choose **gh-pages** branch (will be created by docs workflow)
4. Select **/ (root)** as the folder

## Monitoring Integration

### External Service Setup

For production monitoring, set up accounts and add tokens:

1. **Sentry** (Error tracking):
   - Create account at sentry.io
   - Add `SENTRY_DSN` secret

2. **DataDog** (Metrics):
   - Create account at datadoghq.com
   - Add `DATADOG_API_KEY` secret

3. **Snyk** (Security scanning):
   - Create account at snyk.io
   - Add `SNYK_TOKEN` secret

## Verification Steps

After completing the manual setup:

1. **Test workflows**:
   ```bash
   # Create a test PR to verify CI pipeline
   git checkout -b test-workflows
   echo "# Test" >> test-file.md
   git add test-file.md && git commit -m "test: verify workflows"
   git push origin test-workflows
   # Create PR and verify all checks pass
   ```

2. **Test security scanning**:
   ```bash
   # Manually trigger security workflow
   gh workflow run security-scan.yml
   ```

3. **Verify branch protection**:
   ```bash
   # Try to push directly to main (should fail)
   git checkout main
   echo "test" >> README.md
   git add README.md && git commit -m "test direct push"
   git push origin main  # This should be rejected
   ```

## Troubleshooting

### Common Issues

1. **Workflow not running**: Check that files are in `.github/workflows/` and secrets are configured
2. **Branch protection blocking PRs**: Verify required status checks match workflow job names
3. **Security scans failing**: Ensure security tokens are valid and have appropriate permissions

### Getting Help

If you encounter issues during setup:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow logs in the Actions tab
3. Open an issue with the specific error message

## Post-Setup Recommendations

1. **Test the full pipeline** with a sample PR
2. **Configure notification channels** for your team
3. **Review and adjust** security scan thresholds
4. **Set up monitoring dashboards** for workflow metrics
5. **Document any custom configurations** for your team

---

**Note**: This setup is required for the SDLC implementation to be fully functional. The automated checkpointed process has prepared all templates and documentation, but these manual steps are necessary due to GitHub permission limitations.