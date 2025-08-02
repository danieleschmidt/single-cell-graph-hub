# Security Policy

## Supported Versions

We actively maintain security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of Single-Cell Graph Hub seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing: **security@scgraphhub.org**

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on severity)

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Investigation**: Our security team will investigate the issue
3. **Updates**: Regular status updates during investigation
4. **Resolution**: Coordinated disclosure once fixed
5. **Recognition**: Credit in our security advisories (if desired)

## Security Measures

### Data Protection
- No collection of personal or sensitive biological data
- Local data processing by default
- Encryption recommendations for data at rest
- Secure data transmission protocols

### Code Security
- Regular dependency vulnerability scans
- Static code analysis for security issues
- Input validation and sanitization
- Secure coding practices

### Infrastructure Security
- Secure CI/CD pipelines
- Regular security updates
- Access control and authentication
- Logging and monitoring

## Best Practices for Users

### Data Handling
- Never commit sensitive data to version control
- Use environment variables for API keys and secrets
- Implement proper access controls for datasets
- Follow institutional data governance policies

### Package Security
- Always install from official sources (PyPI)
- Verify package integrity when possible
- Keep dependencies updated
- Use virtual environments

### Model Security
- Validate model inputs and outputs
- Be cautious with pre-trained models from unknown sources
- Monitor for adversarial inputs
- Implement proper error handling

## Vulnerability Disclosure Policy

We believe in coordinated disclosure:

1. **Private Disclosure**: Report directly to our security team
2. **Investigation Period**: Allow reasonable time for investigation
3. **Coordinated Release**: Work together on disclosure timeline
4. **Public Disclosure**: After fix is available and deployed

## Security Updates

Security updates will be:
- Released as soon as possible after confirmation
- Clearly marked in release notes
- Communicated through security advisories
- Available for all supported versions

## Contact Information

- **Security Email**: security@scgraphhub.org
- **General Contact**: hello@scgraphhub.org
- **GitHub Security**: Use GitHub's private vulnerability reporting

## Additional Resources

- [GitHub Security Advisories](https://github.com/yourusername/single-cell-graph-hub/security/advisories)
- [Python Security Guidelines](https://python.org/dev/security/)
- [PyTorch Security](https://pytorch.org/docs/stable/community/security.html)

## Acknowledgments

We thank the security research community for helping keep Single-Cell Graph Hub safe for everyone.

---

**Last Updated**: January 2025  
**Next Review**: April 2025