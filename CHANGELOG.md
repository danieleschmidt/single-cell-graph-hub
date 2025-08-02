# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project foundation and documentation structure
- Architecture decision records (ADRs)
- Project charter and roadmap
- Security policy and vulnerability reporting process

### Changed
- Updated README with comprehensive usage examples

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Established security policy and reporting procedures

## [0.1.0] - 2025-01-XX

### Added
- Initial project structure
- Basic Python package configuration with pyproject.toml
- Core dependencies: PyTorch, PyTorch Geometric, Scanpy
- MIT license
- Code of conduct and contributing guidelines
- Basic test structure

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

---

## Template for Future Releases

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Vulnerability fixes
```

## Release Process

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog Update**: Add new section with changes
3. **Documentation**: Update relevant documentation
4. **Testing**: Ensure all tests pass
5. **Tag Release**: Create git tag with version number
6. **Publish**: Release to PyPI (when ready)

## Categories Explained

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

## Versioning Strategy

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

For pre-release versions:
- **Alpha** (0.x.x): Early development, frequent breaking changes
- **Beta** (1.0.0-beta.x): Feature complete, stabilizing API
- **RC** (1.0.0-rc.x): Release candidate, final testing