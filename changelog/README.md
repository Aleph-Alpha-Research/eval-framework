# Changelog Manager

A command-line tool for managing semantic versioning in CHANGELOG.md files following the Keep a Changelog format.

## Features

- Automatic version bumping based on [MAJOR], [MINOR], or [PATCH] markers
- Release management for unreleased changes
- Version comparison and validation
- Content extraction for specific versions

## Usage

```bash
# Release unreleased changes with automatic version bump
python changelog.py release

# Check and compare versions
python changelog.py check 1.2.3

# Get current version
python changelog.py current

# Get content of specific version
python changelog.py content 1.2.3

```

## Changelog Format

The tool expects a CHANGELOG.md file with the following structure:

```
## [Unreleased]
### Added
- [MAJOR] Breaking change
- [MINOR] New feature
- Simple addition without tag (treated as PATCH)

### Changed
- [PATCH] Updated dependency
- Minor update (treated as PATCH)

### Removed
- [MAJOR] Removed deprecated feature Y

### Fixed
- [PATCH] Bug fix in module Z
- Another bugfix (treated as PATCH)

### Security
- [PATCH] Fixed security vulnerability

## [1.0.0] - 2024-01-01
- Initial release
```

## Change Categories

    Added - New features
    Changed - Changes in existing functionality
    Deprecated - Soon-to-be removed features
    Removed - Now removed features
    Fixed - Bug fixes
    Security - Security vulnerability fixes


## Version Bumping Rules

    [MAJOR] - Increments major version (x.0.0)
    [MINOR] - Increments minor version (0.x.0)
    [PATCH] - Increments patch version (0.0.x)
    No tag - Treated as [PATCH]


If multiple markers exist, the highest precedence wins (MAJOR > MINOR > PATCH).
