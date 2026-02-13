# Contributing to Eval Framework

Thank you for your interest in contributing to the Eval Framework! We welcome contributions from the community and are grateful for your support.

## How to Contribute

We welcome several types of contributions:

- **Bug fixes**: Help us identify and fix issues
- **Feature implementations**: Add new functionality to the framework
- **Documentation improvements**: Enhance or clarify existing documentation
- **Performance optimizations**: Make the framework faster and more efficient
- **Examples and tutorials**: Help others learn how to use the framework

## Getting Started

1. **Fork the repository**: Click the "Fork" button on the GitHub repository page to create your own copy of the project.

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/eval-framework.git
   cd eval-framework
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/Aleph-Alpha-Research/eval-framework.git
   ```

4. **Install uv** if you haven't already following [these instructions](https://docs.astral.sh/uv/getting-started/installation/)

5. **Install the project dependencies**:
   ```bash
   uv sync --all-extras
   ```

6. **Install pre-commit**:
   ```bash
   uv tool install pre-commit
   uv run pre-commit install
   ```
7. **Documentation** After installation, task documentation can be generated with `uv run python -m eval_framework.utils.generate_task_docs` (see [installation docs](docs/installation.md)) for more details.

## Submitting Changes

### Pull Request Process

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes and add tests for them**:
   ```bash
   # Make your code changes
   # Add tests for your new functionality in the tests/ directory
   git add .
   git commit -m "Add meaningful commit message"
   ```


3. **Make sure your code lints**:
   ```bash
   pre-commit run --all-files
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**: Go to GitHub, navigate to your fork, and create a Pull Request from your branch to the original repository's main branch. Fill in the PR description and submit.


## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Clear title**: Summarize the issue briefly
- **Environment details**: OS, Python version, package versions
- **Reproduction steps**: Step-by-step instructions to reproduce
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Include full error messages and stack traces
- **Additional context**: Screenshots, logs, or other relevant information



## Feature Requests

When requesting features:

- **Use case**: Describe why this feature would be useful
- **Proposed solution**: How you think it should work
- **Additional context**: Any other relevant information


## Releasing a New Version

The `eval_framework` package follows [semantic versioning specification](https://semver.org/). That is, starting with the first `1.0.0` release we
aim for backwards-compatible changes within minor version changes and compatibility-breaking changes only within major version.

To launch a new release, please follow these steps:

- Increase the `project.version` number in the `pyproject.toml` either manually, or through `uv version --bump={major,minor,patch}`
- Adapt the `CHANGELOG.md` file to include the new version information
- Merge these changes to the `main` branch
- Create a new release [through Github](https://github.com/Aleph-Alpha-Research/eval-framework/releases)
   - Click on "Create a new release"
   - Create the appropriate tag on the `main` branch. That is, if the package version is `X.Y.Z` the tag must be `vX.Y.Z` or the release workflow will fail.
   - Set the title of the release equivalent to the version number (`X.Y.Z`)
   - Click on `Generate release notes`
   - Manually copy the highlights from the changelog on top of the auto-generated notes
   - Click on Publish Release
- Update the project version to an incremented `dev` release by running `uv version --bump patch --bump dev`. Merge this change to `main`.

This will create a new version tag and run the release workflow. Open the [Github Actions](https://github.com/Aleph-Alpha-Research/eval-framework/actions)
panel and look for the release workflow. Once things are ready, you will have to approve publishing to PyPi.

### If things go wrong...

When a release workflow fails, the best way is to go to the [Release page](https://github.com/Aleph-Alpha-Research/eval-framework/releases) and delete the release
and also delete the corresponding tag on the [tag page](https://github.com/Aleph-Alpha-Research/eval-framework/tags). Then fix the workflow and re-release the package.


## License

By contributing, you agree that your contributions will be licensed under the same license as the project. See the `LICENSE` file in the root directory for details.
