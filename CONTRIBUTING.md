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




## License

By contributing, you agree that your contributions will be licensed under the same license as the project. See the `LICENSE` file in the root directory for details.
