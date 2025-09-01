
# Internal template formatting package

Single source of truth for internal template formatting. Ensures compatibility between `scaling-internal` and `eval-framework`

### Install uv

`uv` is used for dependency management and packaging in Python projects. To install uv, follow the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).


## Project Structure
- src/: Contains the template formatting code
- tests/: Contains pytest test cases.
    - test_formatter_eval.py: Basic unit tests for the template formatter derived from `eval-framework`
    - test_formatter_scaling.py: Basic unit tests for the template formatter derived from `scaling-internal`
- pyproject.toml: Configuration file for uv and other tools like MyPy, ruff and pytest.


## Adding dependencies

- **Adding Production Dependencies**: These are dependencies necessary for your project to run. For example, if your project uses Pydantic for data validation, you would add it as a production dependency:

```bash
uv add pydantic
```
- **Adding Development Dependencies**: These are dependencies that are only needed during development, such as testing libraries or linters. For instance, to add pytest for writing and running tests, you would specify it as a development dependency:

```bash
uv add --group dev pytest
```

After adding any new dependencies, you need to install them to update your project's virtual environment:
```bash
uv sync
```
This command ensures that all dependencies listed in your pyproject.toml file are correctly installed and available for use in your project.

To install all dependencies (including optional ones), run
```bash
uv install --extras optional
```

## Usage
**Running Commands with uv**

`uv`` creates a virtual environment for your project, which isolates your dependencies from the global Python environment. This isolation helps prevent version conflicts and ensures reproducibility. Here's how to use `uv`` to run commands:

  - **Installation**: To set up pre-commit hooks, you first need to install the pre-commit package and then install the hooks.

    You can either follow the [install instructions](https://pre-commit.com/#install) or install it globally through `uv tool install pre-commit`

    ```bash
    pre-commit install
    ```

  - **Running Hooks Manually**: Although pre-commit hooks are triggered automatically before each commit, you can also run them manually to check your files at any time:

    ```bash
    pre-commit run -a
    ```
    This command runs all hooks against all files, which is useful for initial setup or periodic checks.

    - **Current Hooks**:
      - **Check JSON**: Ensures JSON files are valid.
      - **Pretty format JSON**: Formats JSON files to be more readable.
      - **Fix End of Files**: Ensures files end with a newline.
      - **Trim Trailing Whitespace**: Removes unnecessary trailing whitespace.
      - **Ruff**: Runs the Ruff linter to check Python code for stylistic and programming errors.
      - **Ruff-format**: Automatically formats Python code using Ruff.


- **Static Type Checking with MyPy**: To ensure your code is type-safe, run MyPy to check for type errors. This should be done frequently during development to catch type-related issues early:
    ```bash
    uv run --all-extras mypy ./src
    uv run --all-extras mypy ./tests
    ```
  Run these commands after making changes to your source or test files to verify that your changes haven't introduced type errors.

- **Running Tests with pytest**: To ensure your code works as expected and hasn't broken existing functionality, run your tests:
    ```bash
    uv run --all-extras pytest
    ```
  Run this command frequently during development, especially before committing changes, to ensure all tests pass.
