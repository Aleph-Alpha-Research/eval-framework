"""Checking Python code against a test suite in the sandbox (BigCodeBench-style).

``execute_python_code_with_tests`` merges the code with its tests (e.g. ``unittest_merge_snippets``), pip-installs
the external packages its imports need (``get_external_dependencies``), runs it via ``run_python_code``, and turns
the output into an ``ExecutionResult`` (e.g. ``parse_unittest_output``).
"""

import re
from collections.abc import Callable
from typing import NamedTuple

from eval_framework.metrics.completion.code_execution.sandbox import run_python_code


def unittest_merge_snippets(code: str, test_code: str) -> str:
    # Add unittest.main() if not present (note that without "if" sometimes it just reports
    # "Ran 0 tests" errorneously).
    if "unittest.main(" not in test_code:
        test_code += "\n\nif __name__ == '__main__':\n  unittest.main()"

    # Combine the implementation code and test code
    combined_code = code + "\n\n" + test_code
    return combined_code


class ExecutionResult(NamedTuple):
    """
    A named tuple to store the result of code execution.

    Attributes:
        success (bool): Indicates if the execution was successful.
        output (str): Contains the output or error messages from the execution.
    """

    success: bool
    output: str


def execute_python_code_with_tests(
    code: str,
    test_code: str,
    dockerfile: str | None,
    package_mapping: dict[str, str | None],
    merge_code_fn: Callable[[str, str], str],
    image: str | None,
    timeout: int,
    parse_output_fn: Callable[[str], ExecutionResult],
) -> ExecutionResult:
    """
    Executes the given code with test cases in a sandboxed environment.

    :param code: The code to be tested.
    :param test_code: The test cases to run against the code.
    :param package_mapping: Mapping of package names to install commands.
    :param merge_code_fn: function to merge LLM and test code
    :param image: Docker image to use.
    :param timeout: Timeout for the execution in seconds.
    :param parse_otuput_fn: function to parse docker execution output
    :return: An ExecutionResult named tuple with success status and output or errors.
    """
    combined_code = merge_code_fn(code, test_code)

    packages = get_external_dependencies(combined_code, package_mapping)

    # Run the combined code in the sandbox
    output = run_python_code(
        combined_code,
        image=image,
        dockerfile=dockerfile,
        timeout=timeout,
        packages=packages,
    )

    # Parse the output to determine success
    return parse_output_fn(output)


def parse_unittest_output(output: str) -> ExecutionResult:
    """Parse the unittest output to determine success and format the result."""
    # Check for unittest success pattern
    if "OK" in output and "FAILED" not in output:
        # Extract the test summary if possible
        match = re.search(r"Ran (\d+) tests? in [\d.]+s", output)
        if match:
            test_count = match.group(1)
            test_output = f"All {test_count} tests completed successfully."
        else:
            test_output = "All tests completed successfully."

        return ExecutionResult(True, test_output)

    # Check for unittest failure pattern
    elif "FAILED" in output:
        # Try to extract failure details
        match = re.search(r"FAILED \((.+)\)", output)
        if match:
            failure_details = match.group(1)
            return ExecutionResult(False, f"Tests failed: {failure_details}\n{output}")
        else:
            return ExecutionResult(False, f"Tests failed: {output}")

    # Check for common error patterns
    elif "AssertionError" in output:
        return ExecutionResult(False, f"Test failed with assertion error: {output}")
    elif "Error:" in output or "Exception:" in output:
        return ExecutionResult(False, f"Error during execution: {output}")

    # If we can't determine success/failure, return the raw output
    return ExecutionResult(
        False,
        f"Could not determine test results, potentially due to timeout. Output: {output}",
    )


def get_external_dependencies(code: str, package_mapping: dict[str, str | None]) -> list[str]:
    """Identify external dependencies in the code."""
    _, packages = extract_imports(code)

    external_packages = []
    for pkg in packages:
        if pkg in package_mapping and package_mapping[pkg] is not None:
            external_packages.append(package_mapping[pkg])
    return external_packages  # type: ignore[return-value]


def extract_imports(code: str) -> tuple[list[str], set[str]]:
    """Extract all import statements and the imported packages from code."""
    # Pattern for 'import x' or 'import x, y, z'
    import_pattern = r"^import\s+([\w\s,.]+)"

    # Pattern for 'from x import y'
    from_pattern = r"^from\s+([\w.]+)\s+import\s+"

    imports = []
    packages = set()

    for line in code.split("\n"):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle 'import x' or 'import x, y, z'
        import_match = re.match(import_pattern, line)
        if import_match:
            imports.append(line)
            # Extract all packages from the import statement
            imported_items = import_match.group(1).split(",")
            for item in imported_items:
                # Clean up and get the base package name
                pkg = item.strip().split(".")[0].split(" as ")[0]
                if pkg:
                    packages.add(pkg)
            continue

        # Handle 'from x import y'
        from_match = re.match(from_pattern, line)
        if from_match:
            imports.append(line)
            # Get the base package name
            pkg = from_match.group(1).split(".")[0]
            if pkg:
                packages.add(pkg)

    return imports, packages
