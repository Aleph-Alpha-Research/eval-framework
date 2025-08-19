import os
import tempfile
import time

from eval_framework.tasks.utils import (
    BIG_CODE_BENCH_PACKAGE_MAPPING,
    CallableSerializer,
    _parse_unittest_output,
    execute_python_code_with_tests,
    extract_imports,
    get_external_dependencies,
    run_python_code,
)


def test_run_python_code() -> None:
    # GIVEN a code which prints out a file
    with tempfile.TemporaryDirectory() as tmpdir:
        content = "hello!"
        filename = os.path.join(tmpdir, "o.txt")
        code = f"""with open("{filename}", "r") as f:\n""" """  print(f.read())\n"""
        with open(filename, "w") as f:
            f.write(content)
        # WHEN running it a docker with an injected file
        output = run_python_code(code, image="python:3.13-slim", input_files=[(str(filename), filename)])
        # THEN the code is executed and reads the correct content
        assert content == output


def test_run_python_code_timeout() -> None:
    code = """import time\ntime.sleep(15)"""
    start = time.time()
    run_python_code(code, image="python:3.13-slim", timeout=2)
    assert time.time() - start < 3


def test_run_python_code_with_packages() -> None:
    output = run_python_code("import pytest", image="python:3.13-slim")
    assert "ModuleNotFoundError" in output
    output = run_python_code("import pytest", image="python:3.13-slim", packages=["pytest"])
    assert "ModuleNotFoundError" not in output


class TestParseUnittestOutput:
    """Tests for the _parse_unittest_output function."""

    def test_successful_single_test(self) -> None:
        """Test parsing output from a successful single test."""
        output = "Ran 1 test in 0.001s\n\nOK"
        result = _parse_unittest_output(output)

        assert result.success is True
        assert result.output == "All 1 tests completed successfully."

    def test_successful_multiple_tests(self) -> None:
        """Test parsing output from multiple successful tests."""
        output = "Ran 5 tests in 0.015s\n\nOK"
        result = _parse_unittest_output(output)

        assert result.success is True
        assert result.output == "All 5 tests completed successfully."

    def test_successful_with_additional_output(self) -> None:
        """Test parsing output with additional text before/after the unittest result."""
        output = "Some debug output\nRan 3 tests in 0.005s\n\nOK\nMore output after"
        result = _parse_unittest_output(output)

        assert result.success is True
        assert result.output == "All 3 tests completed successfully."

    def test_failed_tests(self) -> None:
        """Test parsing output from failed tests."""
        output = "Ran 4 tests in 0.008s\n\nFAILED (failures=2)"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Tests failed: failures=2")
        assert output in result.output

    def test_failed_with_errors(self) -> None:
        """Test parsing output with both failures and errors."""
        output = "Ran 6 tests in 0.012s\n\nFAILED (failures=1, errors=2)"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Tests failed: failures=1, errors=2")
        assert output in result.output

    def test_assertion_error(self) -> None:
        """Test parsing output with an AssertionError."""
        output = "AssertionError: Expected 5 but got 3"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Test failed with assertion error:")
        assert output in result.output

    def test_runtime_error(self) -> None:
        """Test parsing output with a runtime error."""
        output = "Error: Division by zero"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Error during execution:")
        assert output in result.output

    def test_exception(self) -> None:
        """Test parsing output with an exception."""
        output = "Exception: Invalid input"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Error during execution:")
        assert output in result.output

    def test_indeterminate_output(self) -> None:
        """Test parsing output that doesn't match any known pattern."""
        output = "Some unexpected output format"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Could not determine test results")
        assert output in result.output

    def test_failed_without_details(self) -> None:
        """Test parsing output with FAILED but no details in parentheses."""
        output = "Ran 2 tests in 0.003s\n\nFAILED"
        result = _parse_unittest_output(output)

        assert result.success is False
        assert result.output.startswith("Tests failed:")
        assert output in result.output

    def test_ok_with_skipped(self) -> None:
        """Test parsing output with OK but some skipped tests."""
        output = "Ran 7 tests in 0.020s\n\nOK (skipped=2)"
        result = _parse_unittest_output(output)

        assert result.success is True
        assert result.output == "All 7 tests completed successfully."

    def test_no_test_count(self) -> None:
        """Test parsing output with OK but no test count."""
        output = "OK"
        result = _parse_unittest_output(output)

        assert result.success is True
        assert result.output == "All tests completed successfully."


class TestExecutePythonCodeWithTests:
    """Integration tests for execute_python_code_with_tests."""

    def test_successful_execution(self) -> None:
        # Simple code that should pass all tests using unittest
        code = "def add(a, b): return a + b"
        test_code = """
import unittest

class TestAdd(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add(1, 2), 3)
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is True
        assert "tests completed successfully" in result.output

    def test_failing_assertion(self) -> None:
        # Code with a failing test
        code = "def add(a, b): return a - b"  # Incorrect implementation
        test_code = "assert add(1, 2) == 3"

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert "AssertionError" in result.output

    def test_syntax_error(self) -> None:
        # Code with syntax error
        code = "def add(a, b) return a + b"  # Missing colon
        test_code = "assert add(1, 2) == 3"

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert "SyntaxError" in result.output

    def test_runtime_error(self) -> None:
        # Code that raises a runtime error
        code = "def divide(a, b): return a / b"
        test_code = "assert divide(1, 0) == float('inf')"

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert any(err in result.output for err in ["ZeroDivisionError", "division by zero"])

    def test_timeout(self) -> None:
        # Code that should timeout
        code = "import time\ndef hang(): time.sleep(5)\nhang()"
        test_code = """
import unittest
class TestHang(unittest.TestCase):
    def test_hang(self):
        self.assertTrue(True)  # This won't run because hang() will timeout
unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING, timeout=1)

        assert result.success is False
        assert "timeout" in result.output.lower()

    def test_with_imports(self) -> None:
        # Code that uses imports
        code = "import math\ndef circle_area(r): return math.pi * r * r"
        test_code = """
import unittest
class TestCircleArea(unittest.TestCase):
    def test_area(self):
        self.assertEqual(round(circle_area(2), 2), 12.57)
unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is True
        assert "tests completed successfully" in result.output

    def test_multiple_assertions(self) -> None:
        # Code with multiple test assertions
        code = """
def is_even(n):
    return n % 2 == 0
        """
        test_code = """
import unittest
class TestIsEven(unittest.TestCase):
    def test_even_numbers(self):
        self.assertTrue(is_even(2))
        self.assertFalse(is_even(3))
        self.assertTrue(is_even(0))
        self.assertTrue(is_even(-2))
unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is True
        assert "tests completed successfully" in result.output

    def test_one_failing_among_many(self) -> None:
        # Code with one failing test among many passing ones
        code = """
def is_positive(n):
    return n > 0  # Bug: doesn't handle zero correctly
        """
        test_code = """
assert is_positive(5) == True
assert is_positive(-5) == False
assert is_positive(0) == True  # This will fail
        """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert "AssertionError" in result.output

    def test_complex_code_execution(self) -> None:
        # More complex code example
        code = """
class Stack:
    def __init__(self) -> None:
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None
    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)
        """

        test_code = """
import unittest
class TestStack(unittest.TestCase):
    def test_stack_operations(self):
        s = Stack()
        self.assertTrue(s.is_empty())
        s.push(1)
        s.push(2)
        self.assertEqual(s.peek(), 2)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)
        self.assertIsNone(s.pop())
unittest.main()
        """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is True
        assert "tests completed successfully" in result.output

    def test_missing_import(self) -> None:
        # Test code that tries to use a module that isn't imported
        code = "def get_pi(): return math.pi"  # Missing import
        test_code = "assert get_pi() > 3.1"

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert any(err in result.output for err in ["NameError", "math is not defined"])

    def test_indentation_error(self) -> None:
        # Test code with indentation error
        code = """
def function():
    x = 1
  y = 2  # Wrong indentation
        """
        test_code = "assert True"

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert "IndentationError" in result.output

    def test_empty_code(self) -> None:
        # Test with empty implementation
        code = ""
        test_code = """
import unittest
class TestEmptyCode(unittest.TestCase):
    def test_empty(self):
        self.assertTrue(True)
unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is True
        assert "tests completed successfully" in result.output

    def test_empty_test_code(self) -> None:
        # Test with empty test code
        code = "def function(): return True"
        test_code = ""

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)

        assert result.success is False
        assert "'unittest' is not defined" in result.output

    # Scenario 1: Correct implementation (should pass)
    # Test for the correct implementation
    def test_successful_unittest_execution(self) -> None:
        # Using the correct implementation
        code = r"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def task_func(texts):
    # Handle empty input
    if all(text.strip() == "" for text in texts):
        return [], []

    # Remove URLs
    cleaned_texts = [re.sub('http[s]?://\S+', '', text) for text in texts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Convert the sparse matrix to a dense format, round the values, convert to tuples and return along with
    # feature names
    dense_matrix = [tuple(round(val, 8) for val in row) for row in tfidf_matrix.toarray().tolist()]
    return dense_matrix, list(vectorizer.get_feature_names_out())
    """

        test_code = r"""
import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self):
        input_texts = ['Visit https://www.python.org for more info.', 'Python is great.', 'I love Python.']
        output = task_func(input_texts)
        sorted_indices = sorted(range(len(output[1])), key=lambda k: output[1][k])
        expected_output = (
            [tuple(row[i] for i in sorted_indices) for row in output[0]],
            sorted(output[1])
        )
        self.assertEqual(output, expected_output)

    def test_case_5(self):
        input_texts = ['', '', '']
        expected_output = ([], [])
        self.assertEqual(task_func(input_texts), expected_output)

unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert result.success is True
        assert result.output == "All 2 tests completed successfully."

    # Test for the flawed implementation
    def test_failing_unittests_for_wrong_implementation(self) -> None:
        # Flawed implementation with multiple issues
        code = r"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def task_func(texts):
    # Missing empty input check

    # Incorrectly removes URLs (missing 's' in https)
    cleaned_texts = [re.sub('http://\\S+', '', text) for text in texts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Doesn't round the values, which will cause precision issues
    dense_matrix = [tuple(val for val in row) for row in tfidf_matrix.toarray().tolist()]
    return dense_matrix, list(vectorizer.get_feature_names_out())
    """

        test_code = r"""
import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self):
        input_texts = ['Visit https://www.python.org for more info.', 'Python is great.', 'I love Python.']
        output = task_func(input_texts)
        sorted_indices = sorted(range(len(output[1])), key=lambda k: output[1][k])
        expected_output = (
            [tuple(row[i] for i in sorted_indices) for row in output[0]],
            sorted(output[1])
        )
        self.assertEqual(output, expected_output)

    def test_case_2(self):
        input_texts = ['', '', '']
        expected_output = ([], [])
        self.assertEqual(task_func(input_texts), expected_output)

unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert result.success is False
        assert "FAILED" in result.output

    # Test for missing implementation
    def test_failing_unittests_for_missing_implementation(self) -> None:
        # No implementation at all
        code = """
# No implementation of task_func
    """

        test_code = r"""
import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self):
        input_texts = ['Visit https://www.python.org for more info.', 'Python is great.', 'I love Python.']
        output = task_func(input_texts)
        sorted_indices = sorted(range(len(output[1])), key=lambda k: output[1][k])
        expected_output = (
            [tuple(row[i] for i in sorted_indices) for row in output[0]],
            sorted(output[1])
        )
        self.assertEqual(output, expected_output)

unittest.main()
    """

        result = execute_python_code_with_tests(code, test_code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert result.success is False
        assert "NameError" in result.output
        assert "task_func" in result.output


class TestImportExtraction:
    def test_basic_imports(self) -> None:
        code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
"""
        imports, packages = extract_imports(code)
        assert len(imports) == 5
        assert packages == {"numpy", "pandas", "sklearn", "os", "sys"}

    def test_multiple_imports_on_one_line(self) -> None:
        code = """
    import os, sys, json
    import numpy as np, pandas as pd
    """
        imports, packages = extract_imports(code)
        assert len(imports) == 2
        assert packages == {"os", "sys", "json", "numpy", "pandas"}

    def test_submodule_imports(self) -> None:
        code = """
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
"""
        imports, packages = extract_imports(code)
        assert len(imports) == 3
        assert packages == {"sklearn", "tensorflow", "matplotlib"}

    def test_complex_imports(self) -> None:
        code = """
import numpy as np
from PIL import Image
import os, sys
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
"""
        imports, packages = extract_imports(code)
        assert len(imports) == 5
        assert packages == {"numpy", "PIL", "os", "sys", "sklearn", "matplotlib"}


class TestExternalDependencies:
    def test_mixed_dependencies(self) -> None:
        code = """
import numpy as np
import os
from PIL import Image
import sys
from sklearn.model_selection import train_test_split
"""
        external_deps = get_external_dependencies(code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert "pillow" in external_deps
        assert "numpy" in external_deps
        assert "scikit-learn" in external_deps
        assert "os" not in external_deps
        assert "sys" not in external_deps

    def test_only_stdlib(self) -> None:
        code = """
import os
import sys
import json
from datetime import datetime
"""
        external_deps = get_external_dependencies(code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert len(external_deps) == 0

    def test_only_external(self) -> None:
        code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
"""
        external_deps = get_external_dependencies(code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert len(external_deps) == 4
        assert "numpy" in external_deps
        assert "pandas" in external_deps
        assert "scikit-learn" in external_deps
        assert "pillow" in external_deps


class TestExtractExternalDependencies:
    def test_external_dependencies(self) -> None:
        code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
"""
        packages = get_external_dependencies(code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert sorted(packages) == ["numpy", "pandas", "pillow", "scikit-learn"]

    def test_no_external_dependencies(self) -> None:
        code = """
import os
import sys
import json
"""
        pip_commands = get_external_dependencies(code, BIG_CODE_BENCH_PACKAGE_MAPPING)
        assert pip_commands == []


class TestBigCodeBenchDataset:
    def test_mapping_coverage(self) -> None:
        """Test the coverage of our mapping against a sample of common packages."""
        common_packages = [
            # Standard library
            "os",
            "sys",
            "json",
            "datetime",
            "re",
            "math",
            # External packages
            "numpy",
            "pandas",
            "matplotlib",
            "sklearn",
            "tensorflow",
            "PIL",
            "requests",
            "bs4",
            "flask",
            "django",
        ]

        for pkg in common_packages:
            assert pkg in BIG_CODE_BENCH_PACKAGE_MAPPING, f"Package {pkg} not in mapping"

def test_fn_recover() -> None:
    def fn(x: int) -> int:
        return x * 2

    serializer = CallableSerializer()
    encoded_fn = serializer.encode(fn)
    decoded_fn = serializer.decode(encoded_fn)
    assert decoded_fn(2) == fn(2)