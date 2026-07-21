from unittest.mock import MagicMock

import pytest

from eval_framework.tasks.benchmarks.mbpp import MBPP, MBPP_PROMPT_WITHOUT_TESTS
from tests.tests_eval_framework.utils import DatasetPatcher


class TestMBPPPromptWithoutTests:
    @pytest.fixture
    def mbpp_prompt_without_tests_task(self) -> MBPP_PROMPT_WITHOUT_TESTS:
        with DatasetPatcher(MBPP_PROMPT_WITHOUT_TESTS, num_fewshot=1) as patched_task:
            return patched_task

    def test_function_header_valid_function(self) -> None:
        line = "def example_function(param1, param2):"
        expected = "def example_function(param1, param2):"
        result = MBPP_PROMPT_WITHOUT_TESTS._get_function_header(line)
        assert result == expected

    def test_function_header_no_function(self) -> None:
        line = "print('Hello, World!')"
        expected = ""
        result = MBPP_PROMPT_WITHOUT_TESTS._get_function_header(line)
        assert result == expected

    def test_function_header_multiline_function(self) -> None:
        line = "def example_function(\n    param1,\n    param2\n):"
        expected = ""
        result = MBPP_PROMPT_WITHOUT_TESTS._get_function_header(line)
        assert result == expected

    def test_function_header_incomplete_function(self) -> None:
        line = "def example_function(param1, param2"
        expected = ""
        result = MBPP_PROMPT_WITHOUT_TESTS._get_function_header(line)
        assert result == expected

    def test_function_header_function_with_decorator(self) -> None:
        line = "@decorator\ndef example_function(param1, param2):"
        expected = "def example_function(param1, param2):"
        result = MBPP_PROMPT_WITHOUT_TESTS._get_function_header(line)
        assert result == expected


@pytest.fixture
def mbpp_instance() -> MBPP:
    with DatasetPatcher(MBPP) as patched_task:
        return patched_task


@pytest.fixture
def sample_with_asserts() -> MagicMock:
    sample = MagicMock()
    sample.ground_truth = "['assert func(1) == 2', 'assert func(3) == 4']"
    return sample


class TestPostProcessGenerated:
    def test_case1_begin_code(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test when completion has BEGIN marker followed by code"""
        completion = "```python\ndef func(x):\n    return x + 1"
        expected = (
            "def func(x):\n    return x + 1\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )
        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_case2_code_end(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test when completion has code followed by END marker"""
        completion = "def func(x):\n    return x + 1\n```"
        expected = (
            "def func(x):\n    return x + 1\n\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )
        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_case3_begin_code_end(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test when completion has both BEGIN and END markers"""
        completion = "```python\ndef func(x):\n    return x + 1\n```"
        expected = (
            "def func(x):\n    return x + 1\n\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )
        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_case4_code_only(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test when completion has only code without markers"""
        completion = "def func(x):\n    return x + 1"
        expected = (
            "def func(x):\n    return x + 1\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )
        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_empty_ground_truth(self, mbpp_instance: MBPP) -> None:
        """Test with empty ground truth"""
        sample = MagicMock()
        sample.ground_truth = ""
        completion = "def func(x):\n    return x + 1"
        expected = "def func(x):\n    return x + 1\n"
        result = mbpp_instance.post_process_generated_completion(completion, sample)
        assert result == expected

    def test_multiple_code_blocks(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test with multiple code blocks in completion"""
        completion = (
            "Here's a solution:\n```python\ndef func(x):\n    return x + 1\n```\nAlternatively:\n```python\n"
            "def func(x):\n    return 1 + x\n```"
        )
        expected = (
            "def func(x):\n    return x + 1\n\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )
        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_indentation_preservation_in_post_process(
        self, mbpp_instance: MBPP, sample_with_asserts: MagicMock
    ) -> None:
        """Test that post_process_generated_completion preserves indentation correctly"""
        # Properly indented code
        completion = """```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
```"""

        expected = """def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_complex_indentation_preservation(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test preservation of complex indentation patterns"""
        # Code with nested functions and complex indentation
        completion = """```python
def outer_function(x):
    def inner_function(y):
        if y > 0:
            return y * y
        else:
            return -y * y

    result = 0
    for i in range(x):
        if i % 2 == 0:
            result += inner_function(i)
        else:
            result -= inner_function(i)
    return result
```"""

        expected = """def outer_function(x):
    def inner_function(y):
        if y > 0:
            return y * y
        else:
            return -y * y

    result = 0
    for i in range(x):
        if i % 2 == 0:
            result += inner_function(i)
        else:
            result -= inner_function(i)
    return result

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_mixed_tab_space_indentation(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test preservation of mixed tab and space indentation"""
        # Code with mixed tabs and spaces (represented as \t for clarity)
        # Note: In the actual test, you'd use real tabs
        completion = """```python
def mixed_indentation(n):
    if n <= 0:
\t\treturn 0
    elif n == 1:
\t\treturn 1
    else:
\t\treturn mixed_indentation(n-1) + mixed_indentation(n-2)
```"""

        # Replace \t with actual tabs for the test
        completion = completion.replace("\\t", "\t")

        expected = """def mixed_indentation(n):
    if n <= 0:
\t\treturn 0
    elif n == 1:
\t\treturn 1
    else:
\t\treturn mixed_indentation(n-1) + mixed_indentation(n-2)

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        # Replace \t with actual tabs for the expected result
        expected = expected.replace("\\t", "\t")

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    # (see who implemented and where it was taken from; replace with other prompt we create ourselves)

    def test_indentation_with_comments_and_docstrings(
        self, mbpp_instance: MBPP, sample_with_asserts: MagicMock
    ) -> None:
        """Test preservation of indentation with comments and docstrings"""
        completion = """```python
def calculate_statistics(numbers):
    \u0022\u0022\u0022

    Calculate
    basic
    statistics
    for a list of numbers.

    Args:
    numbers: List
    of
    numeric
    values


Returns:
Dictionary
with mean, median, and mode
\u0022\u0022\u0022
# Check if input is valid
if not numbers:
    return {"mean": None, "median": None, "mode": None}

# Calculate mean
mean = sum(numbers) / len(numbers)

# Calculate median
sorted_numbers = sorted(numbers)
n = len(sorted_numbers)
if n % 2 == 0:
    median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
else:
    median = sorted_numbers[n//2]

# Calculate mode (most frequent value)
frequency = {}
for num in numbers:
    if num in frequency:
        frequency[num] += 1
    else:
        frequency[num] = 1

mode = max(frequency.items(), key=lambda x: x[1])[0]

return {"mean": mean, "median": median, "mode": mode}
```"""

        expected = """def calculate_statistics(numbers):
    \u0022\u0022\u0022

    Calculate
    basic
    statistics
    for a list of numbers.

    Args:
    numbers: List
    of
    numeric
    values


Returns:
Dictionary
with mean, median, and mode
\u0022\u0022\u0022
# Check if input is valid
if not numbers:
    return {"mean": None, "median": None, "mode": None}

# Calculate mean
mean = sum(numbers) / len(numbers)

# Calculate median
sorted_numbers = sorted(numbers)
n = len(sorted_numbers)
if n % 2 == 0:
    median = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
else:
    median = sorted_numbers[n//2]

# Calculate mode (most frequent value)
frequency = {}
for num in numbers:
    if num in frequency:
        frequency[num] += 1
    else:
        frequency[num] = 1

mode = max(frequency.items(), key=lambda x: x[1])[0]

return {"mean": mean, "median": median, "mode": mode}

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_indentation_with_blank_lines(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test preservation of indentation with blank lines between code blocks"""
        completion = """```python
def process_data(data):
    result = []

    for item in data:
        if isinstance(item, str):
            # Process strings
            result.append(item.upper())

        elif isinstance(item, int):
            # Process integers
            result.append(item * 2)

        elif isinstance(item, list):
            # Process nested lists recursively
            result.append(process_data(item))

        else:
            # Skip other types
            continue

    return result
```"""

        expected = """def process_data(data):
    result = []

    for item in data:
        if isinstance(item, str):
            # Process strings
            result.append(item.upper())

        elif isinstance(item, int):
            # Process integers
            result.append(item * 2)

        elif isinstance(item, list):
            # Process nested lists recursively
            result.append(process_data(item))

        else:
            # Skip other types
            continue

    return result

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected


class TestCodeExpander:
    def test_code_expander_with_valid_asserts(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with valid assert statements"""
        code = "def func(x):\n    return x + 1\n"
        gt_asserts = "['assert func(1) == 2', 'assert func(3) == 4']"
        expected = (
            "def func(x):\n    return x + 1\ntry:\n    assert func(1) == 2\n    assert func(3) == 4\n"
            "    score = True\nexcept:\n    score = False\nprint(score)"
        )

        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == expected

    def test_code_expander_with_empty_asserts(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with empty assert list"""
        code = "def func(x):\n    return x + 1\n"
        gt_asserts = "[]"
        expected = "def func(x):\n    return x + 1\ntry:\n    score = True\nexcept:\n    score = False\nprint(score)"

        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == expected

    def test_code_expander_with_no_asserts(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with no assert statements (empty string)"""
        code = "def func(x):\n    return x + 1\n"
        gt_asserts = ""
        expected = "def func(x):\n    return x + 1\n"

        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == expected

    def test_code_expander_with_invalid_asserts_format(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with invalid assert format (not a list)"""
        code = "def func(x):\n    return x + 1\n"
        gt_asserts = "'assert func(1) == 2'"  # Not a list, just a string

        # This should print a warning and return the original code
        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == code

    def test_code_expander_with_multiple_asserts(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with multiple assert statements"""
        code = "def func(x):\n    return x * 2\n"
        gt_asserts = "['assert func(1) == 2', 'assert func(2) == 4', 'assert func(0) == 0']"
        expected = (
            "def func(x):\n    return x * 2\ntry:\n    assert func(1) == 2\n    assert func(2) == 4\n"
            "    assert func(0) == 0\n    score = True\nexcept:\n    score = False\nprint(score)"
        )

        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == expected

    def test_code_expander_with_multiline_code(self, mbpp_instance: MBPP) -> None:
        """Test _code_expander with multiline code"""
        code = "def func(x):\n    if x < 0:\n        return 0\n    return x * 2\n"
        gt_asserts = "['assert func(-1) == 0', 'assert func(2) == 4']"
        expected = (
            "def func(x):\n    if x < 0:\n        return 0\n    return x * 2\ntry:\n    assert func(-1) == 0\n"
            "    assert func(2) == 4\n    score = True\nexcept:\n    score = False\nprint(score)"
        )

        result = mbpp_instance._code_expander(code, gt_asserts)
        assert result == expected

    def test_indentation_preservation(self, mbpp_instance: MBPP, sample_with_asserts: MagicMock) -> None:
        """Test that proper indentation is preserved in the processed code"""
        # This is a correctly indented function in the completion
        completion = """```python
def is_valid_parenthese(str1):
    stack = []
    dict1 = {")": "(", "}": "{", "]": "["}
    for char in str1:
        if char in dict1.values():
            stack.append(char)
        elif char in dict1.keys():
            if stack == [] or dict1[char] != stack.pop():
                return False
    if stack == []:
        return True
    else:
        return False
```"""

        # Expected result should maintain the correct indentation
        expected = """def is_valid_parenthese(str1):
    stack = []
    dict1 = {")": "(", "}": "{", "]": "["}
    for char in str1:
        if char in dict1.values():
            stack.append(char)
        elif char in dict1.keys():
            if stack == [] or dict1[char] != stack.pop():
                return False
    if stack == []:
        return True
    else:
        return False

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected

    def test_indentation_with_empty_line_after_function_def(
        self, mbpp_instance: MBPP, sample_with_asserts: MagicMock
    ) -> None:
        """Test handling of code with an empty line after function definition"""
        # This simulates the specific case you mentioned with an empty line after function definition
        completion = """```python
def is_valid_parenthese(str1):

    stack = []
    dict1 = {")": "(", "}": "{", "]": "["}
    for char in str1:
        if char in dict1.values():
            stack.append(char)
        elif char in dict1.keys():
            if stack == [] or dict1[char] != stack.pop():
                return False
    if stack == []:
        return True
    else:
        return False
```"""

        # Expected result should maintain the empty line and proper indentation
        expected = """def is_valid_parenthese(str1):

    stack = []
    dict1 = {")": "(", "}": "{", "]": "["}
    for char in str1:
        if char in dict1.values():
            stack.append(char)
        elif char in dict1.keys():
            if stack == [] or dict1[char] != stack.pop():
                return False
    if stack == []:
        return True
    else:
        return False

try:
    assert func(1) == 2
    assert func(3) == 4
    score = True
except:
    score = False
print(score)"""

        result = mbpp_instance.post_process_generated_completion(completion, sample_with_asserts)
        assert result == expected
