"""Specification of the composed BigCodeBench task.

Only the OLMES 3-shot variant is registered. ``test_formatter_hash`` pins it against the real HuggingFace data;
the offline tests pin the assembled prompt + execution context, and the reconstruction of the scored snippet.
"""

from typing import Any

import pytest
from datasets import DownloadConfig, load_dataset

from eval_framework.benchmarks.bigcodebench import (
    _PROMPT_INSTRUCTION,
    BIGCODEBENCH_BENCHMARKS,
    _reconstruct,
    bigcodebench_olmes,
)
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.code_execution_pass_at_one import CodeExecutionPassAtOneContext
from eval_framework.tasks.utils import (
    BIG_CODE_BENCH_PACKAGE_MAPPING,
    extract_imports,
    extract_python_code_from_response,
)
from template_formatting.formatter import (
    BaseFormatter,
    ConcatFormatter,
    Llama3Formatter,
    Message,
    NoStripConcatFormatter,
    Role,
)
from tests.tests_eval_framework.benchmarks.utils import DatasetStub, first_sample
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter, NoStripConcatFormatter])
@pytest.mark.parametrize("benchmark", BIGCODEBENCH_BENCHMARKS, ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=3)  # OLMES is 3-shot


_ROW: dict[str, Any] = {
    "complete_prompt": 'def add(a, b):\n    """Adds two numbers."""\n',
    "code_prompt": "def add(a, b):",
    "test": (
        "import unittest\n\nclass T(unittest.TestCase):\n    def test(self):\n        self.assertEqual(add(1, 2), 3)\n"
    ),
    "canonical_solution": "    return a + b",
}


def test_olmes_prompt_and_context() -> None:
    sample = first_sample(bigcodebench_olmes(dataset=DatasetStub({"v0.1.2": [_ROW]})), num_fewshot=0)
    assert sample.messages == [
        Message(role=Role.USER, content=_PROMPT_INSTRUCTION + "\n```\n" + _ROW["complete_prompt"].strip() + "\n")
    ]
    assert sample.ground_truth == _ROW["canonical_solution"]
    assert sample.possible_completions is None
    assert isinstance(sample.context, CodeExecutionPassAtOneContext)
    assert sample.context.code_prompt == _ROW["code_prompt"]
    assert sample.context.test_code == _ROW["test"]


def test_reconstruct_prepends_code_prompt_and_strips_fences() -> None:
    context = CodeExecutionPassAtOneContext(
        run_env="python:3.12",
        code_prompt="def add(a, b):\n",
        test_code="t",
        snippet_merge_fn="merge",
        output_parse_fn="parse",
        package_downloads={},
    )
    result = _reconstruct("```python\n    return a + b\n```", context=context, ground_truth=None, messages=[])
    # code_prompt + the generated body with the markdown fences stripped out.
    assert result == "def add(a, b):\n\n    return a + b\n"


class TestExtractPythonCodeFromResponse:
    def test_python_code_block(self) -> None:
        response = """Here's a solution:

```python
def hello_world():
    print("Hello, World!")
```
Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_python_code_from_response(response) == expected

    def test_markdown_code_block(self) -> None:
        response = """Here's a solution:

```markdown
def hello_world():
    print("Hello, World!")
```
Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_python_code_from_response(response) == expected

    def test_nested_markdown_python(self) -> None:
        response = """Here's a solution:

```markdown
### Solution

```python
def hello_world():
    print("Hello, World!")

```

Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_python_code_from_response(response) == expected

    def test_generic_code_block(self) -> None:
        response = """Here's a solution:

```markdown

def hello_world():
    print("Hello, World!")

```

Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_python_code_from_response(response) == expected

    def test_real_example_1(self) -> None:
        response = """Below is a Python script with a self-contained function that solves the problem and passes
        corresponding tests:

```python
import collections
from itertools import zip_longest
from random import choices

def task_func(l1, l2, K=10):
    \u0022\u0022\u0022
    Combines two lists by alternating their elements, even if they are of different lengths. Elements from the longer
    list without a counterpart in the shorter one will be included on their own. Then, creates a random sample of
    size K from the combined list, and calculates the frequency of each element in the sample.

    Args:
        l1(list): The first list.
        l2(list): The second list.
        K(int, optional): The size of the random sample.Defaults to 10.

    Returns:
        collections.Counter: An object that counts the frequency of each element in the sample.
\u0022\u0022\u0022

# Combine two lists by alternating their elements
combined_list = [item for pair in zip_longest(l1, l2) for item in pair if item is not None]

# Create a random sample of size K from the combined list
sample = choices(combined_list, k=K)

# Calculate the frequency of each element in the sample
frequency = collections.Counter(sample)

return frequency

# Example usage:
l1 = [1, 2, 3]
l2 = ['a', 'b', 'c', 'd']
K = 10

result = task_func(l1, l2, K)
print(result)
```"""
        expected = """import collections
from itertools import zip_longest
from random import choices

def task_func(l1, l2, K=10):
    \u0022\u0022\u0022
    Combines two lists by alternating their elements, even if they are of different lengths. Elements from the longer
    list without a counterpart in the shorter one will be included on their own. Then, creates a random sample of
    size K from the combined list, and calculates the frequency of each element in the sample.

    Args:
        l1(list): The first list.
        l2(list): The second list.
        K(int, optional): The size of the random sample.Defaults to 10.

    Returns:
        collections.Counter: An object that counts the frequency of each element in the sample.
\u0022\u0022\u0022

# Combine two lists by alternating their elements
combined_list = [item for pair in zip_longest(l1, l2) for item in pair if item is not None]

# Create a random sample of size K from the combined list
sample = choices(combined_list, k=K)

# Calculate the frequency of each element in the sample
frequency = collections.Counter(sample)

return frequency

# Example usage:
l1 = [1, 2, 3]
l2 = ['a', 'b', 'c', 'd']
K = 10

result = task_func(l1, l2, K)
print(result)"""
        assert extract_python_code_from_response(response) == expected

    def test_real_example_2(self) -> None:
        response = """Below is a Python script with a self-contained function that solves the problem and passes
        corresponding tests:

```markdown
### Problem: Generate Random Numbers and Plot with Kurtosis
### Solution: Python Script

```python
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

def task_func(intervals=100, seed=0):
    \u0022\u0022\u0022
    Generates a series of random numbers over a specified number of intervals with a delay of 1 second between each
    interval.It then plots these numbers as a function of elapsed time and returns the Axes object along with
    the     kurtosis     value     of     the     generated     numbers.

    Args:
        intervals(int, optional): Number of intervals.Defaults to 100.
        seed(int, optional): Seed for random number generation.Defaults to 0.

    Returns:
        matplotlib.axes.Axes: The Axes object representing the plot. float: The kurtosis value of the generated numbers.
\u0022\u0022\u0022

# Set seed for reproducibility
random.seed(seed)

# Initialize lists to hold time and random numbers
times = []
numbers = []

# Generate random numbers over specified intervals with a delay
for i in range(intervals):
    # Append current time
    times.append(time.time())

    # Generate a random number and append it
    numbers.append(random.random())

    # Introduce a delay of 1 second
    time.sleep(1)

# Calculate elapsed time
elapsed_time = [t - times[0] for t in times]

# Plot the numbers as a function of elapsed time
fig, ax = plt.subplots()
ax.plot(elapsed_time, numbers)

# Set title and labels
ax.set_title('Random Numbers Over Time')
ax.set_xlabel('Elapsed Time (s)')
ax.set_ylabel('Random Number')

# Return the Axes object and the kurtosis value
return ax, kurtosis(numbers)

# Example usage
ax, kurtosis_value = task_func(intervals=10, seed=42)
print(f'Kurtosis Value: {kurtosis_value}')
plt.show()
```"""
        expected = """import time
import random
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

def task_func(intervals=100, seed=0):
    \u0022\u0022\u0022
    Generates a series of random numbers over a specified number of intervals with a delay of 1 second between each
    interval.It then plots these numbers as a function of elapsed time and returns the Axes object along with
    the     kurtosis     value     of     the     generated     numbers.

    Args:
        intervals(int, optional): Number of intervals.Defaults to 100.
        seed(int, optional): Seed for random number generation.Defaults to 0.

    Returns:
        matplotlib.axes.Axes: The Axes object representing the plot. float: The kurtosis value of the generated numbers.
\u0022\u0022\u0022

# Set seed for reproducibility
random.seed(seed)

# Initialize lists to hold time and random numbers
times = []
numbers = []

# Generate random numbers over specified intervals with a delay
for i in range(intervals):
    # Append current time
    times.append(time.time())

    # Generate a random number and append it
    numbers.append(random.random())

    # Introduce a delay of 1 second
    time.sleep(1)

# Calculate elapsed time
elapsed_time = [t - times[0] for t in times]

# Plot the numbers as a function of elapsed time
fig, ax = plt.subplots()
ax.plot(elapsed_time, numbers)

# Set title and labels
ax.set_title('Random Numbers Over Time')
ax.set_xlabel('Elapsed Time (s)')
ax.set_ylabel('Random Number')

# Return the Axes object and the kurtosis value
return ax, kurtosis(numbers)

# Example usage
ax, kurtosis_value = task_func(intervals=10, seed=42)
print(f'Kurtosis Value: {kurtosis_value}')
plt.show()"""
        assert extract_python_code_from_response(response) == expected

    def test_no_code_block(self) -> None:
        response = "Here's a solution without any code block."
        expected = response
        assert extract_python_code_from_response(response) == expected

    def test_empty_code_block(self) -> None:
        response = "Here's an empty code block:\n```\n```"
        expected = ""
        assert extract_python_code_from_response(response) == expected

    def test_code_block_with_whitespace(self) -> None:
        response = """Here's a solution:

    ```python

def hello_world():
    print("Hello, World!")

    ```"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_python_code_from_response(response) == expected


def test_all_dataset_imports_in_mapping() -> None:
    """Every third-party import used by the real BigCodeBench solutions must be in the package mapping, so the
    sandbox can install it. Skips only if the dataset can't be downloaded (unlike the original, which also
    swallowed assertion failures via a broad try/except)."""
    try:
        dataset = load_dataset(path="bigcode/bigcodebench", download_config=DownloadConfig(max_retries=5))
    except Exception as e:
        pytest.skip(f"Could not download the BigCodeBench dataset: {e}")

    all_imports: set[str] = set()
    for item in dataset["v0.1.4"]:
        code = item.get("solution", "")
        if code:
            _, packages = extract_imports(code)
            all_imports.update(packages)
    missing = sorted(imp for imp in all_imports if imp not in BIG_CODE_BENCH_PACKAGE_MAPPING)
    assert not missing, f"{len(missing)} dataset imports not in the mapping: {missing}"
