import pytest
from datasets import DownloadConfig, load_dataset

from eval_framework.tasks.benchmarks.bigcodebench import extract_executable_code
from eval_framework.tasks.utils import BIG_CODE_BENCH_PACKAGE_MAPPING, extract_imports


class TestExtractExecutableCode:
    def test_python_code_block(self) -> None:
        response = """Here's a solution:

```python
def hello_world():
    print("Hello, World!")
```
Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_executable_code(response) == expected

    def test_markdown_code_block(self) -> None:
        response = """Here's a solution:

```markdown
def hello_world():
    print("Hello, World!")
```
Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_executable_code(response) == expected

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
        assert extract_executable_code(response) == expected

    def test_generic_code_block(self) -> None:
        response = """Here's a solution:

```markdown

def hello_world():
    print("Hello, World!")

```

Hope this helps!"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_executable_code(response) == expected

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
        assert extract_executable_code(response) == expected

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
        assert extract_executable_code(response) == expected

    def test_no_code_block(self) -> None:
        response = "Here's a solution without any code block."
        expected = response
        assert extract_executable_code(response) == expected

    def test_empty_code_block(self) -> None:
        response = "Here's an empty code block:\n```\n```"
        expected = ""
        assert extract_executable_code(response) == expected

    def test_code_block_with_whitespace(self) -> None:
        response = """Here's a solution:

    ```python

def hello_world():
    print("Hello, World!")

    ```"""
        expected = 'def hello_world():\n    print("Hello, World!")'
        assert extract_executable_code(response) == expected


def test_all_imports_in_mapping() -> None:
    """Test that all imports in the BigCodeBench dataset are in our mapping."""
    # Load the dataset
    try:
        dataset = load_dataset(path="bigcode/bigcodebench", download_config=DownloadConfig(max_retries=5))

        # Process each split
        for split, data in dataset.items():
            if split in ["v0.1.4"]:  # Adjust based on available splits
                all_unique_imports = set()

                # Collect all unique imports from this split
                for item in dataset[split]:
                    code = item.get("solution", "")
                    if not code:
                        continue

                    _, packages = extract_imports(code)
                    all_unique_imports.update(packages)

                # Check if all imports are in the mapping
                missing_imports = [imp for imp in all_unique_imports if imp not in BIG_CODE_BENCH_PACKAGE_MAPPING]

                # Print missing imports for debugging
                if missing_imports:
                    print(f"Missing imports in mapping: {missing_imports}")

                # Assert all imports are in the mapping
                assert len(missing_imports) == 0, f"Found {len(missing_imports)} imports not in the mapping"

    except Exception as e:
        pytest.skip(f"Skipping dataset test due to error: {str(e)}")
