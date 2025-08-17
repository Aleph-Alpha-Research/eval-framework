# Creating Completion Tasks

This guide shows you how to create **completion tasks**, benchmarks where the model generates text to complete a prompt (e.g., math problems, code generation, question answering).

## Quick Start Template

```python
from typing import Any
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion


class YourCompletionTask(BaseTask[str]):
    # Required attributes
    NAME = "YourTaskName"
    DATASET_PATH = "your-dataset/path"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["default"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Generate the question/prompt for the model."""
        return f"Question: {item['question']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Extract the correct answer from the dataset."""
        return item['answer']
```

## Step-by-Step Implementation

### 1. Basic Setup

Start with the minimal structure:

```python
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion

class MathQATask(BaseTask[str]):
    NAME = "MathQA"
    DATASET_PATH = "math_qa_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["arithmetic"]
```

### 2. Implement Required Methods

#### _get_instruction_text()
This method formats the question from your dataset:

```python
def _get_instruction_text(self, item: dict[str, Any]) -> str:
    """Convert dataset item to a question prompt."""
    # Example for math problems
    return f"Solve this math problem: {item['problem']}"

    # Example for code generation
    # return f"Complete this function:\n{item['function_signature']}"

    # Example for Q&A
    # return f"Q: {item['question']}\nA:"
```

#### _get_ground_truth()
This method extracts the correct answer:

```python
def _get_ground_truth(self, item: dict[str, Any]) -> str:
    """Extract the correct answer from the dataset item."""
    # Simple case - direct answer
    return item['answer']

    # For numeric answers, you might want to normalize
    # return str(float(item['answer']))

    # For code, might return the complete function
    # return item['complete_code']
```

### 3. Common Completion Task Patterns

#### Pattern 1: Question Answering
```python
class QATask(BaseTask[str]):
    NAME = "QA Task"
    DATASET_PATH = "qa_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["general"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\nAnswer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return item['answer']

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"  # Helps model start response correctly
```

#### Pattern 2: Math Problem Solving
```python
class MathTask(BaseTask[str]):
    NAME = "Math Problems"
    DATASET_PATH = "math_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["algebra", "geometry"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Problem: {item['problem']}\nSolution:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return item['solution']

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Extract final numerical answer from solution."""
        import re
        # Look for "The answer is X" pattern
        match = re.search(r'The answer is (\d+(?:\.\d+)?)', completion_text)
        if match:
            return match.group(1)
        return completion_text.strip()
```

#### Pattern 3: Code Generation
```python
from eval_framework.metrics.completion_metrics.code_execution_pass_at_one import CodeExecutionPassAtOne

class CodeTask(BaseTask[str]):
    NAME = "Code Generation"
    DATASET_PATH = "code_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [CodeExecutionPassAtOne]
    SUBJECTS = ["python"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Complete this function:\n{item['prompt']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return item['canonical_solution']

    def _get_eval_kwargs(self, item: dict[str, Any]) -> dict[str, Any]:
        """Provide test cases for code execution."""
        return {
            'test_cases': item['test_cases'],
            'entry_point': item['entry_point']
        }
```

### 4. Advanced Customization

#### System Prompts
Add context or instructions:

```python
def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
    return "You are a helpful assistant. Answer questions accurately and concisely."
```

#### Few-shot Examples
Customize how examples are formatted:

```python
def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
    """Format the answer for few-shot examples."""
    answer = self._get_ground_truth(item)
    return f"The answer is: {answer}"
```

#### Custom Sampling
Control how few-shot examples are selected:

```python
def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
    """Sample examples similar to the current item."""
    # Default: random sampling
    examples = self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)

    # Custom: sample by difficulty or topic
    # same_topic = [ex for ex in self.dataset[self.FEWSHOT_SPLIT]
    #               if ex['topic'] == item['topic']]
    # examples = self.rnd.sample(same_topic, min(self.num_fewshot, len(same_topic)))

    return examples
```

### 5. Metrics for Completion Tasks

Choose appropriate metrics based on your task type:

```python
# Exact match accuracy
from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion

# Text similarity metrics
from eval_framework.metrics.completion_metrics.rouge_1 import Rouge1
from eval_framework.metrics.completion_metrics.rouge_2 import Rouge2
from eval_framework.metrics.completion_metrics.rouge_l import RougeL
from eval_framework.metrics.completion_metrics.bleu import Bleu

# Math-specific metrics
from eval_framework.metrics.completion_metrics.math_reasoning_completion import MathReasoningCompletion

# Code execution metrics
from eval_framework.metrics.completion_metrics.code_execution_pass_at_one import CodeExecutionPassAtOne

# Format validation
from eval_framework.metrics.completion_metrics.json_format import JSONFormat
from eval_framework.metrics.completion_metrics.csv_format import CSVFormat

# Custom metrics using LLM judges
from eval_framework.metrics.llm_metrics.llm_judge_score import LLMJudgeScore

class YourTask(BaseTask[str]):
    # Choose metrics appropriate for your task
    METRICS = [AccuracyCompletion, Rouge1, MathReasoningCompletion]
```

## Complete Example: Geography Quiz

```python
from typing import Any
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.completion_metrics.accuracy_completion import AccuracyCompletion

class GeographyQuizTask(BaseTask[str]):
    NAME = "Geography Quiz"
    DATASET_PATH = "geography_quiz"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["world_capitals", "countries"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Format geography question."""
        return f"Question: What is the capital of {item['country']}?"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Extract the correct capital city."""
        return item['capital']

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        """Provide context about the task."""
        return "Answer geography questions about world capitals."

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Start model response with 'Answer:'"""
        return "Answer:"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        """Clean up the generated answer."""
        # Remove common prefixes and clean whitespace
        cleaned = completion_text.strip()
        if cleaned.startswith("Answer:"):
            cleaned = cleaned[7:].strip()
        return cleaned
```

## Testing Your Completion Task

All tasks automatically go through formatting tests to ensure proper prompt generation. However, if your benchmark has specific functionality that needs testing, create a dedicated test file.

#### Automatic Formatting Tests
All benchmarks are automatically tested for proper prompt formatting across different chat templates. No additional setup required.

#### Custom Task Tests (Optional)
If your benchmark has specific logic that needs testing, create a test file in `tests/tasks/` to test it.
```
