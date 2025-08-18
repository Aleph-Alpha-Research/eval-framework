# Creating Loglikelihood Tasks

This guide shows you how to create **loglikelihood tasks**, benchmarks where the model chooses between predefined answer options by ranking their probabilities (e.g., multiple choice questions, classification tasks).

## Quick Start Template

```python
from typing import Any
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood

class YourLoglikelihoodTask(BaseTask[str]):
    # Required attributes
    NAME = "YourTaskName"
    DATASET_PATH = "your-dataset/path"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["default"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Generate the question prompt."""
        return f"Question: {item['question']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Return the correct answer choice."""
        return item['correct_answer']

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        """Return all answer choices for ranking."""
        return item['choices']
```

## Step-by-Step Implementation

### 1. Basic Setup

Start with the minimal structure:

```python
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood

class MultipleChoiceTask(BaseTask[str]):
    NAME = "Multiple Choice"
    DATASET_PATH = "mcq_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["general"]
```

### 2. Implement Required Methods

#### _get_instruction_text()
Format the question without answer choices:

```python
def _get_instruction_text(self, item: dict[str, Any]) -> str:
    """Create the question prompt."""
    # Simple question
    return f"Question: {item['question']}"

    # With context
    # return f"Context: {item['context']}\nQuestion: {item['question']}"

    # With instructions
    # return f"Choose the best answer:\n{item['question']}"
```

#### _get_ground_truth()
Return the correct answer choice:

```python
def _get_ground_truth(self, item: dict[str, Any]) -> str:
    """Return the correct answer choice."""
    # If dataset has answer index
    correct_idx = item['answer_idx']
    return item['choices'][correct_idx]

    # If dataset has answer directly
    # return item['correct_answer']

    # If dataset has answer key (A, B, C, D)
    # answer_key = item['answer_key']  # e.g., 'B'
    # key_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    # return item['choices'][key_to_idx[answer_key]]
```

#### _get_possible_completions()
Return all answer choices:

```python
def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
    """Return all answer choices for probability ranking."""
    return item['choices']

    # If choices need formatting
    # return [f" {choice}" for choice in item['choices']]  # Add leading space

    # If choices are labeled
    # return [f" {label}) {choice}" for label, choice in zip(['A', 'B', 'C', 'D'], item['choices'])]
```

### 3. Common Loglikelihood Task Patterns

#### Pattern 1: Standard Multiple Choice
```python
from typing import Any
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood

class StandardMCQTask(BaseTask[str]):
    NAME = "Standard MCQ"
    DATASET_PATH = "mcq_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["general"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\nAnswer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return item['choices'][item['answer_idx']]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        # Add leading space for better tokenization
        return [f" {choice}" for choice in item['choices']]
```

#### Pattern 2: MMLU-style with Labeled Choices
```python
class MMLUStyleTask(BaseTask[str]):
    NAME = "MMLU Style"
    DATASET_PATH = "mmlu_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["abstract_algebra", "anatomy", "astronomy"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        choices_text = "\n".join([f"{label}. {choice}"
                                 for label, choice in zip(['A', 'B', 'C', 'D'], item['choices'])])
        return f"Question: {item['question']}\n{choices_text}\nAnswer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        answer_key = item['answer']  # 'A', 'B', 'C', or 'D'
        return f" {answer_key}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        return [" A", " B", " C", " D"]
```

#### Pattern 3: True/False Classification
```python
class TrueFalseTask(BaseTask[str]):
    NAME = "True/False"
    DATASET_PATH = "tf_dataset"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = ["facts"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Statement: {item['statement']}\nTrue or False?"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        return " True" if item['is_true'] else " False"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        return [" True", " False"]
```


### 4. Advanced Customization

#### Few-shot Examples
Format examples consistently:

```python
def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
    """Format the answer for few-shot examples."""
    # For MMLU-style: return the letter
    return item['answer']  # 'A', 'B', 'C', or 'D'

    # For true/false: return the word
    # return "True" if item['is_true'] else "False"
```

#### Context and Instructions
Add helpful context:

```python
def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
    return "You are an expert in multiple choice questions. Choose the best answer."

def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
    return "Instructions: Select the most appropriate answer from the given choices."
```

#### Multi-subject Tasks
Handle different subjects:

```python
from enum import Enum

class MMLUSubject(Enum):
    ABSTRACT_ALGEBRA = "abstract_algebra"
    ANATOMY = "anatomy"
    ASTRONOMY = "astronomy"
    # ... more subjects

class MMLUTask(BaseTask[MMLUSubject]):
    NAME = "MMLU"
    DATASET_PATH = "mmlu"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = list(MMLUSubject)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        subject = item['subject'].replace('_', ' ').title()
        choices_text = "\n".join([f"{label}. {choice}"
                                 for label, choice in zip(['A', 'B', 'C', 'D'], item['choices'])])
        return f"The following is a multiple choice question about {subject}.\n\n{item['question']}\n{choices_text}\nAnswer:"
```

### 5. Metrics for Loglikelihood Tasks

Choose appropriate metrics:

```python
# Standard accuracy
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood

# Normalized accuracy (for unbalanced datasets)
from eval_framework.metrics.loglikelihood_metrics.accuracy_norm_loglikelihood import AccuracyNormLoglikelihood

# Probability mass analysis
from eval_framework.metrics.loglikelihood_metrics.probability_mass import ProbabilityMass
from eval_framework.metrics.loglikelihood_metrics.probability_mass_norm import ProbabilityMassNorm

class YourTask(BaseTask[str]):
    # Most common choice
    METRICS = [AccuracyLoglikelihood]

    # For unbalanced datasets
    # METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]

    # For probability analysis
    # METRICS = [AccuracyLoglikelihood, ProbabilityMass]

    # For comprehensive analysis
    # METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ProbabilityMass, ProbabilityMassNorm]
```


## Complete Example: Science Quiz

```python
from typing import Any
from enum import Enum
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import AccuracyLoglikelihood

class ScienceSubject(Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"

class ScienceQuizTask(BaseTask[ScienceSubject]):
    NAME = "Science Quiz"
    DATASET_PATH = "science_quiz"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood]
    SUBJECTS = list(ScienceSubject)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Format science question with choices."""
        subject = item['subject'].replace('_', ' ').title()
        choices_text = "\n".join([f"{label}. {choice}"
                                 for label, choice in zip(['A', 'B', 'C', 'D'], item['choices'])])
        return f"Science ({subject}) Question:\n{item['question']}\n\n{choices_text}\n\nAnswer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Return the correct answer letter."""
        return f" {item['answer_key']}"  # ' A', ' B', ' C', or ' D'

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str]:
        """Return answer choice letters."""
        return [" A", " B", " C", " D"]

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        return "You are a science expert. Choose the correct answer for each question."

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        """Format target for few-shot examples."""
        return item['answer_key']  # 'A', 'B', 'C', or 'D' (no leading space)
```

## Testing Your Completion Task

All tasks automatically go through formatting tests to ensure proper prompt generation. However, if your benchmark has specific functionality that needs testing, create a dedicated test file.

#### Automatic Formatting Tests
All benchmarks are automatically tested for proper prompt formatting across different chat templates. No additional setup required.

#### Custom Task Tests (Optional)
If your benchmark has specific logic that needs testing, create a test file in `tests/tasks/` to test it.
```
