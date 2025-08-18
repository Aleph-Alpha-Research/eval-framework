## Overview Dataloading

To evaluate models on benchmarks, we define custom tasks that inherit from `BaseTask` to handle dataset loading and formatting. The framework supports two main evaluation types: **completion tasks** (text generation) and **loglikelihood tasks** (multiple choice ranking).

### Core Data Types

The framework uses different data types based on the evaluation approach:

```python
from eval_framework.shared.types import Completion, Loglikelihood, RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message, Role

# For completion tasks (text generation)
class Completion(BaseModel):
    completion_text: str                        # Generated text from the model
    # Additional fields based on actual implementation

# For loglikelihood tasks (multiple choice)
class Loglikelihood(BaseModel):
    loglikelihoods: list[float]                 # Probability scores for each choice
    # Additional fields based on actual implementation

# Raw response types from LLMs
class RawCompletion(BaseModel):
    text: str                                   # Raw generated text
    # Additional fields based on actual implementation

class RawLoglikelihood(BaseModel):
    loglikelihoods: list[float]                 # Raw probability scores
    # Additional fields based on actual implementation
```

### Message Structure

Each prompt is structured as a sequence of messages using the template formatting system:

```python
from template_formatting.formatter import Message, Role

class Message(BaseModel):
    role: Role                                  # SYSTEM, USER, or ASSISTANT
    content: str                                # Message content
    # Additional fields based on actual formatter implementation
```

### Task Implementation Pattern

Custom tasks inherit from `BaseTask` and implement specific methods based on their evaluation type:

#### For Completion Tasks:
```python
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType

class MyCompletionTask(BaseTask[str]):
    NAME = "My Task"
    DATASET_PATH = "dataset_name"
    RESPONSE_TYPE = ResponseType.COMPLETION

    def _get_instruction_text(self, item: dict) -> str:
        """Format the question/instruction."""
        return f"Question: {item['question']}"

    def _get_ground_truth(self, item: dict) -> str:
        """Return the expected answer."""
        return item['answer']
```

#### For Loglikelihood Tasks:
```python
class MyLoglikelihoodTask(BaseTask[str]):
    NAME = "My Task"
    DATASET_PATH = "dataset_name"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS

    def _get_instruction_text(self, item: dict) -> str:
        """Format the question without choices."""
        return f"Question: {item['question']}"

    def _get_ground_truth(self, item: dict) -> str:
        """Return the correct answer choice."""
        return item['choices'][item['answer_idx']]

    def _get_possible_completions(self, item: dict) -> list[str]:
        """Return all answer choices for ranking."""
        return item['choices']
```

### Few-Shot Example Construction

The framework automatically constructs few-shot prompts using these methods:

```python
# Example prompt construction for a two-shot scenario
def construct_prompt(self, item: dict) -> list[Message]:
    messages = []

    # 1. System prompt (optional)
    if system_prompt := self._get_system_prompt_text(item):
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))

    # 2. Few-shot examples
    fewshot_examples = self._sample_fewshot_examples(item)
    for example in fewshot_examples:
        # User instruction
        messages.append(Message(
            role=Role.USER,
            content=self._get_instruction_text(example)
        ))
        # Assistant response
        messages.append(Message(
            role=Role.ASSISTANT,
            content=self._get_fewshot_target_text(example)
        ))

    # 3. Actual instruction
    messages.append(Message(
        role=Role.USER,
        content=self._get_instruction_text(item)
    ))

    # 4. Response cue (optional)
    if cue := self._get_cue_text(item):
        messages.append(Message(role=Role.ASSISTANT, content=cue))

    return messages
```

### Example: Geography Quiz

Here's how a complete geography quiz task might look:

```python
messages = [
    Message(Role.SYSTEM, "Answer geography questions accurately."),
    Message(Role.USER, "Question: What is the capital of Germany?"),
    Message(Role.ASSISTANT, "Answer: Berlin"),
    Message(Role.USER, "Question: What is the capital of France?"),
    Message(Role.ASSISTANT, "Answer: Paris"),
    Message(Role.USER, "Question: What is the capital of Italy?"),
    Message(Role.ASSISTANT, "Answer:")
]
```

For **completion tasks**, the model generates the complete answer: `" Rome"`

For **loglikelihood tasks**, the model ranks options: `[" Rome", " Madrid", " Athens", " Vienna"]`
