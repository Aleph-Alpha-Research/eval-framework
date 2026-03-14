# How to Add a New Benchmark to Eval Framework

This guide provides comprehensive instructions for adding new benchmarks to the eval-framework, including all possible configuration options and attributes.

## Overview

The eval-framework supports two response types:

1. **Completion Tasks** - Generate text completions (e.g., math problems, code generation)
2. **Loglikelihood Tasks** - Multiple choice questions where the model ranks answer options

For detailed information about implementing each task type, please refer to:

- [Completion Task Guide](completion_task_guide.md) - Comprehensive guide for text generation tasks
- [Loglikelihood Task Guide](loglikelihood_task_guide.md) - Detailed guide for multiple choice tasks

## Understanding the Base Task Structure

All benchmarks inherit from `BaseTask[SubjectType]` and must implement several required methods and class attributes.

### Required Class Attributes

```python
class YourBenchmark(BaseTask[str]):  # or BaseTask[Enum] for multiple subjects
    # === CORE CONFIGURATION ===
    NAME: str                           # Display name for the benchmark
    DATASET_PATH: str                   # HuggingFace dataset path or local path
    SAMPLE_SPLIT: str                   # Dataset split for evaluation samples
    FEWSHOT_SPLIT: str                  # Dataset split for few-shot examples
    RESPONSE_TYPE: ResponseType         # COMPLETION or LOGLIKELIHOODS
    METRICS: list[type[BaseMetric]]     # List of metric classes to compute
    SUBJECTS: list[SubjectType]         # List of subjects/categories to evaluate

    # === OPTIONAL CONFIGURATION ===
    HF_REVISION: str | None = None      # Git revision for reproducibility
    PERTURBATION_UNMODIFIABLE_WORDS: list[str] | None = None  # Words to protect from perturbation
    LANGUAGE: Language | dict[str, Language] | dict[str, tuple[Language, Language]] | None = None  # Language(s) tested
```

### Required Methods to Implement

```python
def _get_instruction_text(self, item: dict[str, Any]) -> str:
    """Generate the instruction/question text for a sample."""
    pass

def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
    """Extract the correct answer(s) from a dataset item."""
    pass
```

### Optional Methods to Override

```python
def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
    """Text to prepend to the first message."""
    return ""

def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
    """System message content."""
    return None

def _get_cue_text(self, item: dict[str, Any]) -> str:
    """Text to append as assistant cue (e.g., 'Answer:')."""
    return ""

def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
    """For loglikelihood tasks: list of answer choices."""
    return None

def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
    """Target text for few-shot examples."""
    target = self._get_ground_truth(item)
    assert target is not None and isinstance(target, str)
    return target

def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | list[BaseMetricContext] | None:
    """Additional parameters for evaluation metrics."""
    return None

def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
    """Custom few-shot sampling logic."""
    # Default implementation samples randomly from FEWSHOT_SPLIT
    pass

def _create_samples(self, item: dict[str, Any], index: int, subject: str) -> list[Sample]:
    """Create one or more samples from a dataset item."""
    # Default creates single sample - override for multi-sample items
    pass

def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
    """Post-process model completions (e.g., extract final answer)."""
    return completion_text
```

## Response Types, Metrics, and Configuration Attributes Reference

This section provides a complete reference for all configurations available when creating benchmarks.

### Response Types

The response type determines how your model interacts with the task and what type of output is expected.

```python
from eval_framework.models.sample import ResponseType

# For text generation tasks
RESPONSE_TYPE = ResponseType.COMPLETION

# For multiple choice tasks
RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
```

### All Available Metrics

Metrics define how your task's outputs are evaluated and scored. Choose metrics that align with your response type and evaluation goals.

#### Completion Metrics

These metrics work with generated text outputs from COMPLETION tasks:

```python
# Accuracy metrics
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.math_reasoning_completion import MathReasoningCompletion
from eval_framework.metrics.completion.cwe_accuracy import CWEAccuracy

# Translation and similarity metrics
from eval_framework.metrics.completion.bleu import BLEU
from eval_framework.metrics.completion.chrf import CHRF
from eval_framework.metrics.completion.ter import TER
from eval_framework.metrics.completion.rouge_1 import ROUGE_1
from eval_framework.metrics.completion.rouge_2 import ROUGE_2
from eval_framework.metrics.completion.rouge_l import ROUGE_L
from eval_framework.metrics.completion.rouge_geometric_mean import ROUGE_GEOMETRIC_MEAN
from eval_framework.metrics.completion.f1 import F1

# Code evaluation metrics
from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.metrics.completion.code_execution_pass_at_one import CodeExecutionPassAtOne

# Format validation metrics
from eval_framework.metrics.completion.json_format import JsonFormat
from eval_framework.metrics.completion.csv_format import CSVFormat
from eval_framework.metrics.completion.format_checker import CheckJsonFormat
from eval_framework.metrics.completion.format_checker import CheckPostScriptFormat


# Specialized metrics
from eval_framework.metrics.completion.ifeval import IFEvalMetric
from eval_framework.metrics.completion.language_checker import LanguageChecker
from eval_framework.metrics.completion.length_control import LengthControl
from eval_framework.metrics.completion.niah_accuracy import NIAHAccuracy
from eval_framework.metrics.completion.text_counter import WordCounter
from eval_framework.metrics.completion.text_counter import ParagraphCounter
from eval_framework.metrics.completion.text_counter import ResponseToOriginalLengthRatio

```

#### Loglikelihood Metrics

These metrics work with probability rankings from LOGLIKELIHOODS tasks:

```python
# Standard accuracy metrics
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import AccuracyLoglikelihood
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import AccuracyNormLoglikelihood

# Probability metrics
from eval_framework.metrics.loglikelihood.probability_mass import ProbabilityMass
from eval_framework.metrics.loglikelihood.probability_mass import ProbabilityMassNorm
```

#### LLM Judge Metrics

These metrics use another LLM to evaluate generated outputs, useful for complex or subjective tasks:

```python
from eval_framework.metrics.llm.llm_judge_chatbot_style import LLMJudgeChatbotStyle
# Classifies whether a text generation model's response follows a chatbot-style format by evaluating characteristics like friendly introductions, verbose language, follow-up questions, and conversational fluff, returning a boolean classification with reasoning. (English and German)

from eval_framework.metrics.llm.llm_judge_completion_accuracy import LLMJudgeCompletionAccuracy
# Evaluates whether a text generation model's response matches the target response by assessing if the generation addresses the question correctly and provides the right information, returning a boolean classification. (English)


from eval_framework.metrics.llm.llm_judge_conciseness import LLMJudgeConciseness
# Assesses if the model's response is concise by evaluating whether it is brief and to the point without unnecessary details or elaboration, returning a boolean classification with reasoning. (English and German)


from eval_framework.metrics.llm.llm_judge_contains_names import LLMJudgeAvoidsNames
# Checks whether the model text generation contains names of individuals, returning a boolean classification. (English, French and German)


from eval_framework.metrics.llm.llm_judge_format_correctness import LLMJudgeFormatCorrectness
# Evaluates the correctness and quality of models' outputs by checking if they comply with format specifications stated in the prompt, returning a detailed assessment with format correctness status (1 for correct, 0 for incorrect) and reasoning. (English)


from eval_framework.metrics.llm.llm_judge_instruction import LLMJudgeInstruction
# Evaluates how well the response from a text generation model fulfills the requirements of a given user instruction on a fixed set of criteria, where the user cannot ask follow-up questions to the model. (English, Finnish and German)


from eval_framework.metrics.llm.llm_judge_mtbench_pair import MTBenchJudgePair
# Evaluates the quality of two AI assistant responses by acting as an impartial judge, comparing factors like helpfulness, relevance, accuracy, and depth to determine which assistant provides a better answer (A wins, B wins, or tie). (English, Finnish and German)


from eval_framework.metrics.llm.llm_judge_mtbench_single import MTBenchJudgeSingle
# Evaluates a single AI assistant response on a 1-10 scale, considering factors like helpfulness, relevance, accuracy, depth, creativity, and detail level to provide an objective quality rating. (English, Finnish and German)


from eval_framework.metrics.llm.llm_judge_refusal import LLMJudgeRefusal
# Assesses whether the assistant's response is a refusal to answer the question. (English)


from eval_framework.metrics.llm.llm_judge_sql import LLMJudgeSql
# Evaluates SQL query quality against specified requirements, rating efficiency and accuracy on an A-F scale and providing detailed reasoning for the grade assigned. (English and German)


from eval_framework.metrics.llm.llm_judge_world_knowledge import LLMJudgeWorldKnowledge
# Evaluates whether a summary contains information that goes beyond the reference text (also known as "world knowledge"), returning a boolean classification with detailed reasoning for the assessment. (English, French and German)

```

## Implementation Examples and Patterns

### Practical Example: GeographyQATask

Practical example of creating a geography question-answering benchmark.

**Dataset Structure:** Each item looks like `{"country": "Germany", "capital": "Berlin"}`

```python
from typing import Any
from eval_framework.tasks.base import BaseTask
from eval_framework.models.sample import ResponseType
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion

class GeographyQATask(BaseTask[str]):
    # Required class attributes
    NAME = "GeographyQA"
    DATASET_PATH = "example/geography_qa"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["Europe", "Asia"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Format the question from the dataset item."""
        return f"Q: What's the capital of {item['country']}?"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Extract the correct answer from the dataset item."""
        return f"A: {item['capital']}."

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str:
        """Provide context about the task."""
        return "Answer the geography questions accurately."

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        """Start the model's response with 'A:'."""
        return "A:"

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        """Sample random examples from the training split."""
        return self.rnd.sample(self.dataset[self.FEWSHOT_SPLIT], self.num_fewshot)
```

### Add to Task Registry

Add a registration call for your new benchmark to `register_all_tasks` in `src/eval_framework/tasks/task_names.py`:

```python
register_lazy_task("eval_framework.tasks.benchmarks.geographyqa.GeographyQA")
```

The task will now be available through `get_task("GeographyQA")`.

### Testing your benchmark

All tasks automatically go through formatting tests to ensure proper prompt generation. The formatting test lives in `tests/tests_eval_framework/tasks/test_all_formatters.py` and runs all registered tasks automatically.

> [!TIP]
> CI runs this test across all registered tasks, but during development you can target only your task, for example:
> `uv run pytest tests/tests_eval_framework/tasks/test_all_formatters.py -k "YourTaskName"`

#### Automatic Formatting Tests

All benchmarks are automatically tested for proper prompt formatting across different chat templates. If your new task needs non-default initialization arguments (for example, a specific `num_fewshot`), add an entry for your task to `SPECIAL_ARGS` in `tests/tests_eval_framework/tasks/test_all_formatters.py`.

The expected formatter outputs are tracked as hashes in `tests/tests_eval_framework/tasks/task-prompts-hashes.json`.

When you add a new task:

1. Run the formatter hash test once for your task to generate/check hashes.
2. If your task hash is new, it will be added to `task-prompts-hashes.json`.
3. Commit the updated JSON file together with your task changes.

Run the formatter hash test only for your newly created task (replace `YourTaskName`):

```bash
uv run pytest tests/tests_eval_framework/tasks/test_all_formatters.py -m formatter_hash -k "YourTaskName"
```

#### Custom Task Tests (Optional)

If your benchmark has specific logic that needs testing, create a test file in `tests/tasks/` to test it.

### Update benchmark documentation

After adding a benchmark, you also need to update task documentation:

1. Manually add the new benchmark name(s) to `docs/benchmarks_and_metrics.md` (including `*_IDK` variants if your benchmark has them).
2. Regenerate the task docs:

```bash
uv run -m eval_framework.utils.generate_task_docs
```

This updates `docs/tasks/README.md` and creates per-task documentation files for new tasks in `docs/tasks/`.

## Benchmark Examples by Task Type

Study these existing benchmarks in the codebase for more complex patterns:

#### Simple Classification Tasks

- **ARC** (`src/eval_framework/tasks/arc.py`): Multiple choice with loglikelihoods
- **MMLU** (`src/eval_framework/tasks/mmlu.py`): Multi-subject classification with enum subjects

#### Reasoning Tasks

- **GSM8K** (`src/eval_framework/tasks/gsm8k.py`): Math reasoning with answer extraction patterns

#### Code Generation

- **HumanEval** (`src/eval_framework/tasks/human_eval.py`): Code completion with execution validation
- **MBPP** (`src/eval_framework/tasks/mbpp.py`): Code generation with comprehensive test validation

#### Long Context Tasks

- **InfiniteBench** (`src/eval_framework/tasks/infinite_bench_tasks.py`): Long context reasoning tasks

#### Custom Format Tasks

- **IFEval** (`src/eval_framework/tasks/ifeval.py`): Instruction following with format validation
- **JSON/CSV Tasks:** Custom format validation examples
