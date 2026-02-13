# LLM as Judge Evaluation Guide

This guide provides comprehensive documentation for using LLM-as-a-judge evaluation in the eval-framework. LLM judges leverage language models to evaluate the quality, correctness, and various other aspects of model outputs.

## Table of Contents

- [Architecture](#architecture)
- [Available LLM Judge Metrics](#available-llm-judge-metrics)
- [Configuration](#configuration)
  - [CLI Configuration](#cli-configuration)
  - [Python API Configuration](#python-api-configuration)
- [Adding a New Benchmark with LLM Judges](#adding-a-new-benchmark-with-llm-judges)
- [Advanced: Using LLM Judges for Generation Control](#advanced-using-llm-judges-for-generation-control)
- [Appendix](#appendix)


---

## Architecture

The LLM judge system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    EvalConfig / CLI                         │
│     (llm_judge_class, judge_model_args, judge_model_name)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  BaseLLMJudgeMetric                         │
│        (Base class for all LLM judge metrics)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     Graders Layer                           │
│  (InstructionGrader, ComparisonGrader, ChatbotStyleGrader)  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     BaseLLM (Judge Model)                   │
│       (OpenAIModel, HFLLM, VLLM, or custom model)           │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| `BaseLLMJudgeMetric` | Abstract base class for all LLM judge metrics |
| `Graders` | Specialized classes that format prompts and parse judge outputs |
| `PromptTemplate` | Defines system and user prompts for the judge |
| `GradingOutput` | Structured output from graders containing judge response |
| `MetricResult` | Final result including value, judge prompt, and judge response |

---

## Available LLM Judge Metrics

The framework provides a comprehensive set of pre-built LLM judge metrics. The metric which is used in a given benchmark task is specified in the task implementation class with `METRICS = [...]`.

### Quality & Style Metrics

| Metric | Class | Description | Languages |
|--------|-------|-------------|-----------|
| **Chatbot Style** | `LLMJudgeChatbotStyle` | Evaluates if responses follow chatbot conventions (friendly intro, verbose language, follow-up questions) | EN, DE |
| **Conciseness** | `LLMJudgeConciseness` | Assesses if responses are brief and to the point without unnecessary elaboration | EN, DE |
| **Coherence** | `LLMJudgeCoherence` | Evaluates logical flow and consistency of responses | EN |

### Correctness Metrics

| Metric | Class | Description | Languages |
|--------|-------|-------------|-----------|
| **Completion Accuracy** | `LLMJudgeCompletionAccuracy` | Evaluates if the model response matches the expected answer | EN |
| **Instruction Following** | `LLMJudgeInstruction` | Comprehensive evaluation of instruction adherence with multiple sub-metrics | EN, DE, FI |
| **Format Correctness** | `LLMJudgeFormatCorrectness` | Validates if output follows specified format requirements | EN |

### Specialized Metrics

| Metric | Class | Description | Languages |
|--------|-------|-------------|-----------|
| **SQL Quality** | `LLMJudgeSql` | Evaluates SQL query quality, efficiency, and accuracy (A-F grade) | EN, DE |
| **World Knowledge** | `LLMJudgeWorldKnowledge` | Detects if summaries contain information beyond the source text | EN, FR, DE |
| **Avoids Names** | `LLMJudgeAvoidsNames` | Checks if responses avoid using personal names | EN, FR, DE |
| **Refusal Classification** | `LLMJudgeRefusal` | Detects if the model refused to answer | EN |

### Comparison Metrics (MT-Bench Style)

| Metric | Class | Description | Languages |
|--------|-------|-------------|-----------|
| **Pairwise Judgement** | `MTBenchJudgePair` | Compares two responses and selects the better one (A wins, B wins, tie) | EN, DE, FI |
| **Single Judgement** | `MTBenchJudgeSingle` | Rates a single response on a 1-10 scale | EN, DE, FI |

### Multi-Key Metrics

Some metrics return multiple evaluation keys:

**`LLMJudgeInstruction`** returns:
- `quality` - Overall quality score (normalized 0-1)
- `is_following_instruction` - Boolean instruction adherence
- `has_correct_grammar_and_spelling` - Boolean grammar check
- `is_context_consistent` - Boolean consistency with context
- `is_not_repeating` - Boolean repetition check
- `is_trustworthy` - Boolean truthfulness check
- `is_safe` - Boolean safety check

**`LLMJudgeCoherence`** returns:
- `coherence_score` - Overall coherence rating
- `is_coherent` - Boolean coherence flag
- `has_repetition` - Boolean repetition detection

---

## Configuration

### CLI Configuration

To use LLM judges via the command line:

```bash
uv run eval_framework \
    --models path/to/your/models.py \
    --llm-name YourModelToEvaluate \
    --task-name YourTaskName \
    --judge-models path/to/judge_models.py \
    --judge-model-name OpenAI_gpt_4o_mini \
    --judge-model-args api_key="your-api-key" \
    --output-dir ./eval_results \
    --num-samples 100
```

#### Judge-Specific CLI Arguments

| Argument | Description |
|----------|-------------|
| `--judge-models` | Path to Python module containing judge model classes |
| `--judge-model-name` | Name of the judge model class to instantiate |
| `--judge-model-args` | Key=value pairs for judge model constructor arguments |
| `--randomize-judge-order` | Enable position randomization for pairwise comparisons |

### Python API Configuration

```python
from pathlib import Path
from eval_framework.llm.openai import OpenAI_gpt_4o_mini
from eval_framework.llm.huggingface import HFLLM
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig

# Define your model to evaluate
class MyModel(HFLLM):
    LLM_NAME = "your-model-name"

# Configure evaluation with LLM judge
config = EvalConfig(
    task_name="YourTaskName",
    llm_class=MyModel,
    llm_judge_class=OpenAI_gpt_4o_mini,  # Judge model class
    judge_model_args={                   # Judge model arguments
        "api_key": "your-api-key",
        "temperature": 0.0,              # Lower temperature for consistent judging
    },
    output_dir=Path("./eval_results"),
    num_samples=100,
    randomize_judge_order=True,          # Mitigate position bias
)

# Run evaluation
llm = MyModel()
results = main(llm=llm, config=config)
```

### Using Different Judge Models

#### OpenAI Models

```python
from eval_framework.llm.openai import OpenAIModel

# Using pre-defined alias
from eval_framework.llm.openai import OpenAI_gpt_4o_mini

# Or configure directly
class CustomOpenAIJudge(OpenAIModel):
    LLM_NAME = "gpt-4-turbo"

config = EvalConfig(
    llm_judge_class=CustomOpenAIJudge,
    judge_model_args={
        "api_key": "your-api-key",
        "temperature": 0.0,
    },
    # ...
)
```

#### Deepseek Models

```python
from eval_framework.llm.openai import Deepseek_chat

config = EvalConfig(
    llm_judge_class=Deepseek_chat,
    judge_model_args={
        # Uses DEEPSEEK_API_KEY env variable by default
    },
    # ...
)
```

#### Local vLLM Models

```python
from eval_framework.llm.vllm import VLLM

class LocalJudge(VLLM):
    LLM_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"

config = EvalConfig(
    llm_judge_class=LocalJudge,
    judge_model_args={
        "tensor_parallel_size": 4,
        "max_model_len": 8192,
    },
    # ...
)
```

---

## Adding a New Benchmark with LLM Judges

This section provides a complete guide to creating a new benchmark task that uses LLM judge metrics for evaluation.

### Step 1: Define Your Task Class

Every benchmark inherits from `BaseTask[SubjectType]` and requires these class attributes:

```python
from typing import Any
from eval_framework.tasks.base import BaseTask, ResponseType, Sample, Language

class YourBenchmark(BaseTask[str]):
    # Required attributes
    NAME: str = "YourBenchmarkName"              # Display name
    DATASET_PATH: str = "huggingface/dataset"    # HuggingFace dataset path
    SAMPLE_SPLIT: str = "test"                   # Split for evaluation samples
    FEWSHOT_SPLIT: str = "train"                 # Split for few-shot examples
    RESPONSE_TYPE: ResponseType                  # COMPLETION or LOGLIKELIHOODS
    METRICS: list[type[BaseMetric]]              # List of metrics to compute
    SUBJECTS: list[str]                          # Subjects/categories

    # Optional attributes
    LANGUAGE: Language | None = Language.ENG     # Primary language
    HF_REVISION: str | None = None               # Dataset version pin
```

### Step 2: Choose Your LLM Judge Metrics

Select from the available LLM judge metrics based on your evaluation requirements:

```python
# Quality evaluation
from eval_framework.metrics.llm.llm_judge_instruction import LLMJudgeInstruction
from eval_framework.metrics.llm.llm_judge_chatbot_style import LLMJudgeChatbotStyle
from eval_framework.metrics.llm.llm_judge_conciseness import LLMJudgeConciseness
from eval_framework.metrics.llm.llm_judge_coherence import LLMJudgeCoherence

# Correctness evaluation
from eval_framework.metrics.llm.llm_judge_completion_accuracy import LLMJudgeCompletionAccuracy
from eval_framework.metrics.llm.llm_judge_format_correctness import LLMJudgeFormatCorrectness

# Specialized evaluation
from eval_framework.metrics.llm.llm_judge_sql import LLMJudgeSql
from eval_framework.metrics.llm.llm_judge_world_knowledge import LLMJudgeWorldKnowledge
from eval_framework.metrics.llm.llm_judge_contains_names import LLMJudgeAvoidsNames
from eval_framework.metrics.llm.llm_judge_refusal import LLMJudgeRefusal

# Comparison evaluation
from eval_framework.metrics.llm.llm_judge_mtbench_pair import MTBenchJudgePair
from eval_framework.metrics.llm.llm_judge_mtbench_single import MTBenchJudgeSingle
```

### Step 3: Implement Required Methods

Every benchmark must implement these core methods:

```python
def _get_instruction_text(self, item: dict[str, Any]) -> str:
    """Generate the instruction/question for the model."""
    pass

def _get_ground_truth(self, item: dict[str, Any]) -> str | list[str] | None:
    """Extract the expected answer(s) from a dataset item."""
    pass
```

### Step 4: Provide Context for Judge Metrics

Many LLM judge metrics require additional context via the `_get_context` method:

```python
from eval_framework.shared.types import LanguageMetricContext, BaseMetricContext

def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | None:
    """Provide additional context for metric evaluation."""
    return LanguageMetricContext(
        language=item.get("language", "en"),
    )
```

### Step 5: Create Custom Metrics (If Needed)

If the pre-built LLM judge metrics don't cover your evaluation requirements, you can create custom metrics.

#### Basic Custom Metric

Create a new LLM judge metric by extending `BaseLLMJudgeMetric`:

```python
from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role

class CustomJudgeMetric(BaseLLMJudgeMetric):
    NAME = "Custom Judge Metric"

    def __init__(self, llm_judge: BaseLLM, randomize_order: bool = False):
        super().__init__(llm_judge, randomize_order)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(
                metric_name=self.NAME,
                value=None,
                higher_is_better=True,
                error=response.error,
            )]

        # Create judge prompt
        judge_prompt = f"""Evaluate the following response for quality.

Question: {response.system_user_instruction}

Response: {response.sanitized_completion}

Rate the quality on a scale of 1-5, where 5 is excellent.
Respond with ONLY a JSON object: {{"score": <number>, "reasoning": "<explanation>"}}"""

        # Get judge response
        messages = [Message(role=Role.USER, content=judge_prompt)]
        output = self._llm_judge.generate_from_messages([messages])

        # Parse result (implement your parsing logic)
        import json
        try:
            parsed = json.loads(output[0].completion)
            score = parsed.get("score", 3) / 5.0  # Normalize to 0-1
        except:
            score = None

        return [MetricResult(
            metric_name=self.NAME,
            value=score,
            higher_is_better=True,
            llm_judge_prompt=judge_prompt,
            llm_judge_response=output[0].completion,
        )]
```

#### Creating a Custom Grader

For more sophisticated evaluation, create a custom grader:

```python
from collections.abc import Mapping
from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm.graders.models import (
    GradingOutput,
    PromptTemplate,
    parse_json_output,
)
from eval_framework.metrics.llm.graders.language import Language

class CustomGradingOutput(GradingOutput):
    """Structured output from the grader."""
    quality_score: float | None
    has_errors: bool | None
    feedback: str | None

class CustomGrader:
    RESPONSE_KEY = "response"
    CRITERIA_KEY = "criteria"

    PROMPT_TEMPLATES = {
        Language("en"): PromptTemplate(
            system_prompt="""You are an expert evaluator. Assess the given response
based on the specified criteria.

Provide your evaluation as JSON:
{
    "quality_score": float (0.0 to 1.0),
    "has_errors": bool,
    "feedback": str
}""",
            user_prompt=f"""**Response to Evaluate**:
{{{RESPONSE_KEY}}}

**Evaluation Criteria**:
{{{CRITERIA_KEY}}}""",
        ),
    }

    def __init__(
        self,
        grading_model: BaseLLM,
        prompt_templates: Mapping[Language, PromptTemplate] = PROMPT_TEMPLATES,
    ):
        self._grading_model = grading_model
        self._prompt_templates = prompt_templates

    def grade(
        self,
        response: str,
        criteria: str,
        language: Language,
    ) -> CustomGradingOutput:
        try:
            prompt_template = language.language_config(self._prompt_templates)
        except:
            prompt_template = Language("en").language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],  # system key-value pairs
            [    # user key-value pairs
                (self.RESPONSE_KEY, response),
                (self.CRITERIA_KEY, criteria),
            ],
        )

        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return CustomGradingOutput(
            quality_score=loaded_json.get("quality_score"),
            has_errors=loaded_json.get("has_errors"),
            feedback=loaded_json.get("feedback"),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
```

#### Using the Custom Grader in a Metric

```python
class CustomGraderMetric(BaseLLMJudgeMetric):
    NAME = "Custom Grader Metric"

    def __init__(self, llm_judge: BaseLLM, randomize_order: bool = False):
        super().__init__(llm_judge, randomize_order)
        self._grader = CustomGrader(llm_judge)

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [self._create_metric_result(
                metric_name=self.NAME,
                higher_is_better=True,
                value=None,
                error=response.error,
            )]

        language = Language(response.get_instruction_language())

        grading = self._grader.grade(
            response=response.sanitized_completion,
            criteria="Evaluate for accuracy, clarity, and completeness.",
            language=language,
        )

        return [self._create_metric_result(
            metric_name=self.NAME,
            higher_is_better=True,
            value=grading.quality_score,
            llm_judge_prompt=grading.judge_prompt,
            llm_judge_response=grading.judge_response,
        )]
```

### Step 6: Register Your Benchmark

Add your benchmark to the task registry in `src/eval_framework/tasks/task_names.py`:

```python
# In register_all_tasks() function
register_lazy_task("eval_framework.tasks.benchmarks.your_module.WritingQualityBenchmark")
register_lazy_task("eval_framework.tasks.benchmarks.your_module.SQLGenerationBenchmark")
```

### Step 7: Run Your Benchmark

**Via CLI:**

```bash
uv run eval_framework \
    --models path/to/models.py \
    --llm-name YourModel \
    --task-name WritingQuality \
    --task-subjects "creative_writing" \
    --judge-models eval_framework.llm.openai \
    --judge-model-name OpenAI_gpt_4o_mini \
    --judge-model-args api_key="$OPENAI_API_KEY" \
    --output-dir ./eval_results \
    --num-samples 50
```

**Via Python:**

```python
from pathlib import Path
from eval_framework.llm.openai import OpenAI_gpt_4o_mini
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig

# Your model class
from your_models import YourModel

config = EvalConfig(
    task_name="WritingQuality",
    task_subjects=["creative_writing", "technical_writing"],
    llm_class=YourModel,
    llm_judge_class=OpenAI_gpt_4o_mini,
    judge_model_args={"api_key": "your-api-key", "temperature": 0.0},
    output_dir=Path("./eval_results"),
    num_samples=50,
    num_fewshot=3,
    randomize_judge_order=True,  # For fair pairwise comparisons
)

llm = YourModel()
results = main(llm=llm, config=config)
```


### Notes on LLM Judge Tasks

1. **Judge Configuration**: When running a task with LLM judge metrics, you must configure the judge model or you'll get:
   ```
   AssertionError: The LLM Judge must be defined for this evaluation task.
   ```

2. **Context Matching**: Ensure your `_get_context` returns the correct context type for your metrics:
   - `LanguageMetricContext` - Most metrics
   - `LLMJudgeSqlMetricContext` - SQL evaluation
   - `MTBenchJudgePairMetricContext` - Pairwise comparison

3. **Language Support**: Check that your chosen metrics is compatible with the languages in your dataset.

---

## Advanced: Using LLM Judges for Generation Control

Some tasks use LLM judge graders not for evaluation, but to control the generation process itself.

**Example:** [AidanBench](https://openreview.net/pdf?id=fz969ahcvJ) uses `CoherenceGrader` during iterative generation to decide when to stop. The grader checks each new response for coherence, and stops generating when quality drops below a threshold. The final metric simply counts how many coherent responses were generated.

For implementation details, see [`src/eval_framework/tasks/benchmarks/aidanbench.py`](../src/eval_framework/tasks/benchmarks/aidanbench.py).

---

## Appendix


### Import Reference

```python
# Judge metrics
from eval_framework.metrics.llm.llm_judge_instruction import LLMJudgeInstruction
from eval_framework.metrics.llm.llm_judge_chatbot_style import LLMJudgeChatbotStyle
from eval_framework.metrics.llm.llm_judge_completion_accuracy import LLMJudgeCompletionAccuracy
from eval_framework.metrics.llm.llm_judge_conciseness import LLMJudgeConciseness
from eval_framework.metrics.llm.llm_judge_coherence import LLMJudgeCoherence
from eval_framework.metrics.llm.llm_judge_format_correctness import LLMJudgeFormatCorrectness
from eval_framework.metrics.llm.llm_judge_sql import LLMJudgeSql
from eval_framework.metrics.llm.llm_judge_world_knowledge import LLMJudgeWorldKnowledge
from eval_framework.metrics.llm.llm_judge_contains_names import LLMJudgeAvoidsNames
from eval_framework.metrics.llm.llm_judge_refusal import LLMJudgeRefusal
from eval_framework.metrics.llm.llm_judge_mtbench_pair import MTBenchJudgePair
from eval_framework.metrics.llm.llm_judge_mtbench_single import MTBenchJudgeSingle

# Base classes for custom metrics
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.metrics.llm.graders.models import GradingOutput, PromptTemplate

# Judge model classes
from eval_framework.llm.openai import OpenAIModel, OpenAI_gpt_4o_mini, Deepseek_chat
from eval_framework.llm.vllm import VLLM
from eval_framework.llm.huggingface import HFLLM
```
