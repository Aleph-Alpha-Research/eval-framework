# Task Suites

The eval-framework supports running multiple benchmarks in a single command through task suites. A suite defines a set of tasks, their hyperparameters, and how results should be aggregated. The `--task-suite` flag is mutually exclusive with `--task-name`.

## Suite Structure

A suite is a tree of `TaskSuite` nodes:

- **Leaf node** — points to a single registered task by name.
- **Composite node** — contains a list of children (leaves or other composites), and optionally defines metric aggregates over them.

Hyperparameters set on a composite node serve as defaults for all descendants. Children override only the fields they explicitly set.

```
Top-level suite  (lowest precedence)
  └─ Sub-suite
       └─ Leaf task  (highest precedence)
```

## File Formats

Suites can be defined in YAML or Python. Python supports composition via imports and computed values and is the recommended format.

### Python (Preferred)

Python suite files must expose a module-level variable named `suite` of type `TaskSuite`:

```python
from eval_framework.suite import SuiteAggregate, TaskSuite

suite = TaskSuite(
    name="math-reasoning",
    temperature=0.6,
    top_p=0.6,
    max_tokens=1024,
    tasks=[
        TaskSuite(tasks="GSM8K_OLMES", num_fewshot=8),
        TaskSuite(tasks="MATHMinerva_OLMES", num_fewshot=4),
    ],
    aggregates=[
        SuiteAggregate(name="avg_math_accuracy", metric="Accuracy Completion"),
    ],
)
```

### YAML

```yaml
name: math-reasoning
temperature: 0.6
top_p: 0.6
max_tokens: 1024

tasks:
  - tasks: GSM8K_OLMES
    num_fewshot: 8
  - tasks: MATHMinerva_OLMES
    num_fewshot: 4

aggregates:
  - name: avg_math_accuracy
    metric: Accuracy Completion
    method: mean
```

`temperature`, `top_p`, and `max_tokens` are inherited by both tasks. Each task overrides only `num_fewshot`.


### Composing suites from other files

Python suites can import and reuse sub-suites defined in other files:

```python
from mathsuite import suite as math_tasks
from stemqa_mc import suite as stemqa_mc_tasks

from eval_framework.suite import SuiteAggregate, TaskSuite

suite = TaskSuite(
    name="mysuite",
    tasks=[math_tasks, stemqa_mc_tasks],
    aggregates=[
        SuiteAggregate(name="olmo", metric="accuracy", method="mean"),
    ],
)
```

## Hyperparameters

The following fields can be set on any suite node:

| Field | Description |
|---|---|
| `temperature` | Sampling temperature |
| `top_p` | Top-p nucleus sampling |
| `top_k` | Top-k sampling |
| `extra_llm_args` | Additional arguments passed directly to the LLM |
| `num_samples` | Number of samples per subject |
| `num_fewshot` | Number of few-shot examples |
| `max_tokens` | Maximum tokens to generate |
| `repeats` | Number of times to repeat each sample |
| `batch_size` | Parallel batch size |
| `task_subjects` | Subjects to evaluate; evaluates all subjects if empty |
| `hf_revision` | HuggingFace dataset revision (branch, tag, or commit hash) |

CLI flags (e.g. `--num-samples`) act as the lowest-precedence defaults and are overridden by any value set in the suite file.

## Aggregation

Each composite suite can define a list of `SuiteAggregate` entries that compute summary metrics from its children's results. Aggregation runs bottom-up: child aggregates are computed before the parent's.

### SuiteAggregate fields

| Field | Description |
|---|---|
| `name` | Key used to store the result (e.g. `avg_math_accuracy`) |
| `metric` | Metric name to look up in each child's results. Can be a list — the first non-null value found per child is used. |
| `method` | How to combine values: `mean`, `median`, or `passthrough`. Defaults to `mean`. A Python callable is also accepted. |

**`mean` / `median`** — Reduces the named metric across all children that report a non-null value for it. Children with `null` or `NaN` values are excluded.

**`passthrough`** — Surfaces a metric from the single child whose name matches the aggregate's `name`, without any reduction. Useful for promoting a sub-suite score to the parent level alongside a cross-suite aggregate.

### Example: mixed aggregation methods

The following is based on `suites/olmo3_base_math.py`. It computes a mean across both tasks while also surfacing each task's score individually via `passthrough`:

```python
from eval_framework.suite import SuiteAggregate, TaskSuite

gsm8k = TaskSuite(
    name="gsm8k_olmo3_n8_v2",
    tasks="GSM8K_OLMES",
    repeats=8,
    temperature=0.6,
    top_p=0.6,
    max_tokens=512,
)

minerva = TaskSuite(
    name="minerva_math_olmes_n4_v2",
    tasks="MATHMinerva_OLMES",
    repeats=4,
    temperature=0.6,
    top_p=0.6,
    max_tokens=1024,
    num_fewshot=4,
)

suite = TaskSuite(
    name="olmo3_base_math",
    tasks=[gsm8k, minerva],
    aggregates=[
        # mean over both tasks; first non-null metric per child is used
        SuiteAggregate(
            name="Math Average Accuracy",
            metric=["Average Accuracy Completion", "Average Exact Match (Flex)"],
            method="mean",
        ),
        # surface each task's score at the suite level
        SuiteAggregate(
            name="gsm8k_olmo3_n8_v2",
            metric="Average Accuracy Completion",
            method="passthrough",
        ),
        SuiteAggregate(
            name="minerva_math_olmes_n4_v2",
            metric="Average Exact Match (Flex)",
            method="passthrough",
        ),
    ],
)
```

## Usage

```bash
uv run eval_framework \
    --llm-name 'eval_framework.llm.models.MyModel' \
    --task-suite suites/mathsuite.py \
    --output-dir ./eval_results
```

All tasks in a suite share the same Weights & Biases group (the top-level suite name). Each leaf task creates its own W&B run; each composite suite logs its aggregate metrics as a separate run. A few example suites are defined in the `examples/task_suites/` directory.

## Output Structure

```
outputs/
  <suite_name>/
    suite_aggregated_results.json   ← composite suite aggregates
    <task_output_dir>/
      aggregated_results.json       ← per-task results
      output.jsonl
      results.jsonl
      metadata.json
```

Composite suite aggregates are saved to `suite_aggregated_results.json` inside a directory named after the suite. Per-task outputs are written to their own subdirectories as usual.
