# BigCodeBench

````
NAME = BigCodeBench
DATASET_PATH = bigcode/bigcodebench
SAMPLE_SPLIT = v0.1.4
FEWSHOT_SPLIT = v0.1.4
RESPONSE_TYPE = COMPLETION
METRICS = [CodeExecutionPassAtOne]
SUBJECTS = ['original', 'calibrated']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.bigcodebench`

- File: [src/eval_framework/tasks/benchmarks/bigcodebench.py](../../src/eval_framework/tasks/benchmarks/bigcodebench.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/bigcodebench.py)

- Link to dataset: [https://huggingface.co/datasets/bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "BigCodeBench"`.

---

## BigCodeBench_OLMES

Variant that replicates **oe_eval** `bigcodebench:3shot::olmo3:v2` using eval_framework’s task and prompt structure.

| Setting | Value |
|--------|--------|
| **Task name** | `BigCodeBench_OLMES` |
| **Split** | v0.1.2 |
| **Fewshot** | 3 (from same split, random; current item excluded) |
| **Metric** | pass_at_1 |
| **Prompt** | oe_eval “complete” variant: instruction + `\n` + `` ``` `` + `complete_prompt` + `\n` |

**Recommended run settings** (for parity with oe_eval):

- `temperature=0.6`, `top_p=0.6`
- `repeats=5` (n=5 samples per problem for pass@1)
- `num_fewshot` is fixed to 3 by the task (config value ignored)

Pass@1 over the 5 samples can be computed by post-processing if needed, or run with `repeats=1` for a single sample per problem.
