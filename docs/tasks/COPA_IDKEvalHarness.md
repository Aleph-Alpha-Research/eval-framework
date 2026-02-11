# COPA_IDKEvalHarness

````
NAME = COPA_IDKEvalHarness
DATASET_PATH = aps/super_glue
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, ConfidenceWeightedAccuracy, DistributionalCorrectnessScore, TernaryScore]
SUBJECTS = ['copa']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.copa`

- File: [src/eval_framework/tasks/benchmarks/copa.py](../../src/eval_framework/tasks/benchmarks/copa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/copa.py)

- Link to dataset: [https://huggingface.co/datasets/aps/super_glue](https://huggingface.co/datasets/aps/super_glue)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "COPA_IDKEvalHarness"`.
