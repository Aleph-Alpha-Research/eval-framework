# SCIQEvalHarness

````
NAME = SCIQEvalHarness
DATASET_PATH = allenai/sciq
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.sciq`

- File: [src/eval_framework/tasks/benchmarks/sciq.py](../../src/eval_framework/tasks/benchmarks/sciq.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/sciq.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/sciq](https://huggingface.co/datasets/allenai/sciq)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "SCIQEvalHarness"`.
