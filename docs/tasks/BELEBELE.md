# BELEBELE

````
NAME = BELEBELE
DATASET_PATH = facebook/belebele
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['eng_Latn']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.belebele](eval_framework.tasks.benchmarks.belebele)

- File: [src/eval_framework/tasks/benchmarks/belebele.py](../../src/eval_framework/tasks/benchmarks/belebele.py)

- Link to dataset: [https://huggingface.co/datasets/facebook/belebele](https://huggingface.co/datasets/facebook/belebele)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "BELEBELE"`.
