# BalancedCOPA

````
NAME = BalancedCOPA
DATASET_PATH = pkavumba/balanced-copa
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.balancedcopa`

- File: [src/eval_framework/tasks/benchmarks/balancedcopa.py](../../src/eval_framework/tasks/benchmarks/balancedcopa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/balancedcopa.py)

- Link to dataset: [https://huggingface.co/datasets/pkavumba/balanced-copa](https://huggingface.co/datasets/pkavumba/balanced-copa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "BalancedCOPA"`.
