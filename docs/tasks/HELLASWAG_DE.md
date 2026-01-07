# HELLASWAG_DE

````
NAME = HELLASWAG_DE
DATASET_PATH = LeoLM/HellaSwag_de
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.DEU: 'German'>
````

- Module: `eval_framework.tasks.benchmarks.hellaswag_de`

- File: [src/eval_framework/tasks/benchmarks/hellaswag_de.py](../../src/eval_framework/tasks/benchmarks/hellaswag_de.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/hellaswag_de.py)

- Link to dataset: [https://huggingface.co/datasets/LeoLM/HellaSwag_de](https://huggingface.co/datasets/LeoLM/HellaSwag_de)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "HELLASWAG_DE"`.
