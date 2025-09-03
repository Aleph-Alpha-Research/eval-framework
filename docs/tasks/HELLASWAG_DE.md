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

- Module: [eval_framework.tasks.benchmarks.hellaswag_de](eval_framework.tasks.benchmarks.hellaswag_de)

- File: [src/eval_framework/tasks/benchmarks/hellaswag_de.py](../../src/eval_framework/tasks/benchmarks/hellaswag_de.py)

- Link to dataset: [https://huggingface.co/datasets/LeoLM/HellaSwag_de](https://huggingface.co/datasets/LeoLM/HellaSwag_de)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "HELLASWAG_DE"`.
