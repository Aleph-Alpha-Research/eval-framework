# MedQAMC

````
NAME = MedQAMC
DATASET_PATH = davidheineman/medqa-en
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.medqa`

- File: [src/eval_framework/tasks/benchmarks/medqa.py](../../src/eval_framework/tasks/benchmarks/medqa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/medqa.py)

- Link to dataset: [https://huggingface.co/datasets/davidheineman/medqa-en](https://huggingface.co/datasets/davidheineman/medqa-en)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MedQAMC"`.
