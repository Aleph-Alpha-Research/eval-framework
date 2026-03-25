# DropMC_OLMES

````
NAME = DropMC_OLMES
DATASET_PATH = allenai/drop-gen2mc
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.drop`

- File: [src/eval_framework/tasks/benchmarks/drop.py](../../src/eval_framework/tasks/benchmarks/drop.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/drop.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/drop-gen2mc](https://huggingface.co/datasets/allenai/drop-gen2mc)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "DropMC_OLMES"`.
