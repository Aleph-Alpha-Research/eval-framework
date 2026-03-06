# NaturalQsOpenCloze

````
NAME = NaturalQsOpenCloze
DATASET_PATH = allenai/nq-gen2mc
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.naturalqs_open`

- File: [src/eval_framework/tasks/benchmarks/naturalqs_open.py](../../src/eval_framework/tasks/benchmarks/naturalqs_open.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/naturalqs_open.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/nq-gen2mc](https://huggingface.co/datasets/allenai/nq-gen2mc)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "NaturalQsOpenCloze"`.
