# CommonsenseQACloze

````
NAME = CommonsenseQACloze
DATASET_PATH = tau/commonsense_qa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = validation
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.csqa`

- File: [src/eval_framework/tasks/benchmarks/csqa.py](../../src/eval_framework/tasks/benchmarks/csqa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/csqa.py)

- Link to dataset: [https://huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "CommonsenseQACloze"`.
