# IFEvalDe

````
NAME = IFEvalDe
DATASET_PATH = jzhang86/de_ifeval
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [IFEvalMetric]
SUBJECTS = ['no_subject']
LANGUAGE = {'no_subject': <Language.DEU: 'German'>}
````

- Module: `eval_framework.tasks.benchmarks.ifeval`

- File: [src/eval_framework/tasks/benchmarks/ifeval.py](../../src/eval_framework/tasks/benchmarks/ifeval.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/ifeval.py)

- Link to dataset: [https://huggingface.co/datasets/jzhang86/de_ifeval](https://huggingface.co/datasets/jzhang86/de_ifeval)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "IFEvalDe"`.
