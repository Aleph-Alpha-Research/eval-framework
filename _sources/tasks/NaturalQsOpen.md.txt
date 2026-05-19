# NaturalQsOpen

````
NAME = NaturalQsOpen
DATASET_PATH = google-research-datasets/nq_open
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [DropF1ExactMatch]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.naturalqs_open`

- File: [src/eval_framework/tasks/benchmarks/naturalqs_open.py](../../src/eval_framework/tasks/benchmarks/naturalqs_open.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/naturalqs_open.py)

- Link to dataset: [https://huggingface.co/datasets/google-research-datasets/nq_open](https://huggingface.co/datasets/google-research-datasets/nq_open)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "NaturalQsOpen"`.
