# MultiPLEMBPPJs

````
NAME = MultiPLEMBPPJs
DATASET_PATH = nuprl/MultiPL-E
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [MultiPLECodeAssertion]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.multipl_e`

- File: [src/eval_framework/tasks/benchmarks/multipl_e.py](../../src/eval_framework/tasks/benchmarks/multipl_e.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/multipl_e.py)

- Link to dataset: [https://huggingface.co/datasets/nuprl/MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MultiPLEMBPPJs"`.
