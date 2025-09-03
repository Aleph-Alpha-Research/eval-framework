# CASEHOLD

````
NAME = CASEHOLD
DATASET_PATH = lex_glue
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['case_hold']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.casehold](eval_framework.tasks.benchmarks.casehold)

- File: [src/eval_framework/tasks/benchmarks/casehold.py](../../src/eval_framework/tasks/benchmarks/casehold.py)

- Link to dataset: [https://huggingface.co/datasets/lex_glue](https://huggingface.co/datasets/lex_glue)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "CASEHOLD"`.
