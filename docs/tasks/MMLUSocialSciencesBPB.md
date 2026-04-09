# MMLUSocialSciencesBPB

````
NAME = MMLUSocialSciencesBPB
DATASET_PATH = cais/mmlu
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [BitsPerByteLoglikelihood]
SUBJECTS = ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.mmlu`

- File: [src/eval_framework/tasks/benchmarks/mmlu.py](../../src/eval_framework/tasks/benchmarks/mmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mmlu.py)

- Link to dataset: [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLUSocialSciencesBPB"`.
