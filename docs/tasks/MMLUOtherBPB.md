# MMLUOtherBPB

````
NAME = MMLUOtherBPB
DATASET_PATH = cais/mmlu
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [BitsPerByteLoglikelihood]
SUBJECTS = ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine', 'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.mmlu`

- File: [src/eval_framework/tasks/benchmarks/mmlu.py](../../src/eval_framework/tasks/benchmarks/mmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mmlu.py)

- Link to dataset: [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLUOtherBPB"`.
