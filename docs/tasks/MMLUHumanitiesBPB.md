# MMLUHumanitiesBPB

````
NAME = MMLUHumanitiesBPB
DATASET_PATH = cais/mmlu
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [BitsPerByteLoglikelihood]
SUBJECTS = ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.mmlu`

- File: [src/eval_framework/tasks/benchmarks/mmlu.py](../../src/eval_framework/tasks/benchmarks/mmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mmlu.py)

- Link to dataset: [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLUHumanitiesBPB"`.
