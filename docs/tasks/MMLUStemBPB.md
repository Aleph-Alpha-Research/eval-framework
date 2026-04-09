# MMLUStemBPB

````
NAME = MMLUStemBPB
DATASET_PATH = cais/mmlu
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = dev
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [BitsPerByteLoglikelihood]
SUBJECTS = ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.mmlu`

- File: [src/eval_framework/tasks/benchmarks/mmlu.py](../../src/eval_framework/tasks/benchmarks/mmlu.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/mmlu.py)

- Link to dataset: [https://huggingface.co/datasets/cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MMLUStemBPB"`.
