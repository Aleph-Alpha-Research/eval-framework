# MATHLvl5

````
NAME = MATHLvl5
DATASET_PATH = EleutherAI/hendrycks_math
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.math_reasoning](eval_framework.tasks.benchmarks.math_reasoning)

- File: [src/eval_framework/tasks/benchmarks/math_reasoning.py](../../src/eval_framework/tasks/benchmarks/math_reasoning.py)

- Link to dataset: [https://huggingface.co/datasets/EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/generate_task_docs.py --add-prompt-examples --only-tasks "MATHLvl5"`.
