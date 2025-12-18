# MATH

````
NAME = MATH
DATASET_PATH = EleutherAI/hendrycks_math
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [MathReasoningCompletion, LanguageRawConsistencyChecker]
SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.math_reasoning`

- File: [src/eval_framework/tasks/benchmarks/math_reasoning.py](../../src/eval_framework/tasks/benchmarks/math_reasoning.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/math_reasoning.py)

- Link to dataset: [https://huggingface.co/datasets/EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "MATH"`.
