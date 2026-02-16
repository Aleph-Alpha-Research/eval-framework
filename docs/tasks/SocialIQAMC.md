# SocialIQAMC

````
NAME = SocialIQAMC
DATASET_PATH = allenai/social_i_qa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.social_iqa`

- File: [src/eval_framework/tasks/benchmarks/social_iqa.py](../../src/eval_framework/tasks/benchmarks/social_iqa.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/social_iqa.py)

- Link to dataset: [https://huggingface.co/datasets/allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "SocialIQAMC"`.
