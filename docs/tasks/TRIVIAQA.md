# TRIVIAQA

````
NAME = TRIVIAQA
DATASET_PATH = mandarjoshi/trivia_qa
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion, F1]
SUBJECTS = ['rc.wikipedia.nocontext']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.triviaqa`

- File: [src/eval_framework/tasks/benchmarks/triviaqa.py](../../src/eval_framework/tasks/benchmarks/triviaqa.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/triviaqa.py)

- Link to dataset: [https://huggingface.co/datasets/mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "TRIVIAQA"`.
