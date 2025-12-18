# RenderableStructEval

````
NAME = RenderableStructEval
DATASET_PATH = TIGER-Lab/StructEval
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [RenderableStructMetric]
SUBJECTS = ['Convert Markdown to HTML', 'Convert React to HTML', 'Convert Vue to HTML', 'Text to HTML']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.struct_eval`

- File: [src/eval_framework/tasks/benchmarks/struct_eval.py](../../src/eval_framework/tasks/benchmarks/struct_eval.py) | [View on GitHub](https://github.com/Aleph-Alpha/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/struct_eval.py)

- Link to dataset: [https://huggingface.co/datasets/TIGER-Lab/StructEval](https://huggingface.co/datasets/TIGER-Lab/StructEval)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "RenderableStructEval"`.
