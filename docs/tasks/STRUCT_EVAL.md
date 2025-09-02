# STRUCT_EVAL

````
NAME = STRUCT_EVAL
DATASET_PATH = TIGER-Lab/StructEval
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [StructMetric]
SUBJECTS = ['CSV to YAML', 'JSON to XML', 'JSON to CSV', 'XML to JSON', 'XML to YAML', 'Text to XML', 'Text to YAML', 'Text to TOML', 'YAML to JSON', 'TOML to JSON', 'Text to CSV', 'YAML to XML', 'JSON to YAML', 'TOML to YAML', 'YAML to CSV', 'CSV to JSON', 'CSV to XML', 'Text to JSON', 'XML to CSV']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.struct_eval](eval_framework.tasks.benchmarks.struct_eval)

- File: [src/eval_framework/tasks/benchmarks/struct_eval.py](../../src/eval_framework/tasks/benchmarks/struct_eval.py)

- Link to dataset: [https://huggingface.co/datasets/TIGER-Lab/StructEval](https://huggingface.co/datasets/TIGER-Lab/StructEval)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "STRUCT_EVAL"`.
