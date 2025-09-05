# IFEvalFiSv

````
NAME = IFEvalFiSv
DATASET_PATH = LumiOpen/ifeval_mt
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [IFEvalMetric]
SUBJECTS = ['fi', 'sv']
LANGUAGE = {'fi': <Language.FIN: 'Finnish'>, 'sv': <Language.SWE: 'Swedish'>}
````

- Module: [eval_framework.tasks.benchmarks.ifeval](eval_framework.tasks.benchmarks.ifeval)

- File: [src/eval_framework/tasks/benchmarks/ifeval.py](../../src/eval_framework/tasks/benchmarks/ifeval.py)

- Link to dataset: [https://huggingface.co/datasets/LumiOpen/ifeval_mt](https://huggingface.co/datasets/LumiOpen/ifeval_mt)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run python src/eval_framework/utils/generate_task_docs.py --add-prompt-examples --only-tasks "IFEvalFiSv"`.
