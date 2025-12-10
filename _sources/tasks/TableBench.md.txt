# TableBench

````
NAME = TableBench
DATASET_PATH = Multilingual-Multimodal-NLP/TableBench
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = test
RESPONSE_TYPE = COMPLETION
METRICS = [ROUGE_L]
SUBJECTS = [('PoT', 'NumericalReasoning'), ('PoT', 'DataAnalysis'), ('PoT', 'FactChecking'), ('SCoT', 'NumericalReasoning'), ('SCoT', 'DataAnalysis'), ('SCoT', 'FactChecking'), ('TCoT', 'NumericalReasoning'), ('TCoT', 'DataAnalysis'), ('TCoT', 'FactChecking')]
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.tablebench](eval_framework.tasks.benchmarks.tablebench)

- File: [src/eval_framework/tasks/benchmarks/tablebench.py](../../src/eval_framework/tasks/benchmarks/tablebench.py)

- Link to dataset: [https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "TableBench"`.
