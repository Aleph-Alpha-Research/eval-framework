# LabBenchMC

````
NAME = LabBenchMC
DATASET_PATH = futurehouse/lab-bench
SAMPLE_SPLIT = train
FEWSHOT_SPLIT = train
RESPONSE_TYPE = LOGLIKELIHOODS
METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, AccuracyCompletion]
SUBJECTS = ['CloningScenarios', 'DbQA', 'FigQA', 'LitQA2', 'ProtocolQA', 'SeqQA', 'SuppQA', 'TableQA']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.lab_bench`

- File: [src/eval_framework/tasks/benchmarks/lab_bench.py](../../src/eval_framework/tasks/benchmarks/lab_bench.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/lab_bench.py)

- Link to dataset: [https://huggingface.co/datasets/futurehouse/lab-bench](https://huggingface.co/datasets/futurehouse/lab-bench)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "LabBenchMC"`.
