# SPHYR

````
NAME = SPHYR
DATASET_PATH = philippds/SPhyR
SAMPLE_SPLIT = test
FEWSHOT_SPLIT =
RESPONSE_TYPE = COMPLETION
METRICS = [GridDifference]
SUBJECTS = ['1_random_cell_easy', '5_random_cell_easy', '10_random_cell_easy', '1_random_row_easy', '3_random_row_easy', '1_random_column_easy', '3_random_column_easy', 'full_easy', '1_random_cell_hard', '5_random_cell_hard', '10_random_cell_hard', '1_random_row_hard', '3_random_row_hard', '1_random_column_hard', '3_random_column_hard', 'full_hard']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.sphyr`

- File: [src/eval_framework/tasks/benchmarks/sphyr.py](../../src/eval_framework/tasks/benchmarks/sphyr.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/sphyr.py)

- Link to dataset: [https://huggingface.co/datasets/philippds/SPhyR](https://huggingface.co/datasets/philippds/SPhyR)

More detailed documentation, with prompt examples and ground truth completions, can be generated with `uv run -m eval_framework.utils.generate_task_docs --add-prompt-examples --only-tasks "SPHYR"`.
