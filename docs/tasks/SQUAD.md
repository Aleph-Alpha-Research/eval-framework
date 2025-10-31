# SQUAD

````
NAME = SQUAD
DATASET_PATH = rajpurkar/squad
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion, F1]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: [eval_framework.tasks.benchmarks.squad](eval_framework.tasks.benchmarks.squad)

- File: [src/eval_framework/tasks/benchmarks/squad.py](../../src/eval_framework/tasks/benchmarks/squad.py)

- Link to dataset: [https://huggingface.co/datasets/rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)

- `train` has 87599 samples

- `validation` has 10570 samples

## Example prompt with ConcatFormatter (1-shot)

````
"Given the following context, answer the question.

Context:
The Houston–The Woodlands–Sugar Land metropolitan area is served by one public television station and two public radio stations. KUHT (HoustonPBS) is a PBS member station and is the first public television station in the United States. Houston Public Radio is listener-funded and comprises two NPR member stations: KUHF (KUHF News) and KUHA (Classical 91.7). KUHF is news/talk radio and KUHA is a classical music station. The University of Houston System owns and holds broadcasting licenses to KUHT, KUHF, and KUHA. The stations broadcast from the Melcher Center for Public Broadcasting, located on the campus of the University of Houston.

Question:
What stations comprise Houston Public Radio?
KUHF (KUHF News) and KUHA (Classical 91.7)

Given the following context, answer the question.

Context:
In cpDNA, there are several A → G deamination gradients. DNA becomes susceptible to deamination events when it is single stranded. When replication forks form, the strand not being copied is single stranded, and thus at risk for A → G deamination. Therefore, gradients in deamination indicate that replication forks were most likely present and the direction that they initially opened (the highest gradient is most likely nearest the start site because it was single stranded for the longest amount of time). This mechanism is still the leading theory today; however, a second theory suggests that most cpDNA is actually linear and replicates through homologous recombination. It further contends that only a minority of the genetic material is kept in circular chromosomes while the rest is in branched, linear, or other complex structures.

Question:
How does the secondary theory say most cpDNA is structured?"
````

## Example prompt with Llama3Formatter (1-shot)

````
"<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Given the following context, answer the question.

Context:
The Houston–The Woodlands–Sugar Land metropolitan area is served by one public television station and two public radio stations. KUHT (HoustonPBS) is a PBS member station and is the first public television station in the United States. Houston Public Radio is listener-funded and comprises two NPR member stations: KUHF (KUHF News) and KUHA (Classical 91.7). KUHF is news/talk radio and KUHA is a classical music station. The University of Houston System owns and holds broadcasting licenses to KUHT, KUHF, and KUHA. The stations broadcast from the Melcher Center for Public Broadcasting, located on the campus of the University of Houston.

Question:
What stations comprise Houston Public Radio?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

KUHF (KUHF News) and KUHA (Classical 91.7)<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following context, answer the question.

Context:
In cpDNA, there are several A → G deamination gradients. DNA becomes susceptible to deamination events when it is single stranded. When replication forks form, the strand not being copied is single stranded, and thus at risk for A → G deamination. Therefore, gradients in deamination indicate that replication forks were most likely present and the direction that they initially opened (the highest gradient is most likely nearest the start site because it was single stranded for the longest amount of time). This mechanism is still the leading theory today; however, a second theory suggests that most cpDNA is actually linear and replicates through homologous recombination. It further contends that only a minority of the genetic material is kept in circular chromosomes while the rest is in branched, linear, or other complex structures.

Question:
How does the secondary theory say most cpDNA is structured?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
````

## Possible completions:

````
None
````

## Ground truth:

````
- "linear"
- "linear"
- "linear"
````
