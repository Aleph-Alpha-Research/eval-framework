# SQuAD2MA

````
NAME = SQuAD2MA
DATASET_PATH = rajpurkar/squad_v2
SAMPLE_SPLIT = validation
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [AccuracyCompletion, F1, F1SquadNormalized]
SUBJECTS = ['no_subject']
LANGUAGE = <Language.ENG: 'English'>
````

- Module: `eval_framework.tasks.benchmarks.squad`

- File: [src/eval_framework/tasks/benchmarks/squad.py](../../src/eval_framework/tasks/benchmarks/squad.py) | [View on GitHub](https://github.com/Aleph-Alpha-Research/eval-framework/blob/main/src/eval_framework/tasks/benchmarks/squad.py)

- Link to dataset: [https://huggingface.co/datasets/rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2)

- `train` has 130319 samples

- `validation` has 11873 samples

## Example prompt with ConcatFormatter (1-shot)

````
"You are a helpful assistant and will answer the user's questions carefully, logically, accurately and well-reasoned.
Use the given context to answer the question faithfully. Answer only if the answer is present in the given context, otherwise respond with 'unanswerable' if the answer is not present in the context.

Context:
The subsequent 2005 elections were internationally regarded as the most free and fair in Liberian history. Ellen Johnson Sirleaf, a Harvard-trained economist and former Minister of Finance, was elected as the first female president in Africa. Upon her inauguration, Sirleaf requested the extradition of Taylor from Nigeria and transferred him to the SCSL for trial in The Hague. In 2006, the government established a Truth and Reconciliation Commission to address the causes and crimes of the civil war.

Question:
Movement For democracy in Liberia requested the extradition of whom"
Answer: Taylor from Nigeria

Context:
Pattern recognition receptors are proteins used by nearly all organisms to identify molecules associated with pathogens. Antimicrobial peptides called defensins are an evolutionarily conserved component of the innate immune response found in all animals and plants, and represent the main form of invertebrate systemic immunity. The complement system and phagocytic cells are also used by most forms of invertebrate life. Ribonucleases and the RNA interference pathway are conserved across all eukaryotes, and are thought to play a role in the immune response to viruses.

Question:
What pathway that plays a role in immune response to viruses is present in all eukaryotes?
Answer:"
````

## Example prompt with Llama3Formatter (1-shot)

````
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant and will answer the user's questions carefully, logically, accurately and well-reasoned.
Use the given context to answer the question faithfully. Answer only if the answer is present in the given context, otherwise respond with 'unanswerable' if the answer is not present in the context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
The subsequent 2005 elections were internationally regarded as the most free and fair in Liberian history. Ellen Johnson Sirleaf, a Harvard-trained economist and former Minister of Finance, was elected as the first female president in Africa. Upon her inauguration, Sirleaf requested the extradition of Taylor from Nigeria and transferred him to the SCSL for trial in The Hague. In 2006, the government established a Truth and Reconciliation Commission to address the causes and crimes of the civil war.

Question:
Movement For democracy in Liberia requested the extradition of whom"
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Taylor from Nigeria<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
Pattern recognition receptors are proteins used by nearly all organisms to identify molecules associated with pathogens. Antimicrobial peptides called defensins are an evolutionarily conserved component of the innate immune response found in all animals and plants, and represent the main form of invertebrate systemic immunity. The complement system and phagocytic cells are also used by most forms of invertebrate life. Ribonucleases and the RNA interference pathway are conserved across all eukaryotes, and are thought to play a role in the immune response to viruses.

Question:
What pathway that plays a role in immune response to viruses is present in all eukaryotes?
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
````

## Possible completions:

````
None
````

## Ground truth:

````
- " RNA interference pathway"
- " RNA interference"
- " RNA interference pathway"
````
